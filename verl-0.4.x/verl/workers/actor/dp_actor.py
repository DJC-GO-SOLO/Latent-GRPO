# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits, logprobs_from_logits_topk_gumbel, top_p_renorm_logprobs
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs, \
    ulysses_pad_and_slice_inputs_3d
from verl.workers.actor import BasePPOActor
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, top_p,  calculate_entropy=False, add_noise_dirichlet=False,
                             add_noise_gumbel_softmax=True) -> Tuple[
        torch.Tensor, torch.Tensor]:
        def safe_lookup_embeddings(fsdp_wrapped_module, input_ids, target_device=None, target_dtype=None):
            """Look up embeddings safely when the module is wrapped by FSDP."""
            embed = fsdp_wrapped_module.get_input_embeddings()

            ctx = FSDP.summon_full_params(fsdp_wrapped_module, writeback=False, with_grads=False)
            with ctx:
                w = embed.weight
                _input_ids = input_ids.to(w.device).clone()
                mask = (_input_ids < 0)
                if mask.any():
                    _input_ids[mask] = 0
                
                embs = embed(_input_ids)
                
                if mask.any():
                    embs[mask] = 0
            if target_dtype is not None and embs.dtype != target_dtype:
                embs = embs.to(dtype=target_dtype)
            if target_device is not None and embs.device != target_device:
                embs = embs.to(target_device)
            return embs

        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]],
                                                    dim=0)
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"] # whole sentences B*n:I+O
            rollout_topk_ids = micro_batch["rollout_topk_ids"]
            rollout_topk_gumbels = micro_batch["rollout_topk_gumbels"]
            batch_size, seqlen = input_ids.shape
            k_num = rollout_topk_gumbels.size(-1)
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            gumbel_temperature = micro_batch["gumbel_temperature"][0].item()
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                topk_ids_rmpad, _, *_ = unpad_input(rollout_topk_ids,
                                                    attention_mask)  # topk_ids_rmpad (total_nnz, K)
                topk_gumbels_rmpad, _, *_ = unpad_input(rollout_topk_gumbels,
                                                        attention_mask)  # topk_gumbels_rmpad (total_nnz, K)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                        1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                hard_token_mask = (topk_ids_rmpad[:, 1:] == -100).all(dim=-1) # (total_nnz,)
                
                # Step 1: Always lookup the first candidate (this is the only candidate for hard tokens)
                # We use safe_lookup_embeddings which now handles -100 safely.
                # However, for the first candidate, it shouldn't be -100 anyway.
                all_first_embs = safe_lookup_embeddings(
                    self.actor_module,
                    topk_ids_rmpad[:, :1],
                    target_device=topk_gumbels_rmpad.device,
                    target_dtype=topk_gumbels_rmpad.dtype
                ).squeeze(1) # (total_nnz, dim)
                # Step 2: Handle soft tokens
                if (~hard_token_mask).any():
                    # Soft tokens need all K candidates
                    # We compute the weighted sum only for indices where they are soft tokens
                    # But for simplicity and batching, we can still compute it for all, 
                    # as long as we use the correct mask for the gumbel noise.
                    mask_expanded = hard_token_mask.unsqueeze(-1) # (total_nnz, 1)
                    
                    # Prepare Gumbel weights
                    masked_gumbels = topk_gumbels_rmpad.clone()
                    masked_gumbels[:, 1:] = masked_gumbels[:, 1:].masked_fill(mask_expanded, -torch.inf)
                    gumbel_y = torch.softmax(masked_gumbels / gumbel_temperature, dim=-1).to(topk_gumbels_rmpad.dtype)
                    
                    # Full lookup (only if there are soft tokens)
                    topk_embs = safe_lookup_embeddings(
                        self.actor_module,
                        topk_ids_rmpad,
                        target_device=topk_gumbels_rmpad.device,
                        target_dtype=topk_gumbels_rmpad.dtype
                    ) # (total_nnz, K, dim)
                    
                    soft_embs = torch.sum(gumbel_y.unsqueeze(-1).float() * topk_embs.float(), dim=1).to(torch.bfloat16) # (total_nnz, dim)
                    
                    # Output is hard_embs for hard tokens, soft_embs for soft tokens
                    topk_embs_final = torch.where(mask_expanded, all_first_embs, soft_embs)
                else:
                    # All are hard tokens
                    topk_embs_final = all_first_embs

                topk_embs = topk_embs_final
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                topk_gumbels_rmpad_rolled = torch.roll(topk_gumbels_rmpad, shifts=-1, dims=0)  # (total_nnz, k)
                topk_ids_rmpad_rolled = torch.roll(topk_ids_rmpad, shifts=-1, dims=0)  # (total_nnz, k)
                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    topk_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs_3d(
                        rollout_topk_ids,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    topk_gumbels_rmpad, _, _ = ulysses_pad_and_slice_inputs_3d(
                        rollout_topk_gumbels,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                # Model Forward
                output = self.actor_module(
                    # input_ids=input_ids_rmpad,
                    inputs_embeds=topk_embs.unsqueeze(0).detach(),#
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    # ================================================================
                    # Process advantages for Gumbel gating.
                    # ================================================================
                    advantages = micro_batch.get("advantages", None)
                    current_advantages = None
                    if advantages is not None:
                        # advantages: (B, R) - Response only
                        # input_ids: (B, I+R) - Prompt + Response
                        # 1. Left Pad with 0s for Prompt portion
                        prompt_len = seqlen - advantages.size(1)  # I = (I+R) - R
                        if prompt_len > 0:
                            # F.pad: (left_pad, right_pad)
                            full_advantages = torch.nn.functional.pad(advantages, (prompt_len, 0), value=0.0)
                        else:
                            full_advantages = advantages
                        # full_advantages: (B, I+R)

                        # 2. Unpad to match input_ids_rmpad structure
                        adv_expanded = full_advantages.unsqueeze(-1)  # (B, T, 1)
                        adv_rmpad, _, *_ = unpad_input(adv_expanded, attention_mask)
                        # adv_rmpad: (total_nnz, 1)
                        
                        # 3. Roll to align with input_ids_rmpad_rolled (Next Token)
                        current_advantages = torch.roll(adv_rmpad, shifts=-1, dims=0)
                        # current_advantages: (total_nnz, 1)
                    # ================================================================

                    if add_noise_gumbel_softmax:
                        log_probs = logprobs_from_logits_topk_gumbel(
                            logits=logits_rmpad,
                            rollout_topk_ids=topk_ids_rmpad_rolled,
                            rollout_topk_gumbels=topk_gumbels_rmpad_rolled,
                            labels=input_ids_rmpad_rolled,
                            top_p=top_p,
                            temperature=temperature,
                            inplace_backward=inplace_backward,
                            advantages=current_advantages,
                        )
                    else:
                        full_logprobs = top_p_renorm_logprobs(logits_rmpad / temperature, top_p)
                        log_probs = full_logprobs.gather(-1, input_ids_rmpad_rolled.unsqueeze(-1)).squeeze(-1)

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                safe_gather_ids = topk_ids_rmpad_rolled.clone()
                safe_gather_ids[safe_gather_ids == -100] = 0

                current_topk_logits_rmpad = logits_rmpad.gather(-1, safe_gather_ids)
                full_current_topk_logits = pad_input(
                    hidden_states=current_topk_logits_rmpad, 
                    indices=indices, 
                    batch=batch_size, 
                    seqlen=seqlen
                ) # (B, Seq, K)
                # logits_rmpad.div_(temperature)
                latent_probs = torch.softmax(logits_rmpad / temperature, dim=-1)
                topk_original_probs, topk_indices = torch.topk(latent_probs, k=10, dim=-1)
                full_topk_probs = pad_input(
                    hidden_states=topk_original_probs,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_topk_indices = pad_input(
                    hidden_states=topk_indices,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_topk_probs = full_topk_probs[:, :-1, :].contiguous()
                full_topk_indices = full_topk_indices[:, :-1, :].contiguous()



                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
                full_current_topk_logits = full_current_topk_logits.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1: -1]
                    entropy = output.entropy[:, -response_length - 1: -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1: -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
            return entropy, log_probs, full_topk_probs, full_topk_indices, full_current_topk_logits
    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            logger.warning("Rank %s grad_norm is not finite: %s", torch.distributed.get_rank(), grad_norm)
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        top_p = data.meta_info["top_p"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        add_noise_dirichlet = data.meta_info['add_noise_dirichlet']
        add_noise_gumbel_softmax = data.meta_info['add_noise_gumbel_softmax']

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "rollout_topk_ids",
                       "rollout_topk_gumbels", "gumbel_temperature"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        topk_probs_lst = []
        topk_ids_lst = []
        entropy_lst = []
        topk_logits_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_porbs, topk_indices, cur_topk_logits = self._forward_micro_batch(micro_batch, temperature=temperature, top_p=top_p,
                                                               calculate_entropy=calculate_entropy,
                                                               add_noise_dirichlet=add_noise_dirichlet,
                                                               add_noise_gumbel_softmax=add_noise_gumbel_softmax,)
            log_probs_lst.append(log_probs)
            topk_probs_lst.append(topk_porbs)
            topk_ids_lst.append(topk_indices)
            topk_logits_lst.append(cur_topk_logits)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        topk_probs = torch.concat(topk_probs_lst, dim=0)
        topk_ids = torch.concat(topk_ids_lst, dim=0)
        topk_logits = torch.concat(topk_ids_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
        return log_probs, entropys, topk_probs, topk_ids, topk_logits

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        top_p = data.meta_info["top_p"]
        multi_turn = data.meta_info.get("multi_turn", False)

        add_noise_dirichlet = data.meta_info['add_noise_dirichlet']
        add_noise_gumbel_softmax = data.meta_info['add_noise_gumbel_softmax']
        exclude_overlong_samples_from_advantage = data.meta_info.get("exclude_overlong_samples_from_advantage", False)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages",
                       "rollout_topk_ids", "rollout_topk_gumbels", "gumbel_temperature","token_level_rewards"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob,_,_,current_logits = self._forward_micro_batch(micro_batch=data, temperature=temperature, top_p=top_p,
                                                                  calculate_entropy=calculate_entropy,
                                                                  add_noise_dirichlet=add_noise_dirichlet,
                                                                  add_noise_gumbel_softmax=add_noise_gumbel_softmax,)

                    # Zero max-length sample advantages only when they are kept during GRPO advantage computation.
                    if not exclude_overlong_samples_from_advantage:
                        cur_response_length = data["attention_mask"][:, -response_length:].sum(dim=-1)
                        is_clipped = cur_response_length == response_length
                        if "advantages" in data:
                            data["advantages"][is_clipped] = 0
                    
                    neg_adv_weight = self.config.get("neg_adv_weight", 1.0)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        neg_adv_weight=neg_adv_weight,
                        loss_agg_mode=loss_agg_mode,
                    )
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob,
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item() if hasattr(pg_clipfrac, 'item') else pg_clipfrac,
                        "actor/ppo_kl": ppo_kl.detach().item() if hasattr(ppo_kl, 'item') else ppo_kl,
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item() if hasattr(pg_clipfrac_lower, 'item') else pg_clipfrac_lower,
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics