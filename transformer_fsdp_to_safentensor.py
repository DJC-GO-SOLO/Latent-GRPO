import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge a verl FSDP actor checkpoint into a HuggingFace Transformers "
            "directory with safetensors weights."
        )
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Actor checkpoint directory containing model_world_size_*_rank_*.pt files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where the merged HuggingFace model will be saved.",
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default=None,
        help=(
            "Directory or HuggingFace model id providing config/tokenizer files. "
            "Defaults to --local_dir."
        ),
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype used when saving merged tensors.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def find_world_size(local_dir: Path) -> int:
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            return int(match.group(1))
    raise FileNotFoundError(
        f"No model shard matching model_world_size_*_rank_0.pt found in {local_dir}"
    )


def get_auto_model_class(config):
    architectures = getattr(config, "architectures", None) or []
    if architectures and "ForTokenClassification" in architectures[0]:
        return AutoModelForTokenClassification
    if architectures and "ForCausalLM" in architectures[0]:
        return AutoModelForCausalLM
    raise NotImplementedError(f"Unsupported model architecture: {architectures}")


def main():
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser()
    hf_path = args.hf_path or str(local_dir)
    output_path = Path(args.output_path).expanduser()
    save_dtype = get_torch_dtype(args.torch_dtype)

    if not local_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {local_dir}")

    world_size = find_world_size(local_dir)
    rank = 0

    state_dict = torch.load(
        local_dir / f"model_world_size_{world_size}_rank_{rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    if not isinstance(weight, DTensor):
        raise TypeError(
            f"Expected DTensor checkpoint shard, but key {pivot_key!r} has type {type(weight)}"
        )

    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (
        ("fsdp",),
    ), f"Unsupported mesh_dim_names {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing model shards with {total_shards} {mesh_shape} in total")

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = local_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        futures = [executor.submit(process_one_shard, rank) for rank in range(1, total_shards)]
        for future in futures:
            future.result()

    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except KeyError as exc:
                raise KeyError(f"Missing key {key!r} in one checkpoint shard") from exc
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.to(save_dtype))
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == "dp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.to(save_dtype)

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    print("Writing to local disk")
    config = AutoConfig.from_pretrained(hf_path)
    auto_model = get_auto_model_class(config)
    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=save_dtype)
    model.to_empty(device="cpu")

    print(f"Saving model to {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    tokenizer.save_pretrained(str(output_path))
    model.save_pretrained(str(output_path), state_dict=state_dict, safe_serialization=True)


if __name__ == "__main__":
    main()