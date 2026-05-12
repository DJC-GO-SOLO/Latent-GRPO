<div align="center">
  <img src="figs/latent_reasoning_logo.svg" alt="Latent-GRPO logo" width="480"/>

  <h2>Latent-GRPO: Group Relative Policy Optimization for Latent Reasoning</h2>

  <p>
    <a href="https://arxiv.org/abs/2604.27998"><img src="https://img.shields.io/badge/arXiv-2604.27998-b31b1b.svg" alt="arXiv"/></a>
    <a href="https://huggingface.co/datasets/DJCheng/Latent-GRPO-Data"><img src="https://img.shields.io/badge/Data-HuggingFace-yellow.svg" alt="Hugging Face Data"/></a>
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg" alt="PyTorch"/>
  </p>

  <p><strong>Official implementation of Latent-GRPO, an RL-stage framework for optimizing vocabulary-space latent reasoning.</strong></p>
</div>

---

## News

- **Latent-GRPO paper**: The paper is available on arXiv: [Latent-GRPO](https://arxiv.org/abs/2604.27998).
- **Code release**: This repository provides the data preparation scripts, customized SGLang rollout engine, verl-based Latent-GRPO training pipeline, and evaluation scripts used by Latent-GRPO.

## Overview

Latent-GRPO extends latent-chain reasoning from supervised fine-tuning to reinforcement learning. Instead of training a standard discrete-token CoT policy from scratch, it performs GRPO-style RL on models that have already learned to reason with vocabulary-space latent chains.

This repository includes:

- A customized SGLang inference engine with latent-token rollout support.
- A modified verl-0.4.x training stack for Latent-GRPO.
- Low- and high-difficulty training scripts.
- Data preprocessing scripts for converting jsonl data into verl-compatible parquet files.
- SGLang-based evaluation scripts for low- and high-difficulty reasoning benchmarks.

## Repository Structure

```text
Latent-GRPO/
├── Latent-GRPO-gsm8k-llama3.sh
├── Latent-GRPO-math500-qwen.sh
├── data_preprocess_code/
│   ├── gsm8k_aug.py
│   └── math500_aug.py
├── eval/
│   ├── eval_low_tasks_sglang.py
│   └── eval_high_tasks_sglang.py
├── figs/
│   └── latent_reasoning_logo.svg
├── requirements.txt
├── sglang_latent_reasoning_pkg/
├── transformer_fsdp_to_safentensor.py
└── verl-0.4.x/
```

Only core files and directories are shown here. The customized SGLang and verl directories contain many internal modules inherited from their upstream projects.

## Data Preparation

There are two ways to prepare data.

### Option 1: Download Released Data

Download the released data from Hugging Face:

- [DJCheng/Latent-GRPO-Data](https://huggingface.co/datasets/DJCheng/Latent-GRPO-Data)

Place the files under the repository-level `data/` directory:

```bash
mkdir -p data
```

The released training scripts expect the following parquet files by default:

| Setting | Training file | Validation file |
| --- | --- | --- |
| Low-difficulty GSM8K setting | `GSM8k-Aug-oss-dup-all.parquet` | `GSM8k-Aug-test.parquet` |
| High-difficulty math setting | `DAPO-Math-17k-en-train.parquet` | `Math-500-test.parquet` |

### Option 2: Process Your Own Data

You can also use the scripts under `data_preprocess_code/` to convert jsonl files into the parquet format expected by verl:

```bash
python data_preprocess_code/gsm8k_aug.py \
  --input_path data/GSM8k-Aug-train.jsonl \
  --output_path data/GSM8k-Aug-oss-dup-all.parquet \
  --split train

python data_preprocess_code/math500_aug.py \
  --input_path data/DAPO-Math-17k-en-train.jsonl \
  --output_path data/DAPO-Math-17k-en-train.parquet \
  --split train
```

The preprocessing scripts expect jsonl records to provide `problem` and `answer` fields. They convert each example into the prompt, reward-model, and metadata schema used by verl.

> [!IMPORTANT]
> The training pipeline is tightly coupled to the data schema and prompt format. When using your own data, please verify the field names, answer format, and generated parquet schema before launching RL training.

## Environment Setup

Latent-GRPO uses a customized SGLang inference engine and a modified verl training engine. We recommend installing both in the same clean conda environment.

### Inference Engine Environment

```bash
conda create -n latent_grpo python=3.11.13 -y
conda activate latent_grpo
pip install pip==25.2
pip install torch==2.6.0 transformers==4.51.1 tensorboard==2.20.0 sgl_kernel==0.1.1 accelerate==1.10.1 torch_memory_saver==0.0.8 uvloop==0.21.0 jsonlines math_verify openai
pip install flash_attn==2.7.3 --no-build-isolation

cd sglang_latent_reasoning_pkg
pip install -e "python[all]"
cd ..
```

> **Note:** `flash_attn` may take a long time to compile. If installation fails due to an undefined-symbol error, reinstall it with `--no-build-isolation` or use a compatible wheel from the official FlashAttention repository.

### Training Engine Environment

After installing the inference engine environment, install the modified verl package:

```bash
cd verl-0.4.x
pip3 install -e .
cd ..
```

## Training

> [!IMPORTANT]
> Do not start Latent-GRPO from a model that has not been initialized with Latent-SFT. Directly applying latent RL to a non-latent-initialized model is unstable and can easily collapse.

For RL training, set `MODEL_PATH` to a latent-reasoning checkpoint, set `DATA_DIR` to the directory containing the parquet data, and choose an output directory.

We release the following useful checkpoints:

- [`DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10`](https://huggingface.co/DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10)
- [`DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10`](https://huggingface.co/DJCheng/Qwen2.5-Math-7B-Latent-GRPO-4k-Top10)

### Low-Difficulty Training

Use `Latent-GRPO-gsm8k-llama3.sh` for the GSM8K-style low-difficulty setting:

```bash
DATA_DIR=./data \
MODEL_PATH=DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10 \
OUTPUT_DIR=./saved/latent-grpo-gsm8k-llama3 \
GPUS=0,1,2,3,4,5,6,7 \
bash Latent-GRPO-gsm8k-llama3.sh
```

The script uses:

```bash
data.train_files=${DATA_DIR}/GSM8k-Aug-oss-dup-all.parquet
data.val_files=${DATA_DIR}/GSM8k-Aug-test.parquet
```

### High-Difficulty Training

Use `Latent-GRPO-math500-qwen.sh` for the high-difficulty math setting:

```bash
DATA_DIR=./data \
MODEL_PATH=DJCheng/Qwen2.5-Math-7B-Latent-GRPO-4k-Top10 \
OUTPUT_DIR=./saved/latent-grpo-math500-qwen \
GPUS=0,1,2,3,4,5,6,7 \
bash Latent-GRPO-math500-qwen.sh
```

The script uses:

```bash
data.train_files=${DATA_DIR}/DAPO-Math-17k-en-train.parquet
data.val_files=${DATA_DIR}/Math-500-test.parquet
```

## Important Parameters

| Parameter | Meaning | Recommendation |
| --- | --- | --- |
| `actor_rollout_ref.actor.freeze_embedding=True` | Freezes token embeddings during RL training. | Keep enabled for stable latent-token behavior. |
| `actor_rollout_ref.rollout.enable_latent=True` | Enables latent reasoning rollout in the customized SGLang engine. | Required for Latent-GRPO. |
| `actor_rollout_ref.rollout.latent_end_token_id` | Token id where the model switches from latent reasoning to explicit reasoning. | Usually the first token id of `</think>`. The provided scripts use `524` for LLaMA and `522` for Qwen; verify this id for your tokenizer. |
| `actor_rollout_ref.rollout.max_topk=10` | Number of top vocabulary tokens used for latent-token superposition during rollout. | Keep consistent with the Latent-SFT initialization, e.g. Top10. |
| `actor_rollout_ref.rollout.gumbel_softmax_temperature=1.0` | Temperature for Gumbel-Softmax latent sampling. | Keep at `1.0` unless running controlled ablations. |
| `actor_rollout_ref.rollout.add_noise_gumbel_softmax=True` | Adds Gumbel noise during rollout. | Recommended for Latent-GRPO training. |
| `actor_rollout_ref.rollout.use_one_sided_gumbel_noise=True` | Uses one-sided Gumbel noise. | Recommended default in the released scripts. |
| `actor_rollout_ref.rollout.noise_scale=1.0` | Controls the strength of Gumbel perturbation. | Main noise-strength knob. |
| `algorithm.exclude_overlong_samples_from_advantage` | Controls whether overlong samples are excluded from advantage computation. | Different tasks use different defaults in the scripts; keeping the script default is usually more stable. |
| `algorithm.filter_groups.enable=True` | Enables dynamic group filtering during sampling. | Used by the released scripts. |
| `algorithm.filter_groups.max_num_gen_batches=50` | Maximum number of generation batches for dynamic sampling. | Controls how aggressively the trainer searches for valid groups. |

## Evaluation

The evaluation scripts are under `eval/` and use the customized SGLang engine.

### Low-Difficulty Evaluation

Use `eval/eval_low_tasks_sglang.py` to evaluate low-difficulty datasets including GSM8K-Aug, GSM8K-Hard, SVAMP, and MultiArith:

```bash
python eval/eval_low_tasks_sglang.py \
  --model_path <path-or-hf-id-of-latent-grpo-checkpoint> \
  --output_path <path-to-save-low-task-results>.json \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --gsm8k_aug_path data/GSM8k-Aug-test.jsonl \
  --gsm8k_hard_path data/GSM8k-Hard-test.jsonl \
  --svamp_path data/Svamp-test.jsonl \
  --multiarith_path data/Multiarith-test.jsonl \
  --max_new_tokens 128 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_topk 10 \
  --num_runs 5
```

The script reports average accuracy and average output length for each dataset.

### High-Difficulty Evaluation

Use `eval/eval_high_tasks_sglang.py` to evaluate high-difficulty datasets including Math-500, AIME-2024, AIME-2025, and GPQA:

```bash
python eval/eval_high_tasks_sglang.py \
  --model_path <path-or-hf-id-of-latent-grpo-checkpoint> \
  --output_path <path-to-save-high-task-results>.json \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --math500_path data/Math-500-test.jsonl \
  --aime24_path data/AIME-2024-test.jsonl \
  --aime25_path data/AIME-2025-test.jsonl \
  --gpqa_path data/GPQA-test.jsonl \
  --max_new_tokens 4096 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_topk 10 \
  --add_noise_gumbel_softmax False \
  --gumbel_softmax_temperature 1.0 \
  --noise_scale 1.0
```

The high-task evaluator reports per-dataset accuracy, average output length, pass@k metrics, and macro-average pass@k. It also writes a lightweight summary JSON next to the main output file.

Important evaluation notes:

- `--max_topk` is equivalent to the training-time `actor_rollout_ref.rollout.max_topk`; keep it consistent with the trained checkpoint.
- `--add_noise_gumbel_softmax` is set to `False` by default for deterministic evaluation unless you intentionally evaluate a stochastic sampling setting.
- `--gumbel_softmax_temperature` and `--noise_scale` only affect decoding when Gumbel noise is enabled.

### Released Checkpoints

We release ready-to-evaluate Latent-GRPO checkpoints on Hugging Face:

- [`DJCheng/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10`](https://huggingface.co/DJCheng/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10)
- [`DJCheng/Qwen2.5-Math-7B-Latent-GRPO-4k-Top10`](https://huggingface.co/DJCheng/Qwen2.5-Math-7B-Latent-GRPO-4k-Top10)

You can evaluate these checkpoints directly with the corresponding low- or high-difficulty evaluation scripts above.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{deng2026latentgrpo,
  title        = {Latent-GRPO: Group Relative Policy Optimization for Latent Reasoning},
  author       = {Deng, Jingcheng and Wei, Zihao and Pang, Liang and Wu, Junhong and Xu, Shicheng and Duan, Zenghao and Shen, Huawei},
  journal      = {arXiv preprint arXiv:2604.27998},
  year         = {2026},
  url          = {https://arxiv.org/abs/2604.27998}
}
```

## Acknowledgements

We thank the [Soft-Thinking](https://github.com/eric-ai-lab/Soft-Thinking) project, which provided the base version for our SGLang framework development. We also thank the SGLang, verl, Hugging Face Transformers, PyTorch, and FlashAttention communities for their open-source infrastructure.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
