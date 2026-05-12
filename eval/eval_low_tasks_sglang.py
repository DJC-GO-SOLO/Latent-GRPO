import os
import re
import json
import math
import time
import random
import signal
import socket
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer


# =========================
# Answer extraction / scoring helpers
# =========================


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = str(string).replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(",", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left): -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def extract_answer(pred_str, use_last_number=True):
    pred_str = str(pred_str).replace("\u043a\u0438", "")
    pred = ""

    boxed = last_boxed_only_string(pred_str)
    if boxed is not None:
        try:
            pred = remove_boxed(boxed)
        except Exception:
            pred = ""

    if not pred and "####" in pred_str:
        pred = pred_str.split("####")[-1].strip()

    if not pred and use_last_number:
        matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d*\.?\d+", pred_str)
        if matches:
            pred = matches[-1]

    pred = re.sub(r"\n\s*", "", pred).strip()
    pred = pred.strip("$ ")
    if pred.startswith(":"):
        pred = pred[1:]
    if pred.endswith(".") or pred.endswith("/"):
        pred = pred[:-1]
    return pred


def normalize_answer_text(text):
    text = str(text).strip()
    if "boxed" in text or "####" in text:
        text = extract_answer(text, use_last_number=True)
    return strip_string(text)


def _as_float(text):
    text = strip_string(text)
    frac_match = re.fullmatch(r"\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}", text)
    if frac_match:
        denominator = float(frac_match.group(2))
        if denominator == 0:
            return None
        return float(frac_match.group(1)) / denominator
    try:
        return float(text)
    except ValueError:
        return None


def check_is_correct(pred, gt):
    pred_norm = normalize_answer_text(pred)
    gt_norm = normalize_answer_text(gt)
    if pred_norm == gt_norm:
        return True

    pred_value = _as_float(pred_norm)
    gt_value = _as_float(gt_norm)
    if pred_value is None or gt_value is None:
        return False
    return abs(pred_value - gt_value) <= 1e-5


def make_run_seed(base_seed, dataset_index, run_index):
    return int(base_seed) + dataset_index * 100000 + run_index


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line at {file_path}:{line_idx}: {exc}") from exc
    return data


def write_json(path, data):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_last_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Tokenization produced empty ids for marker: {text!r}")
    return ids[0]


def build_prompt_text_qwen(item, tokenizer):
    """Qwen chat template prompt"""
    problem = item.get("problem", None) or item.get("question", "")
    messages = [
        {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n"+problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if prompt.rstrip().endswith("<think>"):
        return prompt
    return prompt + "<think>"


def batched_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def kill_process_tree(pid):
    """Force kill a process and all its children."""
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        )
        for child_pid in result.stdout.strip().split("\n"):
            if child_pid:
                kill_process_tree(int(child_pid))
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, ValueError, OSError):
        pass


def is_port_free(port):
    """Check if a port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def wait_for_ports_free(gpu_ids, timeout=60):
    """Wait for all GPU MASTER_PORTs to be released."""
    ports = [29500 + int(gid) for gid in gpu_ids]
    start = time.time()
    while time.time() - start < timeout:
        all_free = all(is_port_free(p) for p in ports)
        if all_free:
            return True
        time.sleep(2)
    print(f"  Warning: Ports {ports} not fully released after {timeout}s, proceeding anyway...")
    return False

# =========================
# Persistent DP Worker
# =========================
def persistent_inference_worker(gpu_id, work_queue, result_queue, config):
    """
    Persistent worker: load the model once and receive inference jobs through
    work_queue. Queue item: (prompt_chunk, temp_file_path, run_seed), or None
    to shut down.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500 + int(gpu_id))
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    import sglang as sgl
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
        latent_end_token_id = get_last_token_id(tokenizer, "</think>")

        llm = sgl.Engine(
            model_path=config["model_path"],
            trust_remote_code=True,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            tp_size=1,
            enable_latent=True,
            latent_end_token_id=latent_end_token_id,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            mem_fraction_static=0.90,
            sampling_backend="flashinfer",
            max_running_requests=2048,
            log_level="error",
            skip_tokenizer_init=True,
            max_topk=config["max_topk"],
        )

        # Signal that this worker is ready
        result_queue.put(("ready", gpu_id))

        batch_size = config["gen_batch_size"]

        while True:
            item = work_queue.get()
            if item is None:
                break

            prompt_chunk, temp_file_path, run_seed = item
            seed_value = int(run_seed) + int(gpu_id)
            random.seed(seed_value)
            try:
                import torch
                torch.manual_seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)
            except Exception:
                pass
            local_predictions = []

            for batch_prompts in batched_list(prompt_chunk, batch_size):
                outputs = llm.generate(
                    input_ids=batch_prompts,
                    sampling_params={
                        "temperature": config["temperature"],
                        "top_p": config["top_p"],
                        "max_new_tokens": config["max_new_tokens"],
                        "gumbel_softmax_temperature": config["gumbel_softmax_temperature"],
                        "noise_scale": config["noise_scale"],
                        "add_noise_gumbel_softmax": config["add_noise_gumbel_softmax"],
                        "use_one_sided_gumbel_noise": config["use_one_sided_gumbel_noise"],
                    },
                    return_logprob=True,
                )
                batch_output_ids = [o['output_ids'] for o in outputs]
                batch_output_lens = [len(ids) for ids in batch_output_ids]
                decoded_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)
                local_predictions.extend(list(zip(decoded_texts, batch_output_lens)))

            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(local_predictions, f)

            result_queue.put(("done", temp_file_path))

        llm.shutdown()

    except Exception as e:
        print(f"Worker {gpu_id} failed: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(("error", str(e)))


# =========================
# Worker Pool Management
# =========================
def launch_persistent_workers(target_gpu_ids, config):
    """Launch GPU worker processes and return after all workers are ready."""
    work_queues = {gpu_id: mp.Queue() for gpu_id in target_gpu_ids}
    result_queue = mp.Queue()
    processes = []

    for gpu_id in target_gpu_ids:
        p = mp.Process(
            target=persistent_inference_worker,
            args=(gpu_id, work_queues[gpu_id], result_queue, config)
        )
        p.start()
        processes.append((gpu_id, p))

    # Wait for all workers to signal ready
    ready_count = 0
    while ready_count < len(target_gpu_ids):
        msg_type, payload = result_queue.get()
        if msg_type == "ready":
            print(f"  Worker GPU {payload} ready.")
            ready_count += 1
        elif msg_type == "error":
            raise RuntimeError(f"Worker failed during init: {payload}")
        else:
            raise RuntimeError(f"Unexpected worker init message: {msg_type}")

    return processes, work_queues, result_queue


def shutdown_workers(processes, work_queues):
    """Send sentinels to all workers and wait for process shutdown."""
    for gpu_id, _ in processes:
        work_queues[gpu_id].put(None)
    for gpu_id, p in processes:
        p.join(timeout=60)
        if p.is_alive():
            kill_process_tree(p.pid)
            p.join(timeout=5)
        elif p.pid:
            kill_process_tree(p.pid)


# =========================
# One Eval Run (reuses existing workers)
# =========================
def run_one_eval(data, config, target_gpu_ids, work_queues, result_queue,
                 tokenizer, run_tag, run_seed):
    """Run one inference/scoring pass on existing workers."""
    total_len = len(data)

    # Tokenize prompts
    text_prompts = [build_prompt_text_qwen(item, tokenizer) for item in data]
    encodings = tokenizer(
        text_prompts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False
    )
    input_ids_all = encodings["input_ids"]

    # Split across GPUs
    num_gpus = len(target_gpu_ids)
    chunk_size = math.ceil(total_len / num_gpus)
    prompt_chunks = [input_ids_all[i: i + chunk_size] for i in range(0, total_len, chunk_size)]

    # Dispatch work to persistent workers
    active_gpu_ids = []
    temp_dir = config["output_dir"]
    for i, gpu_id in enumerate(target_gpu_ids):
        if i >= len(prompt_chunks):
            break
        temp_file = os.path.join(temp_dir, f"temp_preds_gpu_{gpu_id}_{run_tag}.json")
        work_queues[gpu_id].put((prompt_chunks[i], temp_file, run_seed))
        active_gpu_ids.append(gpu_id)

    # Collect results
    temp_files = []
    for _ in range(len(active_gpu_ids)):
        msg_type, payload = result_queue.get()
        if msg_type == "done":
            temp_files.append(payload)
        elif msg_type == "error":
            raise RuntimeError(f"Worker error during inference: {payload}")
        else:
            raise RuntimeError(f"Unexpected worker message during inference: {msg_type}")

    # Merge (sort by GPU id to preserve order)
    temp_files.sort(key=lambda x: int(x.split("_gpu_")[1].split("_")[0]))

    predictions = []
    output_lengths = []
    for fpath in temp_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            chunk_preds = json.load(f)
            for text, length in chunk_preds:
                predictions.append(text)
                output_lengths.append(length)
        os.remove(fpath)

    if len(predictions) != total_len:
        raise ValueError(f"Predictions({len(predictions)}) != Data({total_len})")

    # Score
    def score_chunk_fn(indices_range):
        start, end = indices_range
        chunk_scores = []
        for i in range(start, end):
            if i >= len(predictions):
                break
            pred = predictions[i]
            ans = data[i].get("answer", data[i].get("solution", ""))
            try:
                pred_answer = extract_answer(pred)
                score = 1.0 if check_is_correct(pred_answer, ans) else 0.0
            except Exception as e:
                score = 0.0
                if i < 3:
                    print(f"\n  [DEBUG i={i}] Exception: {e}")
                    print(f"    gold_ans: {repr(ans)}")
            chunk_scores.append(score)
        return chunk_scores


    score_batch_size = 10000
    chunk_ranges = [
        (i, min(i + score_batch_size, len(predictions)))
        for i in range(0, len(predictions), score_batch_size)
    ]
    scores = []
    with ThreadPoolExecutor(max_workers=config["num_score_workers"]) as ex:
        for chunk_res in ex.map(score_chunk_fn, chunk_ranges):
            scores.extend(chunk_res)

    accuracy = sum(scores) / len(scores) if scores else 0.0
    max_new_tokens = config["max_new_tokens"]
    non_truncated_lengths = [l for l in output_lengths if l < max_new_tokens]
    avg_len = sum(non_truncated_lengths) / len(non_truncated_lengths) if non_truncated_lengths else 0.0
    truncated_count = len(output_lengths) - len(non_truncated_lengths)
    if truncated_count > 0:
        print(f"  [avg_len] Excluded {truncated_count}/{len(output_lengths)} truncated samples (len=={max_new_tokens})")
    return accuracy, len(scores), avg_len, predictions, scores, output_lengths



# =========================
# Main
# =========================
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Evaluate low tasks with SGLang")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model path (local or remote)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path for aggregated result JSON. Default: {model_path}/low_tasks_eval_result_{sampling_tag}.json")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated physical GPU IDs to use")
    parser.add_argument("--gsm8k_aug_path", type=str, default="GSM8k-Aug-test.jsonl")
    parser.add_argument("--gsm8k_hard_path", type=str, default="GSM8k-Hard-test.jsonl")
    parser.add_argument("--svamp_path", type=str, default="Svamp-test.jsonl")
    parser.add_argument("--multiarith_path", type=str, default="Multiarith-test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_topk", type=int, default=10)
    parser.add_argument("--gen_batch_size", type=int, default=1024)
    parser.add_argument("--num_score_workers", type=int, default=32)
    parser.add_argument("--gumbel_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--add_noise_gumbel_softmax", default=False)
    parser.add_argument(
        "--use_one_sided_gumbel_noise",
        action="store_true",
        default=False,
        help="Enable one-sided Gumbel noise sampling.",
    )
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of evaluation runs per dataset")
    parser.add_argument("--base_seed", type=int, default=12345)
    args = parser.parse_args()

    target_gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not target_gpu_ids:
        raise ValueError("gpu_ids must not be empty.")

    sampling_tag = (
        f"gumbel{args.gumbel_softmax_temperature}"
        f"_noise{args.noise_scale}"
        f"_addnoise{args.add_noise_gumbel_softmax}"
        f"_onesided{args.use_one_sided_gumbel_noise}"
    )

    output_path = args.output_path or os.path.join(args.model_path, "low_tasks_eval_result.json")
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else args.model_path

    config = {
        "model_path": args.model_path,
        "output_dir": os.path.join(output_dir, f"_temp_eval_{sampling_tag}"),
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "noise_scale": args.noise_scale,
        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
        "use_one_sided_gumbel_noise": args.use_one_sided_gumbel_noise,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_topk": args.max_topk,
        "gen_batch_size": args.gen_batch_size,
        "num_score_workers": args.num_score_workers,
    }

    datasets_config = [
        {"dataset_name": "gsm8k_aug", "data_path": args.gsm8k_aug_path},
        {"dataset_name": "gsm8k_hard", "data_path": args.gsm8k_hard_path},
        {"dataset_name": "svamp", "data_path": args.svamp_path},
        {"dataset_name": "multiarith", "data_path": args.multiarith_path},
    ]

    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"GPUs:  {target_gpu_ids}")
    print(f"Output: {output_path}")
    print(f"Runs per dataset: {args.num_runs}")
    print(f"{'='*60}\n")

    # Load tokenizer once (shared across all datasets and runs)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Launch GPU workers once (model loaded once per GPU)
    print("Launching persistent workers (loading model)...")
    processes, work_queues, result_queue = launch_persistent_workers(target_gpu_ids, config)
    print(f"All {len(target_gpu_ids)} workers ready.\n")

    all_results = {}

    try:
        for dataset_index, dataset_cfg in enumerate(datasets_config):
            dataset_name = dataset_cfg["dataset_name"]
            data_path = dataset_cfg["data_path"]
            data = read_jsonl(data_path)
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} ({len(data)} samples)")
            print(f"{'='*60}")

            run_accuracies = []
            run_avg_lens = []
            per_run_details = []

            for run_idx in range(args.num_runs):
                print(f"\n  --- Run {run_idx + 1}/{args.num_runs} ---")
                run_index = run_idx + 1
                run_tag = f"{dataset_name}_run{run_index}"
                run_seed = make_run_seed(args.base_seed, dataset_index, run_index)
                # Each run gets its own temp dir to avoid file collisions
                config["output_dir"] = os.path.join(
                    output_dir, f"_temp_eval_{sampling_tag}", dataset_name, f"run{run_index}"
                )
                accuracy, total, avg_len, predictions, scores, output_lengths = run_one_eval(
                    data, config, target_gpu_ids, work_queues, result_queue,
                    tokenizer, run_tag, run_seed
                )
                run_accuracies.append(accuracy)
                run_avg_lens.append(avg_len)
                per_run_details.append({
                    "run": run_index,
                    "seed": run_seed,
                    "accuracy": round(accuracy, 6),
                    "correct": int(round(accuracy * total)),
                    "total_samples": total,
                    "avg_output_length": round(avg_len, 1),
                })
                print(f"  Run {run_idx + 1}: acc={accuracy:.4f} | avg_len={avg_len:.1f}")

            avg_accuracy = sum(run_accuracies) / len(run_accuracies)
            avg_len_mean = sum(run_avg_lens) / len(run_avg_lens)

            all_results[dataset_name] = {
                "dataset_name": dataset_name,
                "data_path": data_path,
                "avg_accuracy": round(avg_accuracy, 6),
                "avg_output_length": round(avg_len_mean, 1),
                "num_runs": args.num_runs,
                "per_run": per_run_details,
            }

            print(f"\n  [{dataset_name}] avg_acc={avg_accuracy:.4f} | avg_len={avg_len_mean:.1f}")

    finally:
        print("\nShutting down workers...")
        shutdown_workers(processes, work_queues)

    # Save aggregated results

    final_result = {
        "model_path": args.model_path,
        "datasets_config": datasets_config,
        "datasets": all_results,
        # generation params
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_topk": args.max_topk,
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "noise_scale": args.noise_scale,
        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
        "use_one_sided_gumbel_noise": args.use_one_sided_gumbel_noise,
        "gen_batch_size": args.gen_batch_size,
        "num_runs": args.num_runs,
        "base_seed": args.base_seed,
    }

    write_json(output_path, final_result)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Dataset':<25} {'Avg Acc':>10} {'Avg Len':>10}")
    print(f"{'-'*45}")
    for ds_name, res in all_results.items():
        print(f"{ds_name:<25} {res['avg_accuracy']:>10.4f} {res['avg_output_length']:>10.1f}")
    print(f"{'='*60}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
