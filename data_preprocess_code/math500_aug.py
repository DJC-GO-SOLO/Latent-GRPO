import argparse
import json
from pathlib import Path

from datasets import Dataset


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {file_path}") from exc
    return data


def build_record(item, index, data_source, split, instruction):
    problem = item["problem"]
    answer = str(item["answer"])
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": problem},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            "question": problem,
        },
    }


def convert_jsonl_to_parquet(input_path, output_path, data_source, split, instruction):
    data = read_jsonl(input_path)
    processed_data = [
        build_record(item, index, data_source, split, instruction)
        for index, item in enumerate(data)
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(processed_data).to_parquet(str(output_path))
    return output_path, len(processed_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert math JSONL data to VERL parquet format.")
    parser.add_argument("--input_path", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", required=True, help="Path to write the output parquet file.")
    parser.add_argument("--data_source", default="lighteval/MATH", help="Value stored in the data_source field.")
    parser.add_argument("--split", default="train", help="Dataset split name stored in extra_info.")
    parser.add_argument(
        "--instruction",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
        help="System instruction used in each prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, num_examples = convert_jsonl_to_parquet(
        input_path=args.input_path,
        output_path=args.output_path,
        data_source=args.data_source,
        split=args.split,
        instruction=args.instruction,
    )
    print(f"Saved {num_examples} examples to {output_path}")


if __name__ == "__main__":
    main()