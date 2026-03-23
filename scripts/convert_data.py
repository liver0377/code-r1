"""
数据转换脚本: python-codes-25k.jsonl -> python-codes-5k.parquet
将原始数据转换为verl训练所需的格式
"""

import json
import pandas as pd
import os


def convert_jsonl_to_parquet(
    input_path: str, output_path: str, num_samples: int = 5000, use_instruction_only: bool = True
):
    """
    转换JSONL数据为Parquet格式

    Args:
        input_path: 输入JSONL文件路径
        output_path: 输出Parquet文件路径
        num_samples: 提取的样本数量
        use_instruction_only: True则只用instruction作为prompt, False则合并instruction+input
    """
    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            item = json.loads(line.strip())

            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            if use_instruction_only:
                prompt = instruction
            else:
                prompt = f"{instruction}\n{input_text}" if input_text else instruction

            record = {
                "prompt": prompt,
                "data_source": "python_codes",
                "extra_info": {"user_message": instruction, "ground_truth": output_text},
            }

            data.append(record)

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_parquet(output_path, index=False)

    print(f"转换完成: {len(data)} 条数据")
    print(f"输出文件: {output_path}")
    print(f"示例数据:")
    print(f"  prompt: {data[0]['prompt'][:100]}...")
    print(f"  data_source: {data[0]['data_source']}")
    print(f"  extra_info keys: {list(data[0]['extra_info'].keys())}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="转换JSONL数据为Parquet格式")
    parser.add_argument("--input", type=str, default="dataset/python-codes-25k.jsonl", help="输入JSONL文件路径")
    parser.add_argument("--output", type=str, default="dataset/python-codes-5k.parquet", help="输出Parquet文件路径")
    parser.add_argument("--num_samples", type=int, default=5000, help="提取的样本数量")
    parser.add_argument("--use_instruction_only", action="store_true", help="只使用instruction作为prompt")

    args = parser.parse_args()

    convert_jsonl_to_parquet(
        input_path=args.input,
        output_path=args.output,
        num_samples=args.num_samples,
        use_instruction_only=args.use_instruction_only,
    )
