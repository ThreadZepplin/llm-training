import argparse
import sys
from pathlib import Path
"""This script takes the preprocessed real records and the generated synthetic records, 
builds the standard example format for all of them, and then splits them into train and test sets based"""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_jsonl, write_jsonl, write_json
from src.dataset.build_examples import make_example
from src.dataset.splits import choose_manual_test_groups, split_by_group, build_split_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-file", type=Path, required=True)
    parser.add_argument("--synthetic-file", type=Path, required=True)
    parser.add_argument("--out-all", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-test", type=Path, required=True)
    parser.add_argument("--out-meta", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    real_records = load_jsonl(args.real_file)
    synthetic_records = load_jsonl(args.synthetic_file)

    all_records = real_records + synthetic_records

    examples = []
    for idx, record in enumerate(all_records, start=1):
        examples.append(make_example(record, idx))

    held_out_group_ids = choose_manual_test_groups(examples)
    train, test = split_by_group(examples, held_out_group_ids)
    split_meta = build_split_metadata(train, test, held_out_group_ids)

    write_jsonl(args.out_all, examples)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    write_json(args.out_meta, split_meta)

    print(f"Wrote {len(examples)} total examples to {args.out_all}")
    print(f"Wrote {len(train)} train examples to {args.out_train}")
    print(f"Wrote {len(test)} test examples to {args.out_test}")
    print(f"Wrote split metadata to {args.out_meta}")
    print("Held-out groups:")
    for gid in held_out_group_ids:
        print(f"  - {gid}")


if __name__ == "__main__":
    main()
    