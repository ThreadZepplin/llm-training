import argparse
import sys
from pathlib import Path
"""This script takes the train and test examples in the standard example format,
and converts them into the specific input formats we want for fine-tuning and evaluation."""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_jsonl, write_jsonl
from src.dataset.finetune_format import (
    build_train_row,
    build_test_input_row,
    build_test_gold_row,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-test-inputs", type=Path, required=True)
    parser.add_argument("--out-test-gold", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    train_examples = load_jsonl(args.train_input)
    test_examples = load_jsonl(args.test_input)

    finetune_train = [build_train_row(ex) for ex in train_examples]
    finetune_test_inputs = [build_test_input_row(ex) for ex in test_examples]
    finetune_test_gold = [build_test_gold_row(ex) for ex in test_examples]

    write_jsonl(args.out_train, finetune_train)
    write_jsonl(args.out_test_inputs, finetune_test_inputs)
    write_jsonl(args.out_test_gold, finetune_test_gold)

    print(f"Wrote {len(finetune_train)} training rows to {args.out_train}")
    print(f"Wrote {len(finetune_test_inputs)} test input rows to {args.out_test_inputs}")
    print(f"Wrote {len(finetune_test_gold)} test gold rows to {args.out_test_gold}")


if __name__ == "__main__":
    main()
