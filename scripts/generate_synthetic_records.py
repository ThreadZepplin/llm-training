import argparse
import sys
from pathlib import Path
"""This script generates synthetic operator sheet records based on the print telemetry records."""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_jsonl, write_jsonl
from src.dataset.synthetic import build_synthetic_operator_record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_jsonl(args.input_file)

    synthetic_records = []
    for rec in records:
        if rec.get("record_type") != "print_telemetry":
            continue
        synthetic_records.append(build_synthetic_operator_record(rec))

    write_jsonl(args.output_file, synthetic_records)
    print(f"Wrote {len(synthetic_records)} synthetic operator records to {args.output_file}")


if __name__ == "__main__":
    main()