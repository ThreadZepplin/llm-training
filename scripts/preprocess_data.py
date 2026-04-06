import argparse
import sys
from pathlib import Path
"""This script preprocesses the raw CSV files and converts them into a single JSONL file with normalized records."""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import write_jsonl
from src.preprocessing.classify import iter_real_csv_files, classify_file
from src.preprocessing.print_telemetry import preprocess_print_telemetry
from src.preprocessing.ros_log import preprocess_ros_log
from src.preprocessing.operator_sheet import preprocess_operator_sheet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = []

    for path in iter_real_csv_files(args.input_dir):
        kind = classify_file(path)

        if kind == "ros":
            rec = preprocess_ros_log(path)
        elif kind == "operator":
            rec = preprocess_operator_sheet(path)
        else:
            rec = preprocess_print_telemetry(path)

        records.append(rec)

    write_jsonl(args.output_file, records)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
