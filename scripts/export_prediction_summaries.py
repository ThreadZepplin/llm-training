import json
import sys
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_prediction_summaries.py <input_jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    rows = load_jsonl(input_path)
    output_path = input_path.with_suffix(".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            label = row.get("prompt_mode", row.get("source_kind", "output"))
            f.write(f"{row['example_id']} ({label})\n")
            f.write(row["model_output"].strip() + "\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"Wrote summaries to {output_path}")


if __name__ == "__main__":
    main()
