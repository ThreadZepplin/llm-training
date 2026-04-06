from pathlib import Path
import json
import pandas as pd
"""This holds common file helpers for reading/writing JSONL, JSON, and CSV files, 
as well as some common cleaning steps for CSVs."""

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict | list) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def try_read_csv(path: Path, header="infer"):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, header=header)
            return df, enc
        except Exception:
            pass
    raise ValueError(f"Could not read {path.name}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_float(series):
    return pd.to_numeric(series, errors="coerce")
