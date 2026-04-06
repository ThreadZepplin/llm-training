from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np
"""This file holds common helpers for inferring material and part names from filenames,
as well as some common formatting functions for making things more human readable."""

def infer_material(filename: str, part_name: str | None = None):
    text = f"{filename} {part_name or ''}".lower()
    if "polypro" in text:
        return "PolyPro"
    if "hdpe" in text:
        return "HDPE"
    return None


def infer_part_name(df, filename: str):
    candidates = []
    for col in df.columns:
        c = str(col).strip()
        low = c.lower()
        if any(x in low for x in ["hex", "rear", "front", "polypro", "brim"]):
            candidates.append(c)

    if candidates:
        return max(candidates, key=len)

    stem = Path(filename).stem
    stem = re.sub(r"[-_]\d{8,}$", "", stem)
    return stem


def make_operation_id(name: str):
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def format_num(value, decimals=3):
    if value is None:
        return None
    return f"{float(value):.{decimals}f}"


def format_human_datetime(dt_str):
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
    except ValueError:
        return dt_str

    time_part = dt.strftime("%I:%M %p").lstrip("0")
    date_part = dt.strftime("%B %d, %Y").replace(" 0", " ")
    return f"{time_part} on {date_part}"


def format_human_date(dt_str):
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
    except ValueError:
        return dt_str

    return dt.strftime("%B %d, %Y").replace(" 0", " ")
