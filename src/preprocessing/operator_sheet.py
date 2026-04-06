import pandas as pd
import re
from src.io_utils import try_read_csv
from src.schema import make_operation_id, to_json_safe
"""This file holds the main preprocessing function for the operator sheets.
These are the sheets that the operators fill out by hand. They have things like
notes about print quality, anomalies that happened during the print, and sometimes
the start time of the print. There are only a few of these, but they can be useful 
for adding variety to the training, I think"""

def preprocess_operator_sheet(path):
    df, enc = try_read_csv(path, header=None)

    record = {
        "operation_id": make_operation_id(path.name),
        "source_file": path.name,
        "encoding": enc,
        "record_type": "operator_sheet",
        "part_name": None,
        "material": None,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "rows_logged": int(len(df)),
        "notes": [],
        "anomalies": [],
    }

    all_text = []
    for val in df.fillna("").values.flatten():
        text = str(val).strip()
        if text:
            all_text.append(text)

    for text in all_text:
        low = text.lower()

        if "hdpe" in low and record["material"] is None:
            record["material"] = "HDPE"

        if "print started" in low:
            record["notes"].append(text)

            m = re.search(r'(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}\s*[ap]m)', text, re.IGNORECASE)
            if m and record["start_time"] is None:
                dt_text = f"{m.group(1)} {m.group(2)}"
                parsed = pd.to_datetime(dt_text, errors="coerce")
                if pd.notna(parsed):
                    record["start_time"] = parsed.isoformat()

        if "cancel" in low or "lost extrusion load" in low or "lost exteusion load" in low:
            record["anomalies"].append(text)

        if "hexagon" in low and record["part_name"] is None:
            record["part_name"] = text

    return to_json_safe(record)
