import pandas as pd
from src.io_utils import try_read_csv, clean_columns, to_float
from src.schema import infer_part_name, infer_material, make_operation_id, to_json_safe
"""This file holds the main preprocessing function for the print telemetry logs.
It extracts things like nozzle temperature, bed temperature, chamber temperature, 
screw torque, cycle time, total power used, start and end time
Then it turns that into one normalized record. Pretty sure it has all the existing
columns, but we can add more if we find it's inaccurate."""

def preprocess_print_telemetry(path):
    df, enc = try_read_csv(path)
    df = clean_columns(df)

    part_name = infer_part_name(df, path.name)

    record = {
        "operation_id": make_operation_id(path.name),
        "source_file": path.name,
        "encoding": enc,
        "record_type": "print_telemetry",
        "part_name": part_name,
        "material": infer_material(path.name, part_name),
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "rows_logged": int(len(df)),
        "avg_nozzle_temp_c": None,
        "avg_bed_temp_c": None,
        "avg_chamber_temp_c": None,
        "avg_screw_torque": None,
        "max_screw_torque": None,
        "max_cycle_time": None,
        "total_power_used": None,
        "notes": [],
        "anomalies": [],
    }

    if "DateTime" in df.columns:
        dt = pd.to_datetime(
            df["DateTime"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        if dt.notna().any():
            record["start_time"] = dt.min().isoformat()
            record["end_time"] = dt.max().isoformat()
            record["duration_seconds"] = int((dt.max() - dt.min()).total_seconds())

    if "Nozzle Temperature" in df.columns:
        vals = to_float(df["Nozzle Temperature"])
        record["avg_nozzle_temp_c"] = float(round(vals.mean(), 3)) if vals.notna().any() else None

    if "Bed Temperature" in df.columns:
        vals = to_float(df["Bed Temperature"])
        record["avg_bed_temp_c"] = float(round(vals.mean(), 3)) if vals.notna().any() else None

    if "Chamber Temperature" in df.columns:
        vals = to_float(df["Chamber Temperature"])
        record["avg_chamber_temp_c"] = float(round(vals.mean(), 3)) if vals.notna().any() else None

    if "Screw Torque" in df.columns:
        vals = to_float(df["Screw Torque"])
        if vals.notna().any():
            record["avg_screw_torque"] = float(round(vals.mean(), 3))
            record["max_screw_torque"] = float(round(vals.max(), 3))

    if "Cycle Time" in df.columns:
        vals = to_float(df["Cycle Time"])
        record["max_cycle_time"] = float(round(vals.max(), 3)) if vals.notna().any() else None

    if "Total Power Used During Print" in df.columns:
        vals = to_float(df["Total Power Used During Print"])
        record["total_power_used"] = float(round(vals.max(), 3)) if vals.notna().any() else None

    return to_json_safe(record)
