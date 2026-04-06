import pandas as pd
from src.io_utils import try_read_csv, clean_columns, to_float
from src.schema import infer_material, make_operation_id, to_json_safe
"""This file holds the main preprocessing function for the ROS logs.
It extracts things like tool position, extruder acceleration, jerk, and max speed."""

def preprocess_ros_log(path):
    df, enc = try_read_csv(path)
    df = clean_columns(df)

    record = {
        "operation_id": make_operation_id(path.name),
        "source_file": path.name,
        "encoding": enc,
        "record_type": "ros_log",
        "part_name": None,
        "material": infer_material(path.name),
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "rows_logged": int(len(df)),
        "tool_x_mean": None,
        "tool_y_mean": None,
        "tool_z_mean": None,
        "extruder_acceleration_mean": None,
        "extruder_jerk_mean": None,
        "extruder_max_speed_mean": None,
        "notes": [],
        "anomalies": [],
    }

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            record["start_time"] = ts.min().isoformat()
            record["end_time"] = ts.max().isoformat()
            record["duration_seconds"] = int((ts.max() - ts.min()).total_seconds())

    for src_col, dst_col in [
        ("data_tool_x", "tool_x_mean"),
        ("data_tool_y", "tool_y_mean"),
        ("data_tool_z", "tool_z_mean"),
        ("data_extruder_acceleration", "extruder_acceleration_mean"),
        ("data_extruder_jerk", "extruder_jerk_mean"),
        ("data_extruder_max_speed", "extruder_max_speed_mean"),
    ]:
        if src_col in df.columns:
            vals = to_float(df[src_col])
            record[dst_col] = float(round(vals.mean(), 3)) if vals.notna().any() else None

    return to_json_safe(record)
