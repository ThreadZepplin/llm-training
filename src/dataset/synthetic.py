import random
from datetime import datetime
"""This file builds synthetic operator sheet records based on the print telemetry and ROS log records.
It just adds more operator-like examples to the training data, since we only have a few real operator sheets,
and not a lot of records overall."""
random.seed(42)


def format_date(dt_str):
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    return f"{dt.month}/{dt.day}/{dt.year}"


def format_time(dt_str):
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    hour = dt.hour % 12
    if hour == 0:
        hour = 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.minute:02d}:{dt.second:02d} {ampm}"


def infer_operator_status(record):
    duration = record.get("duration_seconds")
    rows = record.get("rows_logged", 0)

    if duration is None:
        return "unknown"
    if duration < 300 or rows < 100:
        return "early_stop"
    if duration >= 3000:
        return "completed"
    return "normal"


def build_note_and_anomaly(record):
    status = infer_operator_status(record)

    normal_notes = [
        "Print started normally; brim layers applied as planned",
        "Operator observed normal run conditions",
        "Material loaded and print started normally",
        "Routine startup completed; print proceeded as expected",
        "Operator confirmed normal startup conditions",
    ]

    completed_notes = [
        "Print started normally; brim layers applied as planned",
        "Run completed without operator-reported issues",
        "Operator observed stable print conditions during run",
        "Print completed full run with no manual intervention noted",
        "Startup completed and print continued under normal conditions",
    ]

    anomaly_notes = [
        "Print started, 2 layers of brim",
        "Operator started print and monitored early layers",
        "Print startup completed; operator observed issue shortly after start",
        "Print began normally, but the run was halted early for inspection",
    ]

    anomaly_events = [
        "Run stopped early and was paused for inspection",
        "Operator halted print after early abnormal behavior was observed",
        "Print did not continue through full run; operator intervened early",
        "Run ended early during initial monitoring",
    ]

    if status == "early_stop":
        return status, [random.choice(anomaly_notes)], [random.choice(anomaly_events)]
    if status == "completed":
        return status, [random.choice(completed_notes)], []
    if status == "normal":
        return status, [random.choice(normal_notes)], []
    return status, ["Operator note unavailable"], []


def build_synthetic_operator_record(record):
    synthesis_rule, notes, anomalies = build_note_and_anomaly(record)

    synthetic = {
        "operation_id": f"{record['operation_id']}_synthetic_operator",
        "source_file": None,
        "record_type": "synthetic_operator_sheet",
        "synthetic": True,
        "derived_from": record["source_file"],
        "synthesis_rule": synthesis_rule,
        "part_name": record.get("part_name"),
        "material": record.get("material"),
        "print_date": format_date(record.get("start_time")),
        "print_start_time_display": format_time(record.get("start_time")),
        "print_end_time_display": format_time(record.get("end_time")),
        "start_time": record.get("start_time"),
        "end_time": record.get("end_time"),
        "duration_seconds": record.get("duration_seconds"),
        "rows_logged": record.get("rows_logged"),
        "avg_nozzle_temp_c": record.get("avg_nozzle_temp_c"),
        "avg_bed_temp_c": record.get("avg_bed_temp_c"),
        "avg_chamber_temp_c": record.get("avg_chamber_temp_c"),
        "avg_screw_torque": record.get("avg_screw_torque"),
        "max_screw_torque": record.get("max_screw_torque"),
        "max_cycle_time": record.get("max_cycle_time"),
        "total_power_used": record.get("total_power_used"),
        "notes": notes,
        "anomalies": anomalies,
    }

    return synthetic
