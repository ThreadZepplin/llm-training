from src.prompts import build_prompt_messages, build_target_summary
"""This file takes raw normalized records and converts them into the 
standard example format for the whole project. Mostly for bookkeeping/analysis purposes later on."""

def classify_source_kind(record):
    rt = record.get("record_type")

    if rt == "print_telemetry":
        return "real_print_telemetry"
    if rt == "ros_log":
        return "real_ros_log"
    if rt == "operator_sheet":
        return "real_operator_sheet"
    if rt == "synthetic_operator_sheet":
        return "synthetic_operator_sheet"
    return "unknown"


def get_group_id(record):
    if record.get("record_type") == "synthetic_operator_sheet":
        return record.get("derived_from")
    return record.get("source_file")


def make_example(record, idx):
    source_kind = classify_source_kind(record)
    target_summary = build_target_summary(record)

    return {
        "example_id": f"ex_{idx:04d}",
        "group_id": get_group_id(record),
        "source_kind": source_kind,
        "is_synthetic": bool(record.get("synthetic", False)),
        "input_record": record,
        "messages": build_prompt_messages(record),
        "target_summary": target_summary,
    }
