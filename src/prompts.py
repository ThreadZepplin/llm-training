import json
from src.schema import format_num, format_human_datetime
"""This file builds the cleaned model input record, stores prompt templates like 
raw_basic, structured, oneshot. and it builds target summaries for training data."""

VALID_PROMPT_MODES = {"raw_basic", "basic", "structured", "oneshot"}


def make_model_input_record(record):
    keep_keys = [
        "record_type",
        "part_name",
        "material",
        "start_time",
        "end_time",
        "duration_seconds",
        "rows_logged",
        "avg_nozzle_temp_c",
        "avg_bed_temp_c",
        "avg_chamber_temp_c",
        "avg_screw_torque",
        "max_screw_torque",
        "max_cycle_time",
        "total_power_used",
        "tool_x_mean",
        "tool_y_mean",
        "tool_z_mean",
        "extruder_acceleration_mean",
        "extruder_jerk_mean",
        "extruder_max_speed_mean",
        "notes",
        "anomalies",
    ]

    cleaned = {}
    for key in keep_keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        cleaned[key] = value

    if cleaned.get("record_type") == "synthetic_operator_sheet":
        cleaned["record_type"] = "operator_sheet"

    return cleaned


def get_system_prompt(mode: str):
    if mode == "raw_basic":
        return None

    if mode == "basic":
        return (
            "You are a manufacturing record summarizer. "
            "Write a short factual summary of the input record. "
            "Only state facts present in the input."
        )

    if mode == "structured":
        return (
            "You are a manufacturing record summarizer.\n\n"
            "# Task\n"
            "Write a short factual summary of the input record.\n\n"
            "# Output Requirements\n"
            "- Write 3 to 5 complete sentences.\n"
            "- Use natural prose, not a field-by-field list.\n"
            "- Sentence 1 should identify the record type and include part name and material if present.\n"
            "- Sentence 2 should describe timing and duration if present.\n"
            "- Sentence 3 should summarize the most important numeric values.\n"
            "- Add one final sentence about notes or anomalies only if present.\n\n"
            "# Rules\n"
            "- Only state facts present in the input.\n"
            "- Do not mention JSON keys, IDs, file names, provenance, or internal metadata.\n"
            "- Do not mention whether the record is synthetic or derived from another record.\n"
            "- Do not mention missing or null values.\n"
            "- Do not repeat the input verbatim."
        )

    if mode == "oneshot":
        return (
            "You are a manufacturing record summarizer.\n\n"
            "# Task\n"
            "Write a short factual summary of the input record.\n\n"
            "# Output Requirements\n"
            "- Write 3 to 5 complete sentences.\n"
            "- Use natural prose, not a field-by-field list.\n"
            "- Sentence 1 should identify the record type and include part name and material if present.\n"
            "- Sentence 2 should describe timing and duration if present.\n"
            "- Sentence 3 should summarize the most important numeric values.\n"
            "- Add one final sentence about notes or anomalies only if present.\n\n"
            "# Rules\n"
            "- Only state facts present in the input.\n"
            "- Do not mention JSON keys, IDs, file names, provenance, or internal metadata.\n"
            "- Do not mention whether the record is synthetic or derived from another record.\n"
            "- Do not mention missing or null values.\n"
            "- Do not repeat the input verbatim.\n\n"
            "# Example\n"
            "Input:\n"
            "{\"record_type\":\"print_telemetry\",\"part_name\":\"Rear Right PolyPro\","
            "\"material\":\"PolyPro\",\"start_time\":\"2025-10-01T08:30:00\","
            "\"end_time\":\"2025-10-01T08:50:00\",\"duration_seconds\":1200,"
            "\"rows_logged\":908,\"avg_nozzle_temp_c\":222.2,\"avg_bed_temp_c\":100.0,"
            "\"avg_chamber_temp_c\":74.1,\"avg_screw_torque\":17.88,"
            "\"max_screw_torque\":35.2,\"max_cycle_time\":2497.0,"
            "\"total_power_used\":3283.0}\n\n"
            "Output:\n"
            "This record contains print telemetry data for the part \"Rear Right PolyPro\" using material PolyPro. "
            "The run started at 8:30 AM on October 1, 2025 and ended at 8:50 AM on October 1, 2025, lasting 1200 seconds. "
            "The record contains 908 logged rows. Key recorded values include an average nozzle temperature of 222.2, "
            "an average bed temperature of 100.0, an average chamber temperature of 74.1, an average screw torque of 17.88, "
            "a maximum screw torque of 35.2, a maximum cycle time of 2497.0, and total power usage of 3283.0."
        )

    raise ValueError(f"Unknown prompt mode: {mode}")


def build_prompt_messages(record):
    model_input = make_model_input_record(record)

    return [
        {
            "role": "system",
            "content": (
                "You are a manufacturing record summarizer.\n\n"
                "# Task\n"
                "Write a short factual summary of the input record.\n\n"
                "# Output Requirements\n"
                "- Write 3 to 6 complete sentences.\n"
                "- Use natural prose, not a field-by-field list.\n"
                "- Sentence 1: identify the record type and include part name and material if present.\n"
                "- Sentence 2: describe timing and duration if present.\n"
                "- Sentence 3: summarize the most important numeric values.\n"
                "- Add one final sentence about notes or anomalies only if present.\n\n"
                "# Rules\n"
                "- Only state facts present in the input.\n"
                "- Do not mention JSON keys, IDs, file names, provenance, or internal metadata.\n"
                "- Do not mention whether the record is synthetic or derived from another record.\n"
                "- Do not mention missing or null values.\n"
                "- Do not repeat the input verbatim.\n\n"
                "# Style\n"
                "- Be concise, factual, and grammatically natural.\n"
                "- Prefer phrases like 'The run started at ... and ended at ...' and "
                "'Key recorded values include ...'.\n\n"
                "# Example\n"
                "Input:\n"
                "{\"record_type\":\"print_telemetry\",\"part_name\":\"Rear Right PolyPro\","
                "\"material\":\"PolyPro\",\"start_time\":\"2025-10-01T08:30:00\","
                "\"end_time\":\"2025-10-01T08:50:00\",\"duration_seconds\":1200,"
                "\"rows_logged\":908,\"avg_nozzle_temp_c\":222.2,\"avg_bed_temp_c\":100.0,"
                "\"avg_chamber_temp_c\":74.1,\"avg_screw_torque\":17.88,"
                "\"max_screw_torque\":35.2,\"max_cycle_time\":2497.0,"
                "\"total_power_used\":3283.0}\n\n"
                "Output:\n"
                "This record contains print telemetry data for the part \"Rear Right PolyPro\" "
                "using material PolyPro. The run started at 8:30 AM on October 1, 2025 and ended at "
                "8:50 AM on October 1, 2025, lasting 1200 seconds. The record contains 908 logged rows. "
                "Key recorded values include an average nozzle temperature of 222.2, an average "
                "bed temperature of 100.0, an average chamber temperature of 74.1, an average "
                "screw torque of 17.88, a maximum screw torque of 35.2, a maximum cycle time "
                "of 2497.0, and total power usage of 3283.0."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(model_input, ensure_ascii=False),
        },
    ]


def build_target_summary(record):
    record_type = record.get("record_type")
    part_name = record.get("part_name")
    material = record.get("material")
    start_time = record.get("start_time")
    end_time = record.get("end_time")
    duration_seconds = record.get("duration_seconds")
    rows_logged = record.get("rows_logged")
    notes = record.get("notes", [])
    anomalies = record.get("anomalies", [])

    def join_details():
        parts = []
        if part_name:
            parts.append(f'for the part "{part_name}"')
        if material:
            parts.append(f"using material {material}")
        return " ".join(parts)

    sentences = []

    if record_type == "print_telemetry":
        intro = "This record contains print telemetry data"
    elif record_type == "ros_log":
        intro = "This record contains ROS machine log data"
    elif record_type in {"operator_sheet", "synthetic_operator_sheet"}:
        intro = "This record contains an operator check sheet"
    else:
        intro = "This record contains manufacturing data"

    details = join_details()
    if details:
        intro += " " + details
    sentences.append(intro + ".")

    human_start = format_human_datetime(start_time)
    human_end = format_human_datetime(end_time)

    if start_time and end_time and duration_seconds is not None:
        sentences.append(
            f"The run started at {human_start} and ended at {human_end}, lasting {duration_seconds} seconds."
        )
    elif start_time and end_time:
        sentences.append(f"The run started at {human_start} and ended at {human_end}.")
    elif start_time:
        sentences.append(f"The run started at {human_start}.")
    elif end_time:
        sentences.append(f"The run ended at {human_end}.")

    metric_parts = []

    if record_type == "ros_log":
        if rows_logged is not None:
            sentences.append(f"The log contains {rows_logged} recorded rows.")
        if record.get("tool_x_mean") is not None:
            metric_parts.append(f"a mean tool x value of {format_num(record['tool_x_mean'], 2)}")
        if record.get("tool_y_mean") is not None:
            metric_parts.append(f"a mean tool y value of {format_num(record['tool_y_mean'], 2)}")
        if record.get("tool_z_mean") is not None:
            metric_parts.append(f"a mean tool z value of {format_num(record['tool_z_mean'], 2)}")
        if record.get("extruder_acceleration_mean") is not None:
            metric_parts.append(f"a mean extruder acceleration of {format_num(record['extruder_acceleration_mean'], 1)}")
        if record.get("extruder_jerk_mean") is not None:
            metric_parts.append(f"a mean extruder jerk of {format_num(record['extruder_jerk_mean'], 1)}")
        if record.get("extruder_max_speed_mean") is not None:
            metric_parts.append(f"a mean extruder max speed of {format_num(record['extruder_max_speed_mean'], 1)}")

    elif record_type == "print_telemetry":
        if rows_logged is not None:
            sentences.append(f"The record contains {rows_logged} logged rows.")
        if record.get("avg_nozzle_temp_c") is not None:
            metric_parts.append(f"an average nozzle temperature of {format_num(record['avg_nozzle_temp_c'], 1)}")
        if record.get("avg_bed_temp_c") is not None:
            metric_parts.append(f"an average bed temperature of {format_num(record['avg_bed_temp_c'], 1)}")
        if record.get("avg_chamber_temp_c") is not None:
            metric_parts.append(f"an average chamber temperature of {format_num(record['avg_chamber_temp_c'], 1)}")
        if record.get("avg_screw_torque") is not None:
            metric_parts.append(f"an average screw torque of {format_num(record['avg_screw_torque'], 2)}")
        if record.get("max_screw_torque") is not None:
            metric_parts.append(f"a maximum screw torque of {format_num(record['max_screw_torque'], 1)}")
        if record.get("max_cycle_time") is not None:
            metric_parts.append(f"a maximum cycle time of {format_num(record['max_cycle_time'], 1)}")
        if record.get("total_power_used") is not None:
            metric_parts.append(f"total power usage of {format_num(record['total_power_used'], 1)}")

    else:
        if rows_logged is not None:
            sentences.append(f"The record contains {rows_logged} logged rows.")
        if record.get("avg_nozzle_temp_c") is not None:
            metric_parts.append(f"an average nozzle temperature of {format_num(record['avg_nozzle_temp_c'], 1)}")
        if record.get("avg_bed_temp_c") is not None:
            metric_parts.append(f"an average bed temperature of {format_num(record['avg_bed_temp_c'], 1)}")
        if record.get("avg_chamber_temp_c") is not None:
            metric_parts.append(f"an average chamber temperature of {format_num(record['avg_chamber_temp_c'], 1)}")
        if record.get("avg_screw_torque") is not None:
            metric_parts.append(f"an average screw torque of {format_num(record['avg_screw_torque'], 2)}")
        if record.get("max_screw_torque") is not None:
            metric_parts.append(f"a maximum screw torque of {format_num(record['max_screw_torque'], 1)}")
        if record.get("max_cycle_time") is not None:
            metric_parts.append(f"a maximum cycle time of {format_num(record['max_cycle_time'], 1)}")
        if record.get("total_power_used") is not None:
            metric_parts.append(f"total power usage of {format_num(record['total_power_used'], 1)}")

    if metric_parts:
        sentences.append("Key recorded values include " + "; ".join(metric_parts) + ".")

    if notes:
        if len(notes) == 1:
            sentences.append(f'Operator note: "{notes[0]}".')
        else:
            sentences.append('Operator notes: "' + '" ; "'.join(notes) + '".')

    if anomalies:
        if len(anomalies) == 1:
            sentences.append(f'An anomaly was recorded: "{anomalies[0]}".')
        else:
            sentences.append('Anomalies were recorded: "' + '" ; "'.join(anomalies) + '".')

    return " ".join(sentences)
