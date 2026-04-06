"""This file holds the logic for how we split the dataset into train and test. 
We want to make sure that all records from the same print are in the same split, 
so we split by group_id. We also want to make sure we have a good variety of records 
in the test set, so we manually choose some groups to hold out based on their source_kind and other metadata. 
Finally, we build some metadata about the splits for bookkeeping purposes."""
def choose_manual_test_groups(examples):
    by_group = {}
    for ex in examples:
        by_group.setdefault(ex["group_id"], []).append(ex)

    real_ros_groups = []
    real_operator_groups = []
    long_print_groups = []
    short_print_groups = []

    for group_id, group_examples in by_group.items():
        real_records = [e for e in group_examples if not e["is_synthetic"]]
        if not real_records:
            continue

        base = real_records[0]
        record = base["input_record"]
        source_kind = base["source_kind"]

        if source_kind == "real_ros_log":
            real_ros_groups.append(group_id)

        elif source_kind == "real_operator_sheet":
            real_operator_groups.append(group_id)

        elif source_kind == "real_print_telemetry":
            duration = record.get("duration_seconds")
            rows = record.get("rows_logged", 0)

            if duration is not None and (duration < 300 or rows < 100):
                short_print_groups.append(group_id)
            elif duration is not None and duration >= 3000:
                long_print_groups.append(group_id)

    real_ros_groups = sorted(real_ros_groups)
    real_operator_groups = sorted(real_operator_groups)
    long_print_groups = sorted(long_print_groups)
    short_print_groups = sorted(short_print_groups)

    selected = []
    selected.extend(real_ros_groups[:2])
    selected.extend(long_print_groups[:2])
    selected.extend(short_print_groups[:2])

    if real_operator_groups:
        selected.append(real_operator_groups[0])

    return list(dict.fromkeys(selected))


def split_by_group(examples, held_out_group_ids):
    train = []
    test = []

    for ex in examples:
        if ex["group_id"] in held_out_group_ids:
            test.append(ex)
        else:
            train.append(ex)

    return train, test


def build_split_metadata(train, test, held_out_group_ids):
    def summarize_bucket(rows):
        out = {
            "count": len(rows),
            "by_source_kind": {},
            "groups": sorted({r["group_id"] for r in rows}),
        }
        for r in rows:
            key = r["source_kind"]
            out["by_source_kind"][key] = out["by_source_kind"].get(key, 0) + 1
        return out

    return {
        "held_out_group_ids": held_out_group_ids,
        "train": summarize_bucket(train),
        "test": summarize_bucket(test),
    }
