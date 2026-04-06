"""This file holds the logic for how we format the examples for training and testing."""
def build_train_row(example):
    return {
        "example_id": example["example_id"],
        "group_id": example["group_id"],
        "source_kind": example["source_kind"],
        "is_synthetic": example["is_synthetic"],
        "messages": example["messages"] + [
            {
                "role": "assistant",
                "content": example["target_summary"],
            }
        ],
    }


def build_test_input_row(example):
    return {
        "example_id": example["example_id"],
        "group_id": example["group_id"],
        "source_kind": example["source_kind"],
        "is_synthetic": example["is_synthetic"],
        "messages": example["messages"],
    }


def build_test_gold_row(example):
    return {
        "example_id": example["example_id"],
        "group_id": example["group_id"],
        "source_kind": example["source_kind"],
        "is_synthetic": example["is_synthetic"],
        "target_summary": example["target_summary"],
        "input_record": example["input_record"],
    }
