import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.io_utils import load_jsonl, write_jsonl
from src.prompts import make_model_input_record, get_system_prompt
"""This file holds the main logic for running inference with an OLMo base model and LoRA adapter."""

def build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def resolve_auth_token(auth_token: str | None = None) -> str | None:
    return (
        auth_token
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )


def load_tokenizer(model_id: str, auth_token: str | None = None):
    auth_token = resolve_auth_token(auth_token)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        use_auth_token=auth_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_base_model(model_id: str, auth_token: str | None = None):
    auth_token = resolve_auth_token(auth_token)
    bnb_config = build_quant_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_auth_token=auth_token,
    )
    model.eval()
    return model


def load_lora_model(base_model_id: str, adapter_path: str, auth_token: str | None = None):
    tokenizer = load_tokenizer(base_model_id, auth_token)
    base_model = load_quantized_base_model(base_model_id, auth_token)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def messages_to_text(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if role == "system":
            label = "System"
        elif role == "user":
            label = "User"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = role.capitalize() or "Message"

        parts.append(f"{label}: {content}")

    return "\n".join(parts)


def generate_from_messages(tokenizer, model, messages, max_new_tokens=320):
    prompt = messages_to_text(messages)
    if not prompt.endswith("\n"):
        prompt += "\n"
    prompt += "Assistant: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_lora_eval(base_model: str, adapter_path: str, test_inputs: Path, output_file: Path, auth_token: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    tokenizer, model = load_lora_model(base_model, adapter_path, auth_token)
    rows = load_jsonl(test_inputs)
    outputs = []

    total_start = time.perf_counter()

    for row in rows:
        example_start = time.perf_counter()
        prediction = generate_from_messages(tokenizer, model, row["messages"])
        elapsed = time.perf_counter() - example_start

        outputs.append({
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "source_kind": row["source_kind"],
            "is_synthetic": row["is_synthetic"],
            "generation_seconds": round(elapsed, 4),
            "model_output": prediction,
        })

        print(f"{row['example_id']}: {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    write_jsonl(output_file, outputs)

    print(f"\nWrote predictions to {output_file}")
    print(f"Total inference time: {total_elapsed:.2f}s")
    print(f"Average time per example: {total_elapsed / len(rows):.2f}s")
