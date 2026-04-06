import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.io_utils import load_jsonl, write_jsonl
from src.prompts import make_model_input_record, get_system_prompt
"""This file holds the main logic for running inference with the Mistral 7B Inference model, 
both the LoRA adapter and the base model as a prompt baseline. It loads the model(s), runs 
inference on the test set examples, and writes the outputs to a JSONL file."""

def build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_base_model(model_id: str):
    bnb_config = build_quant_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model


def load_lora_model(base_model_id: str, adapter_path: str):
    tokenizer = load_tokenizer(base_model_id)
    base_model = load_quantized_base_model(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_from_messages(tokenizer, model, messages, max_new_tokens=320):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
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


def run_lora_eval(base_model: str, adapter_path: str, test_inputs: Path, output_file: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    tokenizer, model = load_lora_model(base_model, adapter_path)
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


def run_prompt_baseline(mode: str, base_model: str, test_inputs: Path, output_file: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    tokenizer = load_tokenizer(base_model)
    model = load_quantized_base_model(base_model)
    rows = load_jsonl(test_inputs)
    outputs = []
    total_start = time.perf_counter()
    system_prompt = get_system_prompt(mode)

    for row in rows:
        example_start = time.perf_counter()
        cleaned_record = make_model_input_record(row["input_record"])

        if mode == "raw_basic":
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Summarize this manufacturing record in a short paragraph:\n"
                        f"{cleaned_record}"
                    ),
                }
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(cleaned_record).replace("'", '"')},
            ]

        prediction = generate_from_messages(tokenizer, model, messages)
        elapsed = time.perf_counter() - example_start

        outputs.append({
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "source_kind": row["source_kind"],
            "is_synthetic": row["is_synthetic"],
            "prompt_mode": mode,
            "generation_seconds": round(elapsed, 4),
            "model_output": prediction,
        })

        print(f"{row['example_id']}: {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    write_jsonl(output_file, outputs)

    print(f"\nWrote predictions to {output_file}")
    print(f"Total inference time: {total_elapsed:.2f}s")
    print(f"Average time per example: {total_elapsed / len(rows):.2f}s")
