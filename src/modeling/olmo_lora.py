import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

"""This file holds the main logic for fine-tuning OLMo with LoRA."""

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(config: dict):
    return LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
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
        token=auth_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


def load_train_dataset(train_path: str, tokenizer):
    dataset = load_dataset("json", data_files=train_path, split="train")

    def add_text(example):
        example["text"] = messages_to_text(example["messages"])
        return example

    return dataset.map(add_text)


def tokenize_train_dataset(dataset, tokenizer, max_length: int):
    def tokenize_fn(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )


def train_from_config(config: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    model_id = config["model_id"]
    train_path = config["train_path"]
    output_dir = config["output_dir"]
    auth_token = resolve_auth_token(config.get("hf_token"))

    tokenizer = load_tokenizer(model_id, auth_token)
    dataset = load_train_dataset(train_path, tokenizer)
    dataset = tokenize_train_dataset(dataset, tokenizer, config["max_seq_length"])

    bnb_config = build_quant_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        token=auth_token,
    )
    model.config.use_cache = False

    peft_config = build_lora_config(config)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_epochs"],
        logging_steps=config.get("logging_steps", 1),
        save_strategy=config.get("save_strategy", "epoch"),
        report_to=config.get("report_to", "none"),
        bf16=True,
        gradient_checkpointing=True,
        fp16=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
