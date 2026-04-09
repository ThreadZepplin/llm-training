# LLM-Training
Research into prompt engineering versus fine-tuning three LLMs

# Commands you can run (WIP)
Note: you can run any command with --help to see usage and options
1. Preprocess raw files
python scripts/preprocess_data.py --input-dir data/raw --output-file data/processed/operations.jsonl

2. Generate synthetic operator-style records

python scripts/generate_synthetic_records.py \
  --input-file data/processed/operations.jsonl \
  --output-file data/processed/synthetic_operator_records.jsonl

3. Build dataset splits

python scripts/build_dataset_splits.py \
  --real-file data/processed/operations.jsonl \
  --synthetic-file data/processed/synthetic_operator_records.jsonl \
  --out-all data/processed/dataset_all.jsonl \
  --out-train data/processed/dataset_train.jsonl \
  --out-test data/processed/dataset_test_manual.jsonl \
  --out-meta data/processed/dataset_split_metadata.json

4. Prepare fine-tuning files

python scripts/prepare_finetune_files.py \
  --train-input data/processed/dataset_train.jsonl \
  --test-input data/processed/dataset_test_manual.jsonl \
  --out-train data/processed/finetune_train.jsonl \
  --out-test-inputs data/processed/finetune_test_inputs.jsonl \
  --out-test-gold data/processed/finetune_test_gold.jsonl

5. Train LoRA model

python scripts/train_mistral_lora.py --config configs/mistral_lora.json

6. Evaluate LoRA model

python scripts/eval_mistral_lora.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-path outputs/mistral7b_lora_manufacturing \
  --test-inputs data/processed/finetune_test_inputs.jsonl \
  --output-file outputs/mistral7b_test_predictions.jsonl

7. Run prompt baseline

python scripts/eval_prompt_baselines.py \
  --mode structured \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --test-inputs data/processed/finetune_test_gold.jsonl \
  --output-file outputs/mistral7b_prompt_structured.jsonl

8. Export readable summaries

python scripts/export_prediction_summaries.py outputs/mistral7b_prompt_structured.jsonl
