import argparse
import sys
from pathlib import Path
"""This script holds the main logic for running inference with the OLMo model and a LoRA adapter."""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.olmo_inference import run_lora_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--test-inputs", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token for gated/private checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    run_lora_eval(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        test_inputs=args.test_inputs,
        output_file=args.output_file,
        auth_token=args.hf_token,
    )

if __name__ == "__main__":
    main()
