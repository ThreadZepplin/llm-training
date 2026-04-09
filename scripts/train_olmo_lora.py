import argparse
import sys
from pathlib import Path
"""This script holds the main logic for fine-tuning the OLMo model with LoRA. """
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.olmo_lora import load_config, train_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token for gated/private checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.hf_token:
        config["hf_token"] = args.hf_token
    output_dir = train_from_config(config)
    print(f"Saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
