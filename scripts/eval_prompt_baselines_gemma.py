import argparse
import sys
from pathlib import Path
"""Mirrors eval_prompt_baselines.py exactly."""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import VALID_PROMPT_MODES
from src.modeling.gemma_inference import run_prompt_baseline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(VALID_PROMPT_MODES))
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--test-inputs", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_prompt_baseline(
        mode=args.mode,
        base_model=args.base_model,
        test_inputs=args.test_inputs,
        output_file=args.output_file,
        auth_token=args.hf_token,
    )


if __name__ == "__main__":
    main()