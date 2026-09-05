"""Merge a RelaCaTS-v1 LoRA adapter into its local base model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    output = Path(args.output_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty merged-model directory: {output}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, use_fast=False
    )
    merged = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(f"Merged model saved to {output}")


if __name__ == "__main__":
    main()

