"""Merge a GRPO LoRA adapter into the bf16 base model (PLAN-week4 Y2).

The adapter was trained on the 4-bit base. Merging into bf16 weights is therefore not bit-identical
to the evaluated adapter model, so the merged model is re-evaluated (bf16 and nf4) before release.
The tokenizer and chat template come from the base model, exactly as in every evaluation.

    ~/venvs/tinysql/bin/python scripts/merge_adapter.py --adapter checkpoints/v2/checkpoint-900 --out checkpoints/v2-900-merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(str(args.out))
    (args.out / "merge_info.json").write_text(json.dumps(
        {"base": args.model, "adapter": str(args.adapter), "dtype": "bfloat16"}, indent=1), encoding="utf-8")
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()
