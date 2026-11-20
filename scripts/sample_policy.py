"""Re-create GRPO-style rollouts from a checkpoint for the reward-hacking check (PLAN-week2 W7).

The trainer did not log completions, so this samples G completions per training prompt with the
training decoding (temperature 1.0, no top-p/top-k, 256 new tokens, thinking off) and scores each
with the training reward. HF generate stands in for vLLM; the sampling distribution is the same.

    ~/venvs/tinysql/bin/python scripts/sample_policy.py --adapter checkpoints/main/checkpoint-1000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tinysql.data import load_exclusions, load_spider_split
from tinysql.grpo_data import build_grpo_records, rows_or_schema_only
from tinysql.reward import extract_sql, sql_reward

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hack_samples.jsonl")
    ap.add_argument("--sample-rows", type=int, default=0,
                    help="example rows per table, with train_grpo's fallback to schema-only over --max-prompt-length (PLAN-week8)")
    ap.add_argument("--max-prompt-length", type=int, default=3000)
    args = ap.parse_args()

    def first_turn(sample_rows):
        return build_grpo_records(
            load_spider_split(ROOT / "data" / "train_filtered.json"),
            ROOT / "data" / "spider" / "database",
            load_exclusions(ROOT / "data" / "spokenspider_exclude.json"),
            sample_rows=sample_rows,
        )

    records = first_turn(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.sample_rows > 0:
        n_tokens = lambda r: len(tok(tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                                             enable_thinking=False), add_special_tokens=False)["input_ids"])
        records, n_fallback = rows_or_schema_only(first_turn(args.sample_rows), records, n_tokens, args.max_prompt_length)
        print(f"sample rows: {n_fallback} prompts fell back to schema-only", flush=True)
    idx = sorted(random.Random(args.seed).sample(range(len(records)), args.n_prompts))

    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.bfloat16, device_map="cuda:0")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    torch.manual_seed(args.seed)

    with args.out.open("w", encoding="utf-8") as f, torch.inference_mode():
        for k, i in enumerate(idx):
            r = records[i]
            text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok([text], return_tensors="pt").to(model.device)
            out = model.generate(
                **enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                max_new_tokens=args.max_new_tokens, num_return_sequences=args.num_generations,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            comps = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for g, c in enumerate(comps):
                f.write(json.dumps({
                    "train_idx": i, "g": g, "db_id": r["db_id"], "question": r["prompt"][-1]["content"].rsplit("\n", 1)[-1],
                    "gold": r["gold"], "completion": c, "pred_sql": extract_sql(c),
                    "reward": sql_reward(c, r["gold"], r["db_path"]),
                }, ensure_ascii=False) + "\n")
            print(f"{k + 1}/{len(idx)}", flush=True)


if __name__ == "__main__":
    main()
