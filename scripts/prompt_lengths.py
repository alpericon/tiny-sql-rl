"""Prompt-length check for PLAN-week7 (S1 and the cost row), written to results/prompt_lengths_rows3.json.

* sample_rows=0 prompts must equal the week-6 run's prompts (compared by prompt_chars per question).
* token lengths (Qwen3 chat template, thinking off) of first-attempt prompts with 0 and 3 example rows.

    ~/venvs/tinysql/bin/python scripts/prompt_lengths.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from tinysql.data import load_spider_split
from tinysql.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
WEEK6 = ROOT / "results" / "baseline_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground.records.jsonl"


def main():
    dev = load_spider_split(ROOT / "data" / "spider" / "dev.json")
    prior = [json.loads(l) for l in open(WEEK6, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    out = {}
    for k in (0, 3):
        prompts = []

        def gen(batch):
            prompts.extend(batch)
            return ["```sql\nSELECT 1\n```"] * len(batch)

        recs, _ = evaluate(dev, ROOT / "data" / "spider" / "database", gen, sample_rows=k)
        lens = [len(tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"])
                for m in prompts]
        out[f"rows{k}"] = {"n": len(lens), "median_tokens": statistics.median(lens), "max_tokens": max(lens),
                           "over_3072": sum(l > 3072 for l in lens),
                           "median_chars": statistics.median(r["prompt_chars"] for r in recs)}
        if k == 0:
            out["rows0_prompt_chars_equal_week6"] = sum(r["prompt_chars"] == p["prompt_chars"] for r, p in zip(recs, prior))
    out["note"] = "first-attempt prompts only; repair prompts add the failed answer and are longer (max seen 2,168 in the rows3 run)"
    (ROOT / "results" / "prompt_lengths_rows3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
