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


def train_main():
    """PLAN-week8 R2: token lengths of the v2 training prompts (pass-rate filtered) with 0 and 3 rows,
    built exactly as scripts/train_grpo.py builds them. Written to results/prompt_lengths_train_rows3.json."""
    from tinysql.data import load_exclusions
    from tinysql.grpo_data import build_grpo_records, filter_by_pass_rate

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    rates = [json.loads(l) for l in open(ROOT / "data" / "train_pass_rate.jsonl", encoding="utf-8")]
    out = {}
    for k in (0, 3):
        records = filter_by_pass_rate(build_grpo_records(
            load_spider_split(ROOT / "data" / "train_filtered.json"),
            ROOT / "data" / "spider" / "database",
            load_exclusions(ROOT / "data" / "spokenspider_exclude.json"),
            drop_empty_gold=True, sample_rows=k), rates)
        lens = sorted((len(tok(tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False), add_special_tokens=False)["input_ids"]),
                       r["db_id"]) for r in records)
        toks = [l for l, _ in lens]
        out[f"rows{k}"] = {"n": len(toks), "median_tokens": statistics.median(toks), "max_tokens": toks[-1],
                           "p99_tokens": toks[int(0.99 * (len(toks) - 1))],
                           "over_2304": sum(t > 2304 for t in toks), "over_3000": sum(t > 3000 for t in toks),
                           "longest_dbs": sorted({d for _, d in lens[-20:]})}
    (ROOT / "results" / "prompt_lengths_train_rows3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    train_main() if "--train" in sys.argv else main()
