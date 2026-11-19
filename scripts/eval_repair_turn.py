"""Paired repair-turn evaluation (PLAN-week5 Z4, primary measure).

Replays the failed first attempts of a fixed reference run (default: v2-900 + repair on full dev, Y9)
through a model's repair turn only, so every model repairs the same inputs.

    ~/venvs/tinysql/bin/python scripts/eval_repair_turn.py --adapter checkpoints/v3/checkpoint-500
    ~/venvs/tinysql/bin/python scripts/eval_repair_turn.py --adapter checkpoints/v2/checkpoint-900   # control
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPIDER = ROOT / "data" / "spider"
REFERENCE = ROOT / "results" / "baseline_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair.records.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from tinysql.data import load_spider_split
    from tinysql.evaluate import evaluate_repair_turn
    from tinysql.hf import HFGenerator

    prior = [json.loads(l) for l in open(args.reference, encoding="utf-8")]
    dev = load_spider_split(SPIDER / "dev.json")
    out = evaluate_repair_turn(prior, dev, SPIDER / "database", HFGenerator(args.model, args.adapter),
                               batch_size=args.batch_size)
    same = sum((r["pred_sql"] or "").strip() == (p["first_pred_sql"] or "").strip()
               for r, p in zip(out, [p for p in prior if p.get("repaired")]))
    summary = {
        "reference": args.reference.name, "adapter": str(args.adapter), "inputs": len(out),
        "rescued": sum(r["match"] for r in out),
        "reference_rescued": sum(r["prior_match"] for r in out),
        "now_executes_but_wrong": sum(r["exec_ok"] and not r["match"] for r in out),
        "still_failing": sum(not r["exec_ok"] for r in out),
        "repeated_identical_sql": same,
    }
    tag = f"repair_turn_{args.adapter.parent.name}-{args.adapter.name}"
    (ROOT / "results" / f"{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (ROOT / "results" / f"{tag}.records.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
