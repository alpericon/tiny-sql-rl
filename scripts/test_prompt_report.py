"""Test-split prompt lengths and frozen-pruner recall, before any generation (PLAN-week11 H3).

Reporting only: nothing is tuned on these numbers. Writes results/prompt_lengths_test.json and
results/prune_recall_test.json.

    ~/venvs/tinysql/bin/python scripts/test_prompt_report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from tinysql.data import load_spider_split, spider_split
from tinysql.prompt import build_messages, serialize_schema
from tinysql.prune import gold_tables, profile, select_tables

ROOT = Path(__file__).resolve().parents[1]
GUARD = 3072


def main():
    questions, db_dir = spider_split(ROOT / "data" / "spider", "test")
    test = load_spider_split(questions)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    n_tok = lambda m: len(tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                              add_special_tokens=False)["input_ids"])
    lens = {"A_schema_only": [], "D_rows3": [], "E_rows3_pruned": []}
    kept, share, fallback, fewer, misses = 0, [], 0, 0, []
    for e in test:
        db = db_dir / e.db_id / f"{e.db_id}.sqlite"
        sel = select_tables(e.question, db)
        n = len(profile(str(db)).names)
        lens["A_schema_only"].append(n_tok(build_messages(serialize_schema(db), e.question)))
        lens["D_rows3"].append(n_tok(build_messages(serialize_schema(db, 3), e.question)))
        lens["E_rows3_pruned"].append(n_tok(build_messages(serialize_schema(db, 3, tables=sel), e.question)))
        if sel is None:
            fallback += 1
            kept += 1
            share.append(1.0)
            continue
        ok = gold_tables(e.query) <= {t.lower() for t in sel}
        kept += ok
        share.append(len(sel) / n)
        fewer += len(sel) < n
        if not ok:
            misses.append({"db_id": e.db_id, "question": e.question, "gold_tables": sorted(gold_tables(e.query)), "kept": sel})
    pl = {k: {"median": statistics.median(v), "max": max(v), f"over_{GUARD}": sum(x > GUARD for x in v)} for k, v in lens.items()}
    pr = {"n": len(test), "recall": round(kept / len(test), 4), "mean_share_of_tables": round(statistics.mean(share), 4),
          "fallback_rate": round(fallback / len(test), 4), "questions_with_fewer_tables": fewer,
          "n_misses": len(misses), "misses_by_db": {d: sum(m["db_id"] == d for m in misses) for d in sorted({m["db_id"] for m in misses})},
          "misses_example": misses[:40]}
    (ROOT / "results" / "prompt_lengths_test.json").write_text(json.dumps(pl, indent=1), encoding="utf-8")
    (ROOT / "results" / "prune_recall_test.json").write_text(json.dumps(pr, indent=1), encoding="utf-8")
    print(json.dumps(pl, indent=1))
    print({k: v for k, v in pr.items() if k != "misses_example"})


if __name__ == "__main__":
    main()
