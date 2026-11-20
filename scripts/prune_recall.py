"""Gold-table recall of the table pruner (PLAN-week10 P2/P3).

    python scripts/prune_recall.py train   # tuning grid, train questions only -> results/prune_recall_train.json
    python scripts/prune_recall.py dev     # frozen defaults, once           -> results/prune_recall_dev.json

A question is "kept" when every table its gold query reads is in the pruned set (a full-schema
fallback always counts as kept). Also reported: the share of tables shown and how often the pruner
falls back to the full schema.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from tinysql.data import load_spider_split
from tinysql.prune import gold_tables, profile, select_tables

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "spider" / "database"
GRID = [{"min_tables": m, "max_col_df": d, "prefix": px, "neighbours": nb}
        for m in (1, 2) for d in (1, 2, 3) for px in (False, True) for nb in (False, True)]


def measure(examples, **kw) -> dict:
    kept_q, share, fallback, small, misses = 0, [], 0, 0, []
    for e in examples:
        db = DB / e.db_id / f"{e.db_id}.sqlite"
        n = len(profile(str(db)).names)
        sel = select_tables(e.question, db, **kw)
        if sel is None:
            fallback += 1
            kept_q += 1
            share.append(1.0)
            continue
        ok = gold_tables(e.query) <= {t.lower() for t in sel}
        kept_q += ok
        share.append(len(sel) / n)
        if not ok and len(misses) < 40:
            misses.append({"db_id": e.db_id, "question": e.question, "gold_tables": sorted(gold_tables(e.query)), "kept": sel})
    n = len(examples)
    return {**kw, "n": n, "recall": round(kept_q / n, 4), "mean_share_of_tables": round(statistics.mean(share), 4),
            "fallback_rate": round(fallback / n, 4), "misses_example": misses}


def main():
    split = sys.argv[1]
    if split == "train":
        ex = load_spider_split(ROOT / "data" / "train_filtered.json")
        out = [measure(ex, **g) for g in GRID]
        for o in out:
            print({k: v for k, v in o.items() if k != "misses_example"})
    elif split == "dev":
        ex = load_spider_split(ROOT / "data" / "spider" / "dev.json")
        out = measure(ex)  # the frozen defaults of select_tables
        print({k: v for k, v in out.items() if k != "misses_example"})
    else:
        raise SystemExit("usage: prune_recall.py train|dev")
    (ROOT / "results" / f"prune_recall_{split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
