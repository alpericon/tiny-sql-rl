"""Create the fixed dev-200 subset (PLAN-week2 W3).

Eligible: dev questions whose gold query runs in the sandbox (data/dev_clean.json).
The dev-200 baseline is NOT taken from the full-dev records: with 4-bit weights, batch
composition changes ~14% of greedy completions, so baseline and checkpoints must be run
with the same command (scripts/eval_baseline.py --subset data/dev200_ids.json).

Usage: uv run python scripts/make_dev_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tinysql.data import load_spider_split
from tinysql.subset import load_subset, select_subset

ROOT = Path(__file__).resolve().parents[1]
SEED, N = 3407, 200


def main():
    dev = load_spider_split(ROOT / "data" / "spider" / "dev.json")
    clean = {(e["db_id"], e["question"], e["query"]) for e in json.loads((ROOT / "data" / "dev_clean.json").read_text(encoding="utf-8"))}
    eligible = [i for i, e in enumerate(dev) if (e.db_id, e.question, e.query) in clean]

    ids = select_subset(dev, eligible, n=N, seed=SEED)
    load_subset(dev, ids)  # self-check
    (ROOT / "data" / "dev200_ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(ids)} of {len(eligible)} eligible dev questions, {len({r['db_id'] for r in ids})} databases, seed {SEED}")


if __name__ == "__main__":
    main()
