"""Check the held-out Spider test split (PLAN-week11 H1) -> results/test_split_check.json.

Counts, database overlap with train/dev, exact question overlap, and that every test database and
gold query executes.

    python scripts/test_split_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tinysql.data import load_spider_split
from tinysql.executor import execute_readonly

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "spider"


def main():
    test = load_spider_split(S / "test.json")
    dev = load_spider_split(S / "dev.json")
    train = json.load(open(S / "train_spider.json", encoding="utf-8"))
    tdb, ddb, trdb = {e.db_id for e in test}, {e.db_id for e in dev}, {e["db_id"] for e in train}
    seen = {(e.db_id, e.question) for e in dev} | {(e["db_id"], e["question"]) for e in train}
    gold_fail = [e.db_id for e in test if not execute_readonly(S / "test_database" / e.db_id / f"{e.db_id}.sqlite", e.query).ok]
    out = {
        "n_questions": len(test), "n_databases": len(tdb),
        "databases_shared_with_train": sorted(tdb & trdb), "databases_shared_with_dev": sorted(tdb & ddb),
        "questions_shared_with_train_or_dev": sum((e.db_id, e.question) in seen for e in test),
        "databases_missing_on_disk": sorted(d for d in tdb if not (S / "test_database" / d / f"{d}.sqlite").exists()),
        "gold_queries_failing": len(gold_fail),
    }
    (ROOT / "results" / "test_split_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
