"""Sanity-check Spider against our pipeline and write the filtered training set.

1. Apply the SpokenSpider leakage filter to Spider train and report what it removes.
2. Run every gold query (train after filtering + dev) through the sandboxed executor;
   examples whose gold fails or exceeds max_rows are listed and dropped.
3. Write data/train_filtered.json, data/dev_clean.json and results/data_check.json.

Usage: uv run python scripts/check_data.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from tinysql.data import assert_no_leakage, filter_leakage, load_exclusions, load_spider_split
from tinysql.executor import execute_readonly

ROOT = Path(__file__).resolve().parents[1]
SPIDER = ROOT / "data" / "spider"
MAX_ROWS = 10_000


def db_path(db_id: str) -> Path:
    return SPIDER / "database" / db_id / f"{db_id}.sqlite"


def gold_check(examples):
    ok, bad = [], []
    for e in examples:
        res = execute_readonly(db_path(e.db_id), e.query, timeout_s=30, max_rows=MAX_ROWS)
        if res.ok and not res.truncated:
            ok.append(e)
        else:
            bad.append({**asdict(e), "error": res.error or "truncated"})
    return ok, bad


def main():
    train = load_spider_split(SPIDER / "train_spider.json")
    dev = load_spider_split(SPIDER / "dev.json")
    exclusions = load_exclusions(ROOT / "data" / "spokenspider_exclude.json")

    kept, removed = filter_leakage(train, exclusions)
    assert_no_leakage(kept, exclusions)
    raw = json.loads((ROOT / "data" / "spokenspider_exclude.json").read_text(encoding="utf-8"))
    first100_match = sum(
        1 for i, r in enumerate(raw) if train[i].db_id == r["db_id"] and train[i].question.strip() == r["question"].strip()
    )
    removed_idx = sorted(i for i, e in enumerate(train) if e in set(removed))

    train_ok, train_bad = gold_check(kept)
    dev_ok, dev_bad = gold_check(dev)

    (ROOT / "data" / "train_filtered.json").write_text(
        json.dumps([asdict(e) for e in train_ok], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (ROOT / "data" / "dev_clean.json").write_text(
        json.dumps([asdict(e) for e in dev_ok], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    report = {
        "train_total": len(train),
        "spokenspider_first100_match_train_spider_json_order": first100_match,
        "leakage_removed": len(removed),
        "leakage_removed_beyond_index_99": sum(1 for i in removed_idx if i >= 100),
        "leakage_removed_index_range": [removed_idx[0], removed_idx[-1]] if removed_idx else None,
        "train_gold_failed": len(train_bad),
        "train_final": len(train_ok),
        "dev_total": len(dev),
        "dev_gold_failed": len(dev_bad),
        "dev_final": len(dev_ok),
        "gold_failure_reasons": Counter(b["error"].split(":")[0] for b in train_bad + dev_bad),
        "failed_examples": (train_bad + dev_bad)[:20],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "failed_examples"}, indent=1))


if __name__ == "__main__":
    main()
