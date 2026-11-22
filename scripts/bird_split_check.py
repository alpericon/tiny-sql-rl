"""Check BIRD dev after download (PLAN-week12 B1) -> results/bird_split_check.json.

    BIRD_DIR=~/bird/dev ~/venvs/tinysql/bin/python scripts/bird_split_check.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from tinysql.executor import execute_readonly

ROOT = Path(__file__).resolve().parents[1]
BIRD = Path(os.environ.get("BIRD_DIR", Path.home() / "bird" / "dev"))


def main():
    dev = json.load(open(BIRD / "dev.json", encoding="utf-8"))
    dbs = sorted({e["db_id"] for e in dev})
    spider = {p.name for p in (ROOT / "data" / "spider" / "database").iterdir()} | {
        p.name for p in (ROOT / "data" / "spider" / "test_database").iterdir()}
    fails, slow = [], []
    for e in dev:
        db = BIRD / "dev_databases" / e["db_id"] / f"{e['db_id']}.sqlite"
        t0 = time.perf_counter()
        r = execute_readonly(db, e["SQL"], timeout_s=30.0, max_rows=10**9)
        dt = time.perf_counter() - t0
        if not r.ok:
            fails.append({"question_id": e["question_id"], "db_id": e["db_id"], "error": r.error})
        elif dt > 10:
            slow.append({"question_id": e["question_id"], "seconds": round(dt, 1)})
    zip_path = Path.home() / "bird" / "download" / "dev.zip"
    out = {
        "n_questions": len(dev), "n_databases": len(dbs), "databases": dbs,
        "difficulty": dict(Counter(e.get("difficulty") for e in dev)),
        "with_evidence": sum(bool(e.get("evidence")) for e in dev),
        "databases_shared_with_spider": sorted(set(dbs) & spider),
        "databases_missing": [d for d in dbs if not (BIRD / "dev_databases" / d / f"{d}.sqlite").exists()],
        "gold_failing_within_30s": fails, "gold_slower_than_10s": slow,
        "dev_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest() if zip_path.exists() else None,
    }
    (ROOT / "results" / "bird_split_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "databases"}, indent=1))


if __name__ == "__main__":
    main()
