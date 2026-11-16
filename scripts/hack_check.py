"""Pick the 20 reward-1.0 rollouts most likely to be reward hacks, for reading by hand (W7).

Every top-reward sample scores exactly 1.0, so "highest reward" is ranked by suspicion instead:
the gold answer is trivially matchable (empty, or a single 0/1), the prediction has no FROM,
or it reads different tables than the gold query. Ties go to the largest SQL length gap.

    python scripts/hack_check.py results/hack_samples.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from tinysql.executor import execute_readonly

ROOT = Path(__file__).resolve().parents[1]
_TABLE = re.compile(r"\b(?:FROM|JOIN)\s+[`\"\[]?(\w+)", re.IGNORECASE)


def tables(sql: str) -> set[str]:
    return {t.lower() for t in _TABLE.findall(sql or "")}


def flags(s: dict, gold_rows) -> list[str]:
    f = []
    if not gold_rows:
        f.append("gold-empty")
    elif len(gold_rows) == 1 and len(gold_rows[0]) == 1 and gold_rows[0][0] in (0, 1):
        f.append("gold-0/1")
    if not re.search(r"\bFROM\b", s["pred_sql"] or "", re.IGNORECASE):
        f.append("no-FROM")
    if tables(s["pred_sql"]) != tables(s["gold"]):
        f.append("tables-differ")
    return f


def main():
    samples = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    db = ROOT / "data" / "spider" / "database"
    print("reward distribution:", Counter(s["reward"] for s in samples))
    top = [s for s in samples if s["reward"] == 1.0]
    for s in top:
        rows = execute_readonly(db / s["db_id"] / f"{s['db_id']}.sqlite", s["gold"]).rows
        s["flags"] = flags(s, rows)
        s["gold_rows"] = rows[:3]
    print("reward-1.0 flag counts:", Counter(f for s in top for f in s["flags"]), "of", len(top))
    seen, picked = set(), []
    for s in sorted(top, key=lambda s: (-len(s["flags"]), -abs(len(s["pred_sql"]) - len(s["gold"])))):
        key = (s["train_idx"], s["pred_sql"].lower())
        if key not in seen:
            seen.add(key)
            picked.append(s)
        if len(picked) == 20:
            break
    for k, s in enumerate(picked, 1):
        print(f"\n#{k} [{', '.join(s['flags']) or 'none'}] {s['db_id']} | {s['question']}")
        print("  gold:", " ".join(s["gold"].split()))
        print("  pred:", " ".join(s["pred_sql"].split()))
        print("  gold rows:", s["gold_rows"])


if __name__ == "__main__":
    main()
