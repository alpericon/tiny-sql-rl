"""Paired comparison of model records files on the same questions (exact McNemar).

    python scripts/paired_compare.py results/baseline_A.records.jsonl results/baseline_B.records.jsonl [...]

The first file is the reference; every other file is compared against it.
"""

from __future__ import annotations

import json
import sys

from tinysql.stats import paired_summary


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    base_path, *cands = sys.argv[1:]
    base = load(base_path)
    for path in cands:
        s = paired_summary(base, load(path))
        print(f"\n{path}\n  vs {base_path}")
        print(f"  n={s['n']}  EX {s['base_ex']:.3f} -> {s['cand_ex']:.3f} ({100 * (s['cand_ex'] - s['base_ex']):+.1f}pp)"
              f"  wins {s['wins']} / losses {s['losses']}  McNemar p={s['p']:.4f}")
        print(f"  wins came from base: {s['wins_by_base_outcome']}")
        print(f"  errors base {s['base_breakdown']}  ->  cand {s['cand_breakdown']}")


if __name__ == "__main__":
    main()
