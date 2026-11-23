"""Paired BIRD dev comparisons for PLAN-week13 (exact McNemar on BIRD EX, via tinysql.bird.bird_pairs).

    python scripts/compare_bird.py

Reads results/bird_<tag>.records.jsonl and writes results/bird_compare_week13.json: each pair
(candidate over base) on all 1,534 questions and without formula_1, plus EX where both produced a
SQL block. K3 and K4 are the pre-registered tests; the other two pairs are descriptive.
"""

from __future__ import annotations

import json
from pathlib import Path

from tinysql.bird import bird_pairs
from tinysql.stats import ex_where_both_formatted

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {  # name: (candidate, base)
    "K3: B vs CA": ("B-v2-900", "CA-coder"),
    "K4: E vs CE": ("E-v2-900-full", "CE-coder-full"),
    "CA vs A (descriptive)": ("CA-coder", "A-base"),
    "CE vs CA (descriptive)": ("CE-coder-full", "CA-coder"),
}


def load(tag: str) -> list[dict]:
    with open(ROOT / "results" / f"bird_{tag}.records.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    out = {}
    for name, (cand, base) in PAIRS.items():
        c, b = load(cand), load(base)
        res = bird_pairs(b, c)
        res["both_formatted"] = ex_where_both_formatted([r["format_ok"] for r in c], [r["match"] for r in c],
                                                        [r["format_ok"] for r in b], [r["match"] for r in b])
        res["candidate"], res["base"] = cand, base
        out[name] = res
    (ROOT / "results" / "bird_compare_week13.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, res in out.items():
        a = res["all"]
        print(f"{name}: {a['cand_ex']:.3f} vs {a['base_ex']:.3f}, {a['wins']} / {a['losses']}, p={a['p']:.2g}")


if __name__ == "__main__":
    main()
