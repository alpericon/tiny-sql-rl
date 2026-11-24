"""Paired BIRD dev comparisons (exact McNemar on BIRD EX, via tinysql.bird.bird_pairs).

    python scripts/compare_bird.py              # PLAN-week13
    python scripts/compare_bird.py --week 14    # PLAN-week14, the seed runs

Reads results/bird_<tag>.records.jsonl and writes results/bird_compare_week<N>.json: each pair
(candidate over base) on all 1,534 questions and without formula_1, plus EX where both produced a
SQL block. Week 13: K3 and K4 are the pre-registered tests; the other two pairs are descriptive.
Week 14: S3 and S4 are the pre-registered tests; the others are descriptive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinysql.bird import bird_pairs
from tinysql.stats import ex_where_both_formatted

ROOT = Path(__file__).resolve().parents[1]
PAIRS_BY_WEEK = {13: {  # name: (candidate, base)
    "K3: B vs CA": ("B-v2-900", "CA-coder"),
    "K4: E vs CE": ("E-v2-900-full", "CE-coder-full"),
    "CA vs A (descriptive)": ("CA-coder", "A-base"),
    "CE vs CA (descriptive)": ("CE-coder-full", "CA-coder"),
}, 14: {  # B-<run> is the run's checkpoint selected on dev-200 (results/selection_<run>.json)
    "S3: v2-s1 vs A": ("B-v2-s1", "A-base"),
    "S4: v2-s2 vs A": ("B-v2-s2", "A-base"),
    "v2-s1 vs B (descriptive)": ("B-v2-s1", "B-v2-900"),
    "v2-s2 vs B (descriptive)": ("B-v2-s2", "B-v2-900"),
    "v2-s2 vs v2-s1 (descriptive)": ("B-v2-s2", "B-v2-s1"),
}}


def load(tag: str) -> list[dict]:
    with open(ROOT / "results" / f"bird_{tag}.records.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, choices=sorted(PAIRS_BY_WEEK), default=13)
    week = ap.parse_args().week
    out = {}
    if week == 14:  # the B-<run> tags are free text: check they hold the selected checkpoint
        from tinysql.seeds import check_selected_adapter

        for run in ("v2-s1", "v2-s2"):
            step = json.loads((ROOT / "results" / f"selection_{run}.json").read_text(encoding="utf-8"))["selected"]
            summary = json.loads((ROOT / "results" / f"bird_B-{run}.json").read_text(encoding="utf-8"))
            check_selected_adapter(summary["adapter"], run, step)
            out[f"selected {run}"] = step
    for name, (cand, base) in PAIRS_BY_WEEK[week].items():
        c, b = load(cand), load(base)
        res = bird_pairs(b, c)
        both = ex_where_both_formatted([r["format_ok"] for r in c], [r["match"] for r in c],
                                       [r["format_ok"] for r in b], [r["match"] for r in b])
        res["both_formatted"] = {"n": both["n"], "cand_ex": both["a_ex"], "base_ex": both["b_ex"]}
        res["candidate"], res["base"] = cand, base
        out[name] = res
    (ROOT / "results" / f"bird_compare_week{week}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, res in out.items():
        if name.startswith("selected "):
            continue
        a = res["all"]
        print(f"{name}: {a['cand_ex']:.3f} vs {a['base_ex']:.3f}, {a['wins']} / {a['losses']}, p={a['p']:.2g}")


if __name__ == "__main__":
    main()
