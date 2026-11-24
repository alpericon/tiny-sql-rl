"""PLAN-week13 K3, descriptive numbers in results/week13_results.md: levels, format failures, 256-token hits,
EX where both have a SQL block (Spider test, official rows), BIRD error mix and pair summaries.

    ~/venvs/tinysql/bin/python scripts/week13_extras.py
"""
import json, sys
from collections import Counter
sys.path.insert(0, "src")
from tinysql.stats import ex_where_both_formatted, mcnemar_exact
R = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
oe = json.load(open("results/official_ex_test.json"))["models"]
for k, v in oe.items():
    print(f"{k:42s} official {v['official_ex']:.4f} ours {v['ours_ex']:.4f}")
stems = {"A-base": "Qwen__Qwen3-1.7B", "B-v2-900": "Qwen__Qwen3-1.7B_v2-checkpoint-900",
         "E-v2-900-repair-ground-rows3-prune": "Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3_prune",
         "CA-coder": "Qwen__Qwen2.5-Coder-1.5B-Instruct",
         "CE-coder-repair-ground-rows3-prune": "Qwen__Qwen2.5-Coder-1.5B-Instruct_repair_ground_rows3_prune"}
rec = {k: R(f"results/test_{s}.records.jsonl") for k, s in stems.items()}
off = {k: [json.loads(l)["official"] for l in open(f"results/official_test/{k}.rows.jsonl")] for k in stems}
for k in stems:
    s = json.load(open(f"results/test_{stems[k]}.json"))
    print("spider", k, "fmt_fail", sum(not r["format_ok"] for r in rec[k]), "hit", round(s.get("hit_max_new_tokens_rate") or -1, 4), "wall", s.get("wall_seconds"), "rescued", s.get("repair_rescued"))
for c, b in [("B-v2-900", "CA-coder"), ("E-v2-900-repair-ground-rows3-prune", "CE-coder-repair-ground-rows3-prune"), ("CA-coder", "A-base")]:
    print("spider both-formatted", c, b, ex_where_both_formatted([r["format_ok"] for r in rec[c]], off[c], [r["format_ok"] for r in rec[b]], off[b]))
    w = sum(x and not y for x, y in zip(off[c], off[b])); l = sum(y and not x for x, y in zip(off[c], off[b]))
    wf = sum(x and not y and not rb["format_ok"] for x, y, rb in zip(off[c], off[b], rec[b]))
    print("   wins", w, "of which base format fail", wf, "losses", l)
for t in ["A-base", "B-v2-900", "E-v2-900-full", "CA-coder", "CE-coder-full"]:
    s = json.load(open(f"results/bird_{t}.json")); rs = R(f"results/bird_{t}.records.jsonl")
    err = Counter("format" if not r["format_ok"] else "no such column" if r["error"] and "no such column" in r["error"] else "other error" if not r["exec_ok"] else "wrong" if not r["match"] else "match" for r in rs)
    print("bird", t, round(s["execution_accuracy"], 4), {k: round(v["ex"], 3) for k, v in s["by_difficulty"].items()}, dict(err), "hit", s.get("hit_max_new_tokens_rate"), "wall", s["wall_seconds"], "rescued", s.get("repair_rescued"), "maxtok", s["max_prompt_tokens"])
c = json.load(open("results/bird_compare_week13.json"))
for k, v in c.items():
    print(k, {kk: (vv["n"], vv["wins"], vv["losses"], f"{vv['p']:.2g}", round(vv["cand_ex"]-vv["base_ex"], 4)) for kk, vv in v.items() if kk.startswith(("all", "without"))}, v["both_formatted"], "wins by base outcome", v["all"]["wins_by_base_outcome"])
