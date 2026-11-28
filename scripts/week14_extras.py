"""PLAN-week14, descriptive numbers in results/week14_results.md: the GRPO gain per seed and its spread,
the release seed's rank, selection, dev-200 curves, training reward and KL, format failures, run times.

    python scripts/week14_extras.py
"""
import csv, json
from pathlib import Path

R = Path("results")
J = lambda p: json.loads((R / p).read_text(encoding="utf-8"))
L = lambda p: [json.loads(l) for l in open(R / p, encoding="utf-8")]
RUNS = {"v2": 3407, "v2-s1": 1, "v2-s2": 2}
sel = {"v2": 900, **{r: J(f"selection_{r}.json")["selected"] for r in ("v2-s1", "v2-s2")}}
spider_tag = {"v2": "B-v2-900", "v2-s1": "B-v2-s1", "v2-s2": "B-v2-s2"}
bird_tag = {"v2": "B-v2-900", "v2-s1": "B-v2-s1", "v2-s2": "B-v2-s2"}
ot = J("official_ex_test.json")["models"]
bird = {t: J(f"bird_{t}.json") for t in ["A-base", *bird_tag.values()]}

gains = {"spider": {}, "bird": {}}
for run in RUNS:
    gains["spider"][run] = 100 * (ot[spider_tag[run]]["official_ex"] - ot["A-base"]["official_ex"])
    gains["bird"][run] = 100 * (bird[bird_tag[run]]["execution_accuracy"] - bird["A-base"]["execution_accuracy"])
for bench, g in gains.items():
    vals = list(g.values())
    order = sorted(g, key=g.get)
    print(bench, {k: round(v, 2) for k, v in g.items()}, "mean", round(sum(vals) / 3, 2), "range", round(min(vals), 2), round(max(vals), 2),
          "rank of v2 (1=lowest):", order.index("v2") + 1, "order", order)

for run in RUNS:
    curve = {n: J(f"dev200_Qwen__Qwen3-1.7B_{run}-checkpoint-{n}.json")["execution_accuracy"] for n in range(100, 1001, 100)}
    print(run, "selected", sel[run], "dev200", {k: round(v, 3) for k, v in curve.items()})
    log = [r for r in csv.DictReader(open(R / f"train_log_{run}.csv", encoding="utf-8")) if r["reward"]]
    mean = lambda rs, k: sum(float(r[k]) for r in rs) / len(rs)
    parts = [log[:100], log[450:550], log[-100:]]
    print("   reward first/mid/last 100", [round(mean(p, "reward"), 3) for p in parts],
          "kl last 100", round(mean(log[-100:], "kl"), 3), "zero-std groups", round(mean(log, "frac_reward_zero_std"), 3))
    peak = [r for r in csv.DictReader(open(R / f"train_log_{run}.csv", encoding="utf-8")) if r.get("peak_reserved_gib")]
    tlast = peak[-1]
    steps = [r for r in csv.DictReader(open(R / f"train_log_{run}.csv", encoding="utf-8")) if r.get("step_time")]
    print("   training steps h", round(sum(float(r["step_time"]) for r in steps) / 3600, 2), "steps", len(steps),
          "peak reserved", tlast.get("peak_reserved_gib"), "max temp", max(float(r["gpu_temp_c"]) for r in peak if r.get("gpu_temp_c")))
    tst = L(f"test_Qwen__Qwen3-1.7B_{run}-checkpoint-{sel[run]}.records.jsonl")
    brd = L(f"bird_{bird_tag[run]}.records.jsonl")
    print("   format failures: spider test", sum(not r["format_ok"] for r in tst), "bird dev", sum(not r["format_ok"] for r in brd),
          "| spider test official", round(100 * ot[spider_tag[run]]["official_ex"], 2), "bird", round(100 * bird[bird_tag[run]]["execution_accuracy"], 2))
    if f"{spider_tag[run]}-900" in ot:
        print("   step 900 on test (official)", round(100 * ot[f"{spider_tag[run]}-900"]["official_ex"], 2))
print("A-base spider", round(100 * ot["A-base"]["official_ex"], 2), "bird", round(100 * bird["A-base"]["execution_accuracy"], 2))

c = J("bird_compare_week14.json")
for k, v in c.items():
    if isinstance(v, dict):
        print(k, {kk: (vv["n"], round(100 * vv["cand_ex"], 2), round(100 * vv["base_ex"], 2), vv["wins"], vv["losses"], f"{vv['p']:.2g}")
                  for kk, vv in v.items() if kk in ("all", "without_formula_1")})
