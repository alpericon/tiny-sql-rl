"""Check that the numbers quoted in README.md are recomputed from results/ (PLAN-week4 Y7).

Each check renders a string from the result files and asserts that the README contains it.
    python scripts/check_readme.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def j(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def recs(name):
    return [json.loads(l) for l in open(R / name, encoding="utf-8")]


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def main():
    checks = []
    off = j("official_ex.json")
    B = "baseline_Qwen__Qwen3-1.7B"
    ours = {"base": j(f"{B}.json"), "main": j(f"{B}_main-checkpoint-1000.json"),
            "v2": j(f"{B}_v2-checkpoint-900.json"), "bin": j(f"{B}_v2-binary-checkpoint-800.json")}
    om = off["models"]
    for k, o in [("base", "base"), ("main", "main-1000"), ("v2", "v2-900"), ("bin", "v2-binary-800")]:
        checks.append(pct(ours[k]["execution_accuracy"]))
        checks.append(pct(om[o]["official_ex"]))
    p = off["pairs_official"]
    for key in ["v2-900 vs base", "main-1000 vs base", "v2-900 vs main-1000", "v2-900 vs v2-binary-800"]:
        checks.append(f"{p[key]['wins']} / {p[key]['losses']}")
    checks.append(f"p={p['v2-900 vs main-1000']['p']:.2f}")
    checks.append(f"+{100 * (om['v2-900']['official_ex'] - om['base']['official_ex']):.1f} points")

    v2r = recs(f"{B}_v2-checkpoint-900.records.jsonl")
    fmt = sum(not r["format_ok"] for r in v2r)
    wrong = sum(r["format_ok"] and r["exec_ok"] and not r["match"] for r in v2r)
    fail = sum(not r["match"] for r in v2r)
    checks.append(f"{wrong} of {fail} failures")
    checks.append(f"only {fmt} are format failures")

    voice = j("voice_Qwen__Qwen3-1.7B_v2-checkpoint-900.json")["conditions"]
    vbase = j("voice_Qwen__Qwen3-1.7B.json")["conditions"]
    for c in ["text", "clean", "20db", "10db", "0db"]:
        checks.append(f"{round(100 * voice[c]['ours_ex'])}%")
        checks.append(f"{round(100 * vbase[c]['ours_ex'])}%")
        if c != "text":
            checks.append(f"{round(100 * voice[c]['sqlcoder_ex'])}%")

    lat = {k: j(f)["summary"] for k, f in [("merged", "latency_ours_checkpoints__v2-900-merged_y2.json"),
                                           ("sqlcoder", "latency_sqlcoder_y2.json")]}
    checks.append(f"{lat['merged']['text_to_sql']['median_s']:.2f} vs {lat['sqlcoder']['text_to_sql']['median_s']:.2f} s")
    checks.append(f"reproduces {lat['sqlcoder']['same_sql_as_voice_to_sql']}/100")
    gguf_q4 = j("dev200_gguf-v2-900-merged-Q4_K_M.json")["execution_accuracy"]
    bf16 = [recs("baseline_v2-900-merged-bf16.records.jsonl")[i["idx"]] for i in json.load(open(ROOT / "data" / "dev200_ids.json"))]
    bf16_ex = sum(r["match"] for r in bf16) / len(bf16)
    checks.append(f"−{100 * (bf16_ex - gguf_q4):.1f} points")
    checks.append(f"{sum(1 for l in open(ROOT / 'data' / 'train_pass_rate.jsonl') if 0 < json.loads(l)['n_correct'] < json.loads(l)['n']):,} of")

    # Y9 repair
    rep = j(f"{B}_v2-checkpoint-900_repair.json")
    checks.append(pct(rep["execution_accuracy"]))
    checks.append(pct(om["v2-900-repair"]["official_ex"]))
    checks.append(f"{rep['repair_rescued']} more rescued")
    checks.append(f"rescues {rep['repair_rescued']} of {rep['repair_attempts']}")
    rrecs = recs(f"{B}_v2-checkpoint-900_repair.records.jsonl")
    same = sum(r["repaired"] and (r["pred_sql"] or "").strip() == (r["first_pred_sql"] or "").strip() for r in rrecs)
    checks.append(f"In {same} of the retries")
    vrep = j("voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair.json")["conditions"]
    for c in ["text", "clean", "20db", "10db", "0db"]:
        checks.append(f"{round(100 * vrep[c]['ours_ex'])}%")

    # Week 5 (v3 repair-turn GRPO), README "Future work"
    rt3, rt2 = j("repair_turn_v3-checkpoint-400.json"), j("repair_turn_v2-checkpoint-900.json")
    checks.append(f"v3 rescued {rt3['rescued']} vs {rt2['rescued']}")
    checks.append(f"{100 * om['v3-400-repair']['official_ex']:.1f}% vs {100 * om['v2-900-repair']['official_ex']:.1f}% official")
    import csv
    log = [r for r in csv.DictReader(open(R / "train_log_v3.csv", encoding="utf-8")) if r["frac_reward_zero_std"]]
    zero = sum(float(r["frac_reward_zero_std"]) for r in log) / len(log)
    checks.append(f"About {round(100 * zero)}% of all training groups")

    # Week 6 grounding (voice table column + guard rail)
    vg = j("voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground.json")["conditions"]
    for c in ["text", "clean", "20db", "10db", "0db"]:
        checks.append(f"| {round(100 * vg[c]['ours_ex'])}% | **")  # grounding column, followed by the week-7 column
        if c != "text":
            checks.append(f"{round(100 * vrep[c]['sqlcoder_ex'])}%; {round(100 * vg[c]['sqlcoder_ex'])}%")
    dg = j(f"{B}_v2-checkpoint-900_repair_ground.json")
    checks.append(f"{pct(rep['execution_accuracy'])} → {pct(dg['execution_accuracy'])}, 5 / 4")

    # Week 7 example rows
    r3 = j(f"{B}_v2-checkpoint-900_repair_ground_rows3.json")
    checks.append(f"**{pct(r3['execution_accuracy'])}**")
    checks.append(f"**{pct(om['v2-900-repair-ground-rows3']['official_ex'])}**")
    pr = off["pairs_official"]["v2-900-repair-ground-rows3 vs v2-900-repair-ground"]
    checks.append(f"{pr['wins']} / {pr['losses']}, p={pr['p']:.3f}")
    v3r = j("voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3.json")["conditions"]
    for c in ["text", "clean", "20db", "10db", "0db"]:
        checks.append(f"**{round(100 * v3r[c]['ours_ex'])}%**")
    checks.append(f"{round(100 * v3r['clean']['ours_ex'])} vs {round(100 * v3r['clean']['sqlcoder_ex'])} clean")

    # Week 8 training with example rows (v4)
    r4 = j(f"{B}_v4-checkpoint-900_repair_ground_rows3.json")
    o4 = om["v4-900-repair-ground-rows3"]
    p4 = off["pairs_official"]["v4-900-repair-ground-rows3 vs v2-900-repair-ground-rows3"]
    checks.append(f"| {pct(r4['execution_accuracy'])} | {pct(o4['official_ex'])} | vs the row above: {p4['wins']} / {p4['losses']}, p={p4['p']:.2f}")
    checks.append(f"{pct(o4['official_ex'])} official vs {pct(om['v2-900-repair-ground-rows3']['official_ex'])}")
    checks.append(f"({pct(om['v4-900-repair-ground']['official_ex'])} vs {pct(om['v2-900-repair-ground']['official_ex'])})")

    # Week 10 schema pruning
    rp = j(f"{B}_v2-checkpoint-900_repair_ground_rows3_prune.json")
    op = om["v2-900-repair-ground-rows3-prune"]
    pp = off["pairs_official"]["v2-900-repair-ground-rows3-prune vs v2-900-repair-ground-rows3"]
    checks.append(f"| **{pct(rp['execution_accuracy'])}** | **{pct(op['official_ex'])}** | vs rows without pruning: {pp['wins']} / {pp['losses']}, p={pp['p']:.3f}")
    checks.append(f"Official EX {pct(om['v2-900-repair-ground-rows3']['official_ex'])} → {pct(op['official_ex'])} ({pp['wins']} / {pp['losses']}, p={pp['p']:.3f})")
    vp = j("voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3_prune.json")["conditions"]
    checks.append(" / ".join(str(round(100 * vp[c]["ours_ex"])) for c in ["text", "clean", "20db", "10db", "0db"]))
    rec = json.loads((R / "prune_recall_dev.json").read_text(encoding="utf-8"))
    checks.append(f"{100 * rec['recall']:.1f}% gold-table recall")
    lat = {k: j(f"demo_latency_w10_{k}.json")["text_to_sql_and_query"]["median_s"] for k in ("rows3_cold_now", "rows3_prune_cold")}
    checks.append(f"latency {lat['rows3_cold_now']:.1f} → {lat['rows3_prune_cold']:.1f} s")

    # Week 11 held-out test
    ot = j("official_ex_test.json")
    for m, d in [("A-base", "base"), ("B-v2-900", "v2-900"), ("C-v2-900-repair-ground", "v2-900-repair-ground"),
                 ("D-v2-900-repair-ground-rows3", "v2-900-repair-ground-rows3")]:
        checks.append(f"| {pct(ot['models'][m]['official_ex'])} | {pct(om[d]['official_ex'])} |")
    checks.append(f"| **{pct(ot['models']['E-v2-900-repair-ground-rows3-prune']['official_ex'])}** | {pct(op['official_ex'])} |")
    tp = ot["pairs_official"]
    t1, t4 = tp["B-v2-900 vs A-base"], tp["E-v2-900-repair-ground-rows3-prune vs A-base"]
    t2, t3 = tp["D-v2-900-repair-ground-rows3 vs C-v2-900-repair-ground"], tp["E-v2-900-repair-ground-rows3-prune vs D-v2-900-repair-ground-rows3"]
    checks.append(f"{t1['wins']} / {t1['losses']}")
    checks.append(f"{t4['wins']} / {t4['losses']}")
    checks.append(f"p={t2['p']:.3f}")
    checks.append(f"p={t3['p']:.2f}")

    # Week 12 BIRD
    bd = {t: j(f"bird_{t}.json")["execution_accuracy"] for t in ("A-base", "B-v2-900", "E-v2-900-full")}
    checks.append(f"| Qwen3-1.7B zero-shot | {pct(bd['A-base'])} |")
    checks.append(f"| GRPO (released model) | {pct(bd['B-v2-900'])} |")
    checks.append(f"| **{pct(bd['E-v2-900-full'])}** |")

    missing = [c for c in checks if c not in README]
    for c in checks:
        print(("OK   " if c not in missing else "MISS ") + c)
    if missing:
        raise SystemExit(f"{len(missing)} README numbers do not match results/")
    print(f"all {len(checks)} checks pass")


if __name__ == "__main__":
    main()
