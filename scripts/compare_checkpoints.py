"""Paired comparison of a run's checkpoints vs a dev-200 baseline (exact McNemar, two-sided).

    python scripts/compare_checkpoints.py [run-name]   # default: main, vs the base model
    # PLAN-week8: checkpoints evaluated with --sample-rows 3, vs v2-900 with rows
    python scripts/compare_checkpoints.py v4 --suffix _rows3 \
        --base dev200_Qwen__Qwen3-1.7B_v2-checkpoint-900_rows3.records.jsonl
    # PLAN-week14: a seed run; also writes results/selection_<run>.json (the release rule)
    python scripts/compare_checkpoints.py v2-s1 --write-selection
"""
import argparse, json, math
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
ap = argparse.ArgumentParser()
ap.add_argument("run", nargs="?", default="main")
ap.add_argument("--suffix", default="", help="output-name suffix of the checkpoint evals, e.g. _rows3")
ap.add_argument("--write-selection", action="store_true",
                help="needs all 10 checkpoints; writes the checkpoint the release rule picks (PLAN-week3/14)")
ap.add_argument("--base", default="dev200_Qwen__Qwen3-1.7B.records.jsonl", help="baseline records in results/")
args = ap.parse_args()
base = load(R / args.base)
base_ex = sum(r["match"] for r in base) / len(base)
def exact_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
print(f"{'ckpt':>6} {'EX':>6} {'delta':>6} {'fmt':>5} {'hit':>5} {'win':>4} {'loss':>4} {'p':>6}")
ex_by_step = {}
print(f"{'base':>6} {base_ex:6.3f}   ({args.base})")
for n in range(100, 1001, 100):
    p = R / f"dev200_Qwen__Qwen3-1.7B_{args.run}-checkpoint-{n}{args.suffix}.records.jsonl"
    if not p.exists(): continue
    recs = load(p); s = json.load(open(str(p).replace(".records.jsonl", ".json")))
    assert len(recs) == len(base)
    assert [r["question"] for r in recs] == [b["question"] for b in base], "not the same questions in the same order"
    win = sum(r["match"] and not b["match"] for r, b in zip(recs, base))
    loss = sum(b["match"] and not r["match"] for r, b in zip(recs, base))
    ex = s["execution_accuracy"]
    ex_by_step[n] = ex
    print(f"{n:>6} {ex:6.3f} {100*(ex-base_ex):+6.1f} {s['format_failure_rate']:5.3f} {s['hit_max_new_tokens_rate']:5.3f} {win:>4} {loss:>4} {exact_p(win,loss):6.3f}")
if args.write_selection:
    from tinysql.seeds import select_checkpoint
    assert sorted(ex_by_step) == list(range(100, 1001, 100)), f"missing checkpoint evals: {sorted(ex_by_step)}"
    sel = select_checkpoint(ex_by_step)
    (R / f"selection_{args.run}.json").write_text(json.dumps(
        {"run": args.run, "rule": "best dev-200 EX, ties to the later step (PLAN-week3)",
         "dev200_ex": ex_by_step, "selected": sel}, indent=1), encoding="utf-8")
    print(f"selected: checkpoint-{sel}")
