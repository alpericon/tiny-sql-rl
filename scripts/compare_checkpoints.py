"""Paired comparison of main-run checkpoints vs the dev-200 baseline (exact McNemar, two-sided)."""
import json, math, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
base = load(R / "dev200_Qwen__Qwen3-1.7B.records.jsonl")
def exact_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
print(f"{'ckpt':>6} {'EX':>6} {'delta':>6} {'fmt':>5} {'hit':>5} {'win':>4} {'loss':>4} {'p':>6}")
print(f"{'base':>6} {sum(r['match'] for r in base)/len(base):6.3f}")
for n in range(100, 1001, 100):
    p = R / f"dev200_Qwen__Qwen3-1.7B_main-checkpoint-{n}.records.jsonl"
    if not p.exists(): continue
    recs = load(p); s = json.load(open(str(p).replace(".records.jsonl", ".json")))
    assert len(recs) == len(base)
    win = sum(r["match"] and not b["match"] for r, b in zip(recs, base))
    loss = sum(b["match"] and not r["match"] for r, b in zip(recs, base))
    ex = s["execution_accuracy"]
    print(f"{n:>6} {ex:6.3f} {100*(ex-0.55):+6.1f} {s['format_failure_rate']:5.3f} {s['hit_max_new_tokens_rate']:5.3f} {win:>4} {loss:>4} {exact_p(win,loss):6.3f}")
