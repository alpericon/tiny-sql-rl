"""PLAN-week13 K3 (review MEDIUM): 256-token hits for the week 12 BIRD runs, which did not record them,
estimated by re-tokenising stored answers (validated on Coder CA, which did); formula_1 EX per model.

    ~/venvs/tinysql/bin/python scripts/week13_hits.py
"""
import json
from transformers import AutoTokenizer
R = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
toks = {"q3": AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"), "coder": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")}
for t, tk in [("A-base", "q3"), ("B-v2-900", "q3"), ("CA-coder", "coder")]:
    rs = R(f"results/bird_{t}.records.jsonl")
    est = [len(toks[tk](r["completion"], add_special_tokens=False)["input_ids"]) >= 256 for r in rs]
    line = f"{t}: re-tokenised first-attempt hits {sum(est)} / {len(rs)}"
    if "hit_max_new_tokens" in rs[0]:
        rec = [r["hit_max_new_tokens"] for r in rs]
        line += f"; recorded {sum(rec)}; agree on {sum(a == b for a, b in zip(est, rec))}"
    print(line)
for t in ["A-base", "B-v2-900", "E-v2-900-full", "CA-coder", "CE-coder-full"]:
    rs = [r for r in R(f"results/bird_{t}.records.jsonl") if r["db_id"] == "formula_1"]
    print(t, "formula_1", len(rs), round(sum(r["match"] for r in rs) / len(rs), 3), "fmt fail", sum(not r["format_ok"] for r in rs))
