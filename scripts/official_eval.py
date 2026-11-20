"""Official Spider execution accuracy for the week-3 models (PLAN-week4 Y1).

Re-scores the stored full-dev predictions with Spider's official evaluation script
(third_party/test-suite-sql-eval, commit e97acc5) on the standard Spider databases, with values
predicted (no --plug_value) and DISTINCT handling at the official default (no --keep_distinct).

For each model it
  1. writes the official gold/prediction files (tinysql.official.eval_lines),
  2. runs the official CLI and keeps its printout (the authoritative aggregate),
  3. re-scores every row with the script's own eval_exec_match and hardness, asserting that the
     per-row total reproduces the CLI's overall accuracy, so rows can be paired (McNemar) and
     compared with our comparator.

Run-time cap: the official script's 60 s timeout wraps a blocking sqlite call, so it never fires,
and a few predictions (nested correlated subqueries on wta_1 / world_1) run for tens of minutes to
hours. Before scoring, every prediction is run once with our executor; any that does not finish in
CAP_S seconds is replaced by the never-executing placeholder, i.e. scored wrong. The official
scoring code itself is unchanged. The capped rows are listed in each model's json.

Runs in a separate CPU venv (sqlparse, nltk):
    ~/venvs/officialeval/bin/python scripts/official_eval.py
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

from tinysql.data import load_spider_split
from tinysql.executor import execute_readonly
from tinysql.official import FORMAT_FAILURE_SQL, eval_lines
from tinysql.stats import mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "test-suite-sql-eval"
# The 20 dev DBs copied onto the WSL filesystem: sqlite over /mnt/c makes the script ~10x slower.
# Same files as data/spider/database/<db_id>/ (copied, not modified).
DB_DIR = Path(os.environ.get("SPIDER_DEV_DB", Path.home() / "spider_dev_db"))
OUT = ROOT / "results" / "official"
MODELS = {
    "base": "",
    "main-1000": "_main-checkpoint-1000",
    "v2-900": "_v2-checkpoint-900",
    "v2-binary-800": "_v2-binary-checkpoint-800",
    "v2-900-repair": "_v2-checkpoint-900_repair",  # PLAN-week4 Y9
    "v3-400-repair": "_v3-checkpoint-400_repair",  # PLAN-week5 Z4
    "v2-900-repair-ground": "_v2-checkpoint-900_repair_ground",  # PLAN-week6
    "v2-900-repair-ground-rows3": "_v2-checkpoint-900_repair_ground_rows3",  # PLAN-week7
    "v4-900-repair-ground-rows3": "_v4-checkpoint-900_repair_ground_rows3",  # PLAN-week8 primary
    "v4-900-repair-ground": "_v4-checkpoint-900_repair_ground",  # PLAN-week8 secondary (no rows)
}
CAP_S = 120
PAIRS = [("v2-900", "base"),("v2-900", "main-1000"), ("v2-900", "v2-binary-800"), ("main-1000", "base"),
         ("v3-400-repair", "v2-900-repair"), ("v2-900-repair-ground", "v2-900-repair"),
         ("v2-900-repair-ground-rows3", "v2-900-repair-ground"),
         ("v4-900-repair-ground-rows3", "v2-900-repair-ground-rows3"), ("v4-900-repair-ground", "v2-900-repair-ground")]
# v2-900-repair vs v2-900 is deliberately not tested: repair can only rescue, never lose (Y9), so it
# is reported as a count of rescued rows.

sys.path.insert(0, str(OFFICIAL))
from evaluation import Evaluator  # noqa: E402
from exec_eval import eval_exec_match  # noqa: E402
from process_sql import Schema, get_schema, get_sql  # noqa: E402


def cli_accuracy(gold: Path, pred: Path, log: Path) -> float:
    out = subprocess.run(
        [sys.executable, "evaluation.py", "--gold", str(gold), "--pred", str(pred),
         "--db", str(DB_DIR), "--etype", "exec"],
        cwd=OFFICIAL, capture_output=True, text=True, check=True,
    ).stdout
    log.write_text(out, encoding="utf-8")
    row = next(l for l in out.splitlines() if l.strip().startswith("execution"))
    return float(row.split()[1:][4])  # columns: easy medium hard extra all [joint_all]


def per_row(gold_lines: list[str], pred_lines: list[str]) -> list[dict]:
    ev, rows, schemas = Evaluator(), [], {}
    for i, (g, p) in enumerate(zip(gold_lines, pred_lines)):
        g_str, db_id = g.split("\t")
        db = str(DB_DIR / db_id / f"{db_id}.sqlite")
        if db not in schemas:
            schemas[db] = Schema(get_schema(db))
        hardness = ev.eval_hardness(get_sql(schemas[db], g_str))
        p_str = p.replace("value", "1")  # same preprocessing as evaluation.py's evaluate()
        with redirect_stdout(io.StringIO()):
            ok = eval_exec_match(db=db, p_str=p_str, g_str=g_str, plug_value=False,
                                 keep_distinct=False, progress_bar_for_each_datapoint=False)
        rows.append({"idx": i, "db_id": db_id, "hardness": hardness, "official": bool(ok)})
    return rows


def score_model(name: str, dev) -> dict:
    recs = [json.loads(l) for l in open(ROOT / "results" / f"baseline_Qwen__Qwen3-1.7B{MODELS[name]}.records.jsonl", encoding="utf-8")]
    gold_lines, pred_lines = eval_lines(recs, dev)
    capped = []
    for i, (g, p) in enumerate(zip(gold_lines, pred_lines)):
        db_id = g.split("\t")[1]
        if p != FORMAT_FAILURE_SQL and execute_readonly(DB_DIR / db_id / f"{db_id}.sqlite", p, timeout_s=CAP_S,
                                                         max_rows=10**9).error == "timeout":
            capped.append(i)
            pred_lines[i] = FORMAT_FAILURE_SQL
    gold_f, pred_f = OUT / f"{name}.gold.txt", OUT / f"{name}.pred.txt"  # per model: runs go in parallel
    gold_f.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    pred_f.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")

    cli = cli_accuracy(gold_f, pred_f, OUT / f"{name}.official.log")
    rows = per_row(gold_lines, pred_lines)
    for r, rec in zip(rows, recs):
        r["ours"] = bool(rec["match"])
    n = len(rows)
    official = sum(r["official"] for r in rows) / n
    if abs(official - cli) > 0.0005:
        raise SystemExit(f"{name}: per-row official EX {official:.4f} != CLI {cli:.3f}")
    with open(OUT / f"{name}.rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_h = {}
    for h in ["easy", "medium", "hard", "extra"]:
        hs = [r for r in rows if r["hardness"] == h]
        by_h[h] = {"n": len(hs), "official_ex": sum(r["official"] for r in hs) / len(hs),
                   "ours_ex": sum(r["ours"] for r in hs) / len(hs)}
    result = {
        "n": n,
        "official_ex": official,
        "official_cli_all": cli,
        "ours_ex": sum(r["ours"] for r in rows) / n,
        "agree_both": sum(r["official"] and r["ours"] for r in rows),
        "ours_only": sum(r["ours"] and not r["official"] for r in rows),
        "official_only": sum(r["official"] and not r["ours"] for r in rows),
        "capped_over_s": CAP_S,
        "capped_rows": capped,
        "by_hardness": by_h,
    }
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(name, json.dumps({k: v for k, v in result.items() if k != "by_hardness"}), flush=True)
    return result


def summarize():
    results = {m: json.loads((OUT / f"{m}.json").read_text(encoding="utf-8")) for m in MODELS}
    matches = {m: [json.loads(l)["official"] for l in open(OUT / f"{m}.rows.jsonl", encoding="utf-8")] for m in MODELS}
    pairs = {}
    for a, b in PAIRS:
        w = sum(x and not y for x, y in zip(matches[a], matches[b]))
        l = sum(y and not x for x, y in zip(matches[a], matches[b]))
        pairs[f"{a} vs {b}"] = {"wins": w, "losses": l, "p": mcnemar_exact(w, l)}
    summary = {
        "script": "taoyds/test-suite-sql-eval@e97acc5 evaluation.py --etype exec (standard Spider DBs, values predicted, DISTINCT removed)",
        "models": results,
        "pairs_official": pairs,
    }
    (ROOT / "results" / "official_ex.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(pairs, indent=1))


def main():
    """Usage: official_eval.py [model ...]   (scores the named models, skipping finished ones;
    with no names, scores any unfinished model sequentially; always summarizes when all are done).
    Models can run in parallel processes, one name each."""
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_spider_split(ROOT / "data" / "spider" / "dev.json")
    for name in sys.argv[1:] or list(MODELS):
        if name not in MODELS:
            raise SystemExit(f"unknown model {name!r}; expected one of {list(MODELS)}")
        if not (OUT / f"{name}.json").exists():
            score_model(name, dev)
    if all((OUT / f"{m}.json").exists() for m in MODELS):
        summarize()


if __name__ == "__main__":
    main()
