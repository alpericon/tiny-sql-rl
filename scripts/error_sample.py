"""Build the wrong-result error-analysis sample (PLAN-week9 E1) -> results/error_analysis_sample.jsonl.

Population: released configuration (v2-900, rows, repair, grounding), answer runs but is wrong on our
metric and on the official one. 100 questions, stratified by official hardness, seed 3407. Each row
carries what labelling needs: question, schema with 3 example rows (the model's prompt), gold and
predicted SQL, and a preview of both results. 20 rows are marked for the user's spot check (E3).

    python scripts/error_sample.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from tinysql.error_sample import stratified_sample, wrong_on_both
from tinysql.executor import execute_readonly
from tinysql.prompt import serialize_schema

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
DB = ROOT / "data" / "spider" / "database"
SEED, N, N_CHECK = 3407, 100, 20


def preview(db_path, sql):
    res = execute_readonly(db_path, sql)
    if not res.ok:
        return {"ok": False, "error": res.error}
    return {"ok": True, "n_rows": len(res.rows), "rows": [list(map(str, r)) for r in res.rows[:5]]}


def main():
    recs = [json.loads(l) for l in open(R / "baseline_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3.records.jsonl", encoding="utf-8")]
    off = [json.loads(l) for l in open(R / "official" / "v2-900-repair-ground-rows3.rows.jsonl", encoding="utf-8")]
    pop = wrong_on_both(recs, off)
    strata = {i: off[i]["hardness"] for i in pop}
    chosen = stratified_sample(strata, N, SEED)
    check = set(random.Random(SEED).sample(range(N), N_CHECK))
    with (R / "error_analysis_sample.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for k, i in enumerate(chosen):
            r = recs[i]
            db_path = DB / r["db_id"] / f"{r['db_id']}.sqlite"
            f.write(json.dumps({
                "k": k, "idx": i, "db_id": r["db_id"], "hardness": off[i]["hardness"], "spot_check": k in check,
                "question": r["question"], "gold": r["gold"], "pred": r["pred_sql"],
                "repaired": r["repaired"], "grounding": r["grounding"],
                "gold_result": preview(db_path, r["gold"]), "pred_result": preview(db_path, r["pred_sql"]),
                "schema": serialize_schema(db_path, sample_rows=3),
            }, ensure_ascii=False) + "\n")
    print(f"population {len(pop)}: {dict(Counter(strata.values()))}")
    print(f"sample {len(chosen)}: {dict(Counter(strata[i] for i in chosen))}; spot check {len(check)}")


if __name__ == "__main__":
    main()
