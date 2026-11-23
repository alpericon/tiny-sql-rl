"""BIRD dev: loader, evidence prompt and BIRD's execution accuracy (PLAN-week12).

BIRD's EX (bird/llm/src/evaluation.py in AlibabaResearch/DAMO-ConvAI) runs the predicted and the gold
SQL and scores 1 iff ``set(predicted_rows) == set(gold_rows)``:
* row order and duplicate rows are ignored;
* rows are compared as tuples, so column order and column count matter;
* Python equality applies (1 == 1.0, None == None);
* an exception or a timeout (30 s) scores 0.

Differences from the official script, kept on purpose: queries run through our read-only executor
(``execute_readonly``), and each query gets its own 30 s, where the official script gives 30 s to
both together. The executor also rejects a single value over 1 MB, a result over 64 MB and any
statement other than a read (its authorizer); none of the gold queries is affected.
"""

from __future__ import annotations

import json
from pathlib import Path

from tinysql.data import Example
from tinysql.executor import execute_readonly

TIMEOUT_S = 30.0
MAX_ROWS = 10_000_000


def bird_question(question: str, evidence: str | None) -> str:
    """The question text the model sees: the question, then the expert evidence on its own line."""
    return f"{question}\nEvidence: {evidence}" if evidence else question


def load_bird(path: str | Path) -> tuple[list[Example], list[dict]]:
    """Examples in file order, with the evidence folded into the question, plus per-question metadata."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = [Example(r["db_id"], bird_question(r["question"], r.get("evidence")), r["SQL"]) for r in rows]
    meta = [{"question_id": r["question_id"], "difficulty": r["difficulty"]} for r in rows]
    return examples, meta


def bird_pairs(base: list[dict], cand: list[dict], exclude_db: str = "formula_1") -> dict:
    """Paired comparison (cand over base) on all questions and without one database (PLAN-week12/13).
    Refuses records whose question_id order differs."""
    from tinysql.stats import paired_summary

    if [r["question_id"] for r in base] != [r["question_id"] for r in cand]:
        raise ValueError("records are not in the same question order")
    keep = [i for i, r in enumerate(base) if r["db_id"] != exclude_db]
    return {"all": paired_summary(base, cand),
            f"without_{exclude_db}": paired_summary([base[i] for i in keep], [cand[i] for i in keep])}


def bird_ex(db_path: str | Path, gold_sql: str, pred_sql: str, timeout_s: float = TIMEOUT_S) -> dict:
    """BIRD EX of one prediction: {exec_ok, match, error}, the same shape as ``evaluate.score_sql``."""
    gold = execute_readonly(db_path, gold_sql, timeout_s=timeout_s, max_rows=MAX_ROWS)
    pred = execute_readonly(db_path, pred_sql, timeout_s=timeout_s, max_rows=MAX_ROWS)
    match = (pred.ok and gold.ok and not pred.truncated and not gold.truncated
             and set(map(tuple, pred.rows)) == set(map(tuple, gold.rows)))
    error = pred.error if not pred.ok else (None if gold.ok else f"gold: {gold.error}")
    return {"exec_ok": pred.ok, "match": bool(match), "error": error}
