"""GRPO training records (PLAN-week2 W1).

Each record carries the chat prompt plus the columns the reward needs (``db_path``, ``gold``).
TRL passes extra dataset columns to reward functions as keyword arguments.
The prompt is built with the same ``build_messages`` used in evaluation, so train and eval match.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tinysql.data import Example, Exclusions, assert_no_leakage
from tinysql.executor import execute_readonly
from tinysql.prompt import build_messages, serialize_schema


def build_grpo_records(
    examples: Sequence[Example],
    database_dir: str | Path,
    exclusions: Exclusions,
    sample_rows: int = 0,
    drop_empty_gold: bool = False,
) -> list[dict]:
    """drop_empty_gold: skip examples whose gold query returns no rows. Against an empty gold
    answer any query that returns nothing scores 1.0, so those examples reward wrong SQL (W7)."""
    # Defence in depth: the input should already be filtered, but never train on SpokenSpider.
    assert_no_leakage(list(examples), exclusions)

    database_dir = Path(database_dir)
    schemas: dict[str, str] = {}
    records = []
    for e in examples:
        db_path = database_dir / e.db_id / f"{e.db_id}.sqlite"
        if e.db_id not in schemas:
            if not db_path.is_file():
                raise FileNotFoundError(db_path)
            schemas[e.db_id] = serialize_schema(db_path, sample_rows=sample_rows)
        if drop_empty_gold:
            gold = execute_readonly(db_path, e.query)
            if gold.ok and not gold.rows:  # a failing gold is kept so the reward raises loudly
                continue
        records.append(
            {
                "prompt": build_messages(schemas[e.db_id], e.question),
                "db_path": str(db_path),
                "gold": e.query,
                "db_id": e.db_id,
                "question": e.question,
            }
        )
    return records


def filter_by_pass_rate(records: list[dict], rates: list[dict]) -> list[dict]:
    """Keep records whose base-model pass rate is strictly between 0 and n (W3/X2).

    A group of n samples that all fail or all succeed has zero advantage in GRPO, so it gives no
    learning signal. ``rates`` items are {db_id, question, gold, n_correct, n}, keyed on all three
    because Spider train repeats some questions; a record without a rate
    raises KeyError instead of being silently dropped.
    """
    by_key = {(r["db_id"], r["question"], r["gold"]): r for r in rates}
    kept = []
    for rec in records:
        rate = by_key[(rec["db_id"], rec["question"], rec["gold"])]
        if 0 < rate["n_correct"] < rate["n"]:
            kept.append(rec)
    return kept


def build_repair_records(samples: Sequence[dict], max_per_question: int = 2, timeout_s: float = 5.0) -> list[dict]:
    """Repair-turn GRPO records from sampled policy answers (PLAN-week5 Z1).

    ``samples`` carry the original record fields (db_id, question, gold, db_path, prompt) plus the
    sampled ``completion``. Only answers with no ```sql block, or whose SQL does not execute, become
    records; the prompt is exactly the inference-time retry (``tinysql.repair.repair_messages``).
    Duplicate (question, failed answer) pairs are dropped and each question keeps at most
    ``max_per_question`` records, in sample order.
    """
    from tinysql.repair import NO_BLOCK, repair_messages
    from tinysql.reward import extract_sql

    records, seen, per_question = [], set(), {}
    for s in samples:
        qkey = (s["db_id"], s["question"], s["gold"])
        if per_question.get(qkey, 0) >= max_per_question:
            continue
        sql = extract_sql(s["completion"])
        if sql is None:
            error = None
        else:
            res = execute_readonly(s["db_path"], sql, timeout_s=timeout_s)
            if res.ok:
                continue
            error = res.error
        key = (*qkey, (sql if sql is not None else s["completion"]).strip())
        if key in seen:
            continue
        seen.add(key)
        per_question[qkey] = per_question.get(qkey, 0) + 1
        records.append({
            "prompt": repair_messages(s["prompt"], s["completion"], error),
            "db_path": s["db_path"], "gold": s["gold"], "db_id": s["db_id"], "question": s["question"],
            "kind": "repair", "first_error": NO_BLOCK if error is None else error,
        })
    return records
