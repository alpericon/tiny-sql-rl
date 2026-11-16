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
