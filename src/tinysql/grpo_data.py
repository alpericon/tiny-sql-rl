"""GRPO training records (PLAN-week2 W1).

Each record carries the chat prompt plus the columns the reward needs (``db_path``, ``gold``).
TRL passes extra dataset columns to reward functions as keyword arguments.
The prompt is built with the same ``build_messages`` used in evaluation, so train and eval match.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tinysql.data import Example, Exclusions, assert_no_leakage
from tinysql.prompt import build_messages, serialize_schema


def build_grpo_records(
    examples: Sequence[Example],
    database_dir: str | Path,
    exclusions: Exclusions,
    sample_rows: int = 0,
) -> list[dict]:
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
        records.append(
            {
                "prompt": build_messages(schemas[e.db_id], e.question),
                "db_path": str(db_path),
                "gold": e.query,
                "db_id": e.db_id,
            }
        )
    return records
