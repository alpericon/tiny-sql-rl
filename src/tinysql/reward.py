"""Execution-based reward for GRPO (SPEC §5.2)."""

from __future__ import annotations

import re
from pathlib import Path

from tinysql.compare import gold_is_ordered, results_match
from tinysql.executor import execute_readonly

_FENCED_SQL = re.compile(r"```sql\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

FULL = 1.0
PARTIAL = 0.1


def extract_sql(completion: str) -> str | None:
    """Return the SQL inside the last ```sql fenced block, or None if there is none."""
    blocks = _FENCED_SQL.findall(completion)
    if not blocks:
        return None
    sql = blocks[-1].strip()
    return sql or None


def sql_reward(
    completion: str,
    gold_sql: str,
    db_path: str | Path,
    binary: bool = False,
    timeout_s: float = 5.0,
) -> float:
    gold = execute_readonly(db_path, gold_sql, timeout_s=timeout_s)
    if not gold.ok:
        raise ValueError(f"gold SQL failed on {db_path}: {gold.error}")

    sql = extract_sql(completion)
    if sql is None:
        return 0.0
    pred = execute_readonly(db_path, sql, timeout_s=timeout_s)
    if not pred.ok:
        return 0.0
    if results_match(pred.rows, gold.rows, ordered=gold_is_ordered(gold_sql)):
        return FULL
    return 0.0 if binary else PARTIAL
