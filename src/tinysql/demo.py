"""Demo pipeline core: question text -> SQL -> read-only result -> one-sentence answer (PLAN-week4 Y5).

Model-agnostic: ``generate`` is the same batch callable used by evaluation (HF, llama.cpp or a fake),
and the prompt is the training/evaluation prompt. Every failure comes back as ``error`` on the
result, never as an exception, so the UI can always show something.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tinysql.executor import execute_readonly
from tinysql.prompt import build_messages, serialize_schema
from tinysql.reward import extract_sql


@dataclass(frozen=True)
class DemoResult:
    question: str
    sql: str | None
    columns: tuple[str, ...] | None
    rows: list[tuple] | None
    answer: str | None
    error: str | None
    completion: str | None = None


@lru_cache(maxsize=32)
def _schema(db_path: str) -> str:
    return serialize_schema(db_path)


def _fmt(v) -> str:
    if v is None:
        return "empty (NULL)"
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def _answer(rows: list[tuple], truncated: bool) -> str:
    if not rows:
        return "No rows match."
    if truncated:
        return f"Showing the first {len(rows)} rows."
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"The answer is {_fmt(rows[0][0])}."
    return f"Found {len(rows)} row{'s' if len(rows) != 1 else ''}."


def answer_question(question: str, db_path: str | Path, generate: Callable, max_rows: int = 200,
                    timeout_s: float = 10.0) -> DemoResult:
    question = question.strip()
    if not question:
        return DemoResult(question, None, None, None, None, "empty question")
    completion = generate([build_messages(_schema(str(db_path)), question)])[0]
    sql = extract_sql(completion)
    if sql is None:
        return DemoResult(question, None, None, None, None, "the model returned no SQL block", completion)
    res = execute_readonly(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)
    if not res.ok:
        return DemoResult(question, sql, None, None, None, res.error, completion)
    return DemoResult(question, sql, res.columns, res.rows, _answer(res.rows, res.truncated), None, completion)
