"""Demo pipeline core: question text -> SQL -> read-only result -> one-sentence answer (PLAN-week4 Y5).

Model-agnostic: ``generate`` is the same batch callable used by evaluation (HF, llama.cpp or a fake),
and the prompt is the training/evaluation prompt. Every failure comes back as ``error`` on the
result, never as an exception, so the UI can always show something.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tinysql.executor import execute_readonly
from tinysql.ground import cached_index, ground_sql
from tinysql.prompt import build_messages, serialize_schema
from tinysql.prune import select_tables
from tinysql.repair import NO_BLOCK, repair_messages
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
    repaired: bool = False  # the first answer failed and this is the one retry (tinysql.repair)
    first_sql: str | None = None
    first_error: str | None = None
    grounding: list = field(default_factory=list)  # (old, new) literal replacements (tinysql.ground)


@lru_cache(maxsize=32)
def _schema(db_path: str, sample_rows: int = 0, tables: tuple[str, ...] | None = None) -> str:
    return serialize_schema(db_path, sample_rows=sample_rows, tables=tables)


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


def _attempt(messages, db_path, generate, max_rows, timeout_s, ground, grounding):
    completion = generate([messages])[0]
    sql = extract_sql(completion)
    if sql is None:
        return completion, None, None
    if ground:
        sql, reps = ground_sql(sql, cached_index(str(db_path)))
        grounding.extend(reps)
    return completion, sql, execute_readonly(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)


def answer_question(question: str, db_path: str | Path, generate: Callable, max_rows: int = 200,
                    timeout_s: float = 10.0, repair: bool = True, ground: bool = True,
                    sample_rows: int = 0, prune: bool = False) -> DemoResult:
    """With ``repair`` (the default), an answer with no SQL block or SQL that does not run gets one
    retry with the error message, under the same rule as the evaluation (PLAN-week4 Y9).
    With ``prune``, the first attempt shows only the tables the pruner keeps; the retry uses the full
    schema (PLAN-week10)."""
    question = question.strip()
    if not question:
        return DemoResult(question, None, None, None, None, "empty question")
    full = build_messages(_schema(str(db_path), sample_rows), question)
    tables = select_tables(question, db_path) if prune else None
    messages = full if tables is None else build_messages(_schema(str(db_path), sample_rows, tuple(tables)), question)
    grounding: list = []
    completion, sql, res = _attempt(messages, db_path, generate, max_rows, timeout_s, ground, grounding)
    extra = {}
    if repair and (sql is None or not res.ok):
        first_error = NO_BLOCK if sql is None else res.error
        extra = {"repaired": True, "first_sql": sql, "first_error": first_error}
        retry = repair_messages(full, completion, None if sql is None else res.error)
        grounding.clear()  # report only the final answer's replacements
        completion, sql, res = _attempt(retry, db_path, generate, max_rows, timeout_s, ground, grounding)
    extra["grounding"] = grounding
    if sql is None:
        return DemoResult(question, None, None, None, None, "the model returned no SQL block", completion, **extra)
    if not res.ok:
        return DemoResult(question, sql, None, None, None, res.error, completion, **extra)
    return DemoResult(question, sql, res.columns, res.rows, _answer(res.rows, res.truncated), None, completion, **extra)
