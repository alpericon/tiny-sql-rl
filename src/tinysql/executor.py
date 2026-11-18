"""Sandboxed, read-only execution of model-written SQL against a SQLite file.

Layers of defence (SPEC §5.2), any one of which blocks a write:
1. The database is opened with ``mode=ro`` (and never created if missing).
2. An authorizer allows only reads: SELECT, column reads, functions, recursive CTEs.
   ATTACH, PRAGMA, DDL and DML are denied.
3. Only a single statement is accepted (Python's ``execute`` also refuses more).

Resource limits, so a hostile query cannot stall or kill the trainer:
- wall-clock timeout via a progress handler,
- ``SQLITE_LIMIT_LENGTH`` caps any single string/blob (blocks ``zeroblob(1e9)``),
- a row cap and a total result-size cap while fetching,
- every exception is turned into ``ok=False``; this function never raises on model input.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}

MAX_VALUE_BYTES = 1_000_000  # largest single string/blob SQLite may build
MAX_RESULT_BYTES = 64_000_000  # approximate total size of the fetched result


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: list[tuple] | None
    error: str | None = None
    truncated: bool = False
    columns: tuple[str, ...] | None = None  # result column names (for display)


def _authorizer(action, *_args):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _strip_trailing_semicolons(sql: str) -> str:
    sql = sql.strip()
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql


def _value_size(v) -> int:
    if isinstance(v, (str, bytes)):
        return len(v)
    return 8


def execute_readonly(
    db_path: str | Path,
    sql: str,
    timeout_s: float = 5.0,
    max_rows: int = 10_000,
) -> ExecResult:
    try:
        return _execute(db_path, sql, timeout_s, max_rows)
    except Exception as exc:  # noqa: BLE001 - model input must never crash the caller
        return ExecResult(False, None, f"internal: {type(exc).__name__}")


def _execute(db_path, sql: str, timeout_s: float, max_rows: int) -> ExecResult:
    if "\x00" in sql:
        return ExecResult(False, None, "rejected: NUL character")
    try:
        sql.encode("utf-8")
    except UnicodeEncodeError:
        return ExecResult(False, None, "rejected: invalid unicode")

    sql = _strip_trailing_semicolons(sql)
    if not sql:
        return ExecResult(False, None, "rejected: empty")
    if not sqlite3.complete_statement(sql + ";"):
        return ExecResult(False, None, "sql_error: incomplete statement")

    path = Path(db_path)
    if not path.is_file():
        return ExecResult(False, None, "rejected: database not found")

    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    except sqlite3.Error as exc:
        return ExecResult(False, None, f"sql_error: {exc}")

    deadline = time.monotonic() + timeout_s
    timed_out = False

    def _progress():
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # non-zero aborts the running statement
        return 0

    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(_progress, 1000)
        cur = conn.execute(sql)

        rows: list[tuple] = []
        size = 0
        truncated = False
        for row in cur:
            if len(rows) == max_rows:
                truncated = True
                break
            size += sum(_value_size(v) for v in row)
            if size > MAX_RESULT_BYTES:
                return ExecResult(False, None, "rejected: result too large")
            rows.append(tuple(row))
        columns = tuple(d[0] for d in cur.description) if cur.description else None
        return ExecResult(True, rows, None, truncated, columns)
    except sqlite3.ProgrammingError as exc:  # e.g. more than one statement
        return ExecResult(False, None, f"rejected: {exc}")
    except sqlite3.DatabaseError as exc:
        if timed_out:
            return ExecResult(False, None, "timeout")
        msg = str(exc)
        if "not authorized" in msg or "readonly" in msg:
            return ExecResult(False, None, f"rejected: {msg}")
        return ExecResult(False, None, f"sql_error: {msg}")
    finally:
        conn.close()
