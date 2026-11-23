"""Inputs for Spider's official evaluation script (third_party/test-suite-sql-eval), PLAN-week4 Y1.

The script reads a gold file (`SQL\\tdb_id` per line) and a prediction file (one SQL per line),
paired by line number. ``eval_lines`` builds both from our evaluation records, refusing any
misalignment with the dev split, and never drops a row: a format failure becomes a placeholder
query that cannot execute, so it scores 0 exactly as in our own EX.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from tinysql.data import Example

# Fails to execute on every Spider DB (no such column), so it can never match a gold result.
FORMAT_FAILURE_SQL = "SELECT tinysql_format_failure"

_BREAKS = re.compile(r"\s*[\r\n\t]+\s*")
QWEN3 = "Qwen/Qwen3-1.7B"


def record_stem(entry: str | tuple[str, str]) -> str:
    """The record-file stem after the split prefix. A bare suffix means Qwen3-1.7B (every model up to
    week 12); a (model, suffix) pair names another model, as eval_baseline.py tags it (PLAN-week13)."""
    model, suffix = (QWEN3, entry) if isinstance(entry, str) else entry
    return model.replace("/", "__") + suffix


def one_line(sql: str) -> str:
    """Join a multi-line query onto one line. Spaces inside a line, including inside string
    literals, are kept; only line breaks and tabs (which would split the file's columns) go."""
    return _BREAKS.sub(" ", sql).strip()


def eval_lines(records: Sequence[dict], dev: Sequence[Example]) -> tuple[list[str], list[str]]:
    if len(records) != len(dev):
        raise ValueError(f"{len(records)} records for {len(dev)} dev examples")
    gold, pred = [], []
    for i, (r, e) in enumerate(zip(records, dev)):
        if (r["db_id"], r["question"], r["gold"]) != (e.db_id, e.question, e.query):
            raise ValueError(f"record {i} does not match dev example {i} ({e.db_id}: {e.question!r})")
        gold.append(f"{one_line(e.query)}\t{e.db_id}")
        p = one_line(r["pred_sql"] or "")
        pred.append(p or FORMAT_FAILURE_SQL)
    return gold, pred
