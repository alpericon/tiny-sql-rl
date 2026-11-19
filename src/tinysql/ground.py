"""Ground SQL string literals in the database's actual values (PLAN-week6).

ASR transcripts lose capitalisation and quotes, so the model writes `theme = 'aliens'` while the
database holds 'Aliens'; SQLite compares strings case-sensitively. Before execution, every
single-quoted literal that does not occur exactly in the database, but whose normalised form
(case-folded, whitespace collapsed) matches exactly one distinct database value, is replaced by that
value. Nothing else is touched: exact matches, ambiguous or unmatched literals, LIKE-style literals
containing % or _, numbers, and double-quoted identifiers.

Known limitation: the scanner does not understand SQL comments. An apostrophe inside ``--`` or
``/* */`` opens a phantom literal, so later literals may go ungrounded. This fails safe (nothing is
wrongly replaced), and the model rarely writes comments.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def normalise(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass
class ValueIndex:
    exact: set[str] = field(default_factory=set)
    by_norm: dict[str, set[str]] = field(default_factory=dict)
    skipped_tables: list[str] = field(default_factory=list)


def build_value_index(db_path: str | Path, max_rows: int = 50_000) -> ValueIndex:
    """Distinct string values of every column in every table with at most ``max_rows`` rows."""
    index = ValueIndex()
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for t in tables:
            q = '"' + t.replace('"', '""') + '"'
            if conn.execute(f"SELECT count(*) FROM {q}").fetchone()[0] > max_rows:
                index.skipped_tables.append(t)
                continue
            for row in conn.execute(f"SELECT * FROM {q}"):
                for v in row:
                    if isinstance(v, str) and v:
                        index.exact.add(v)
                        index.by_norm.setdefault(normalise(v), set()).add(v)
    finally:
        conn.close()
    return index


@lru_cache(maxsize=64)
def cached_index(db_path: str, max_rows: int = 50_000) -> ValueIndex:
    return build_value_index(db_path, max_rows)


def _grounded(literal: str, index: ValueIndex) -> str | None:
    if "%" in literal or "_" in literal or literal in index.exact:
        return None
    matches = index.by_norm.get(normalise(literal), set())
    return next(iter(matches)) if len(matches) == 1 else None


def ground_sql(sql: str, index: ValueIndex) -> tuple[str, list[tuple[str, str]]]:
    """Return the grounded SQL and the (old, new) literal replacements, in order."""
    out, reps, i, n = [], [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == '"':  # identifier: copy verbatim, honouring "" escapes
            j = i + 1
            while j < n and not (sql[j] == '"' and (j + 1 >= n or sql[j + 1] != '"')):
                j += 2 if sql[j] == '"' else 1
            out.append(sql[i : j + 1])
            i = j + 1
        elif ch == "'":  # string literal with '' escapes
            j, buf = i + 1, []
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(sql[j])
                j += 1
            literal = "".join(buf)
            new = _grounded(literal, index) if j < n else None
            if new is None:
                out.append(sql[i : j + 1])
            else:
                reps.append((literal, new))
                out.append("'" + new.replace("'", "''") + "'")
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out), reps
