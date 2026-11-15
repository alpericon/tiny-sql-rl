"""Schema serialization and prompt construction (SPEC §5.1).

Prompts are never truncated: a truncated prompt hides the schema and silently drives the
reward to zero. ``check_fits`` raises instead, so the problem shows up in the logs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert SQLite assistant. Given a database schema and a question, "
    "write one SQLite query that answers the question. "
    "Return only the query inside a ```sql fenced code block."
)


class PromptTooLong(ValueError):
    pass


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _one_line(v) -> str:
    # A newline in a sample value would escape the "--" comment and look like SQL.
    return str(v).replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def serialize_schema(db_path: str | Path, sample_rows: int = 0) -> str:
    conn = _connect_ro(db_path)
    try:
        tables = conn.execute(
            "SELECT name, sql, type FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' AND sql IS NOT NULL "
            "ORDER BY type, name"
        ).fetchall()
        parts = []
        for name, ddl, kind in tables:
            block = ddl.strip().rstrip(";") + ";"
            if sample_rows > 0 and kind == "table":
                rows = conn.execute(f"SELECT * FROM {_quote_ident(name)} LIMIT ?", (sample_rows,)).fetchall()
                if rows:
                    shown = "\n".join("--   " + " | ".join(_one_line(v) for v in r) for r in rows)
                    block += f"\n-- Example rows from {name}:\n{shown}"
            parts.append(block)
        return "\n\n".join(parts)
    finally:
        conn.close()


def build_messages(schema: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion: {question}"},
    ]


def check_fits(messages: list[dict[str, str]], max_tokens: int, count_tokens: Callable[[str], int]) -> int:
    """Return the prompt length in tokens; raise PromptTooLong if it exceeds ``max_tokens``.

    ``count_tokens`` receives the joined message contents, e.g. ``lambda s: len(tok(s).input_ids)``.
    Chat-template tokens are not included, so pass a ``max_tokens`` with some headroom.
    """
    n = count_tokens("\n".join(m["content"] for m in messages))
    if n > max_tokens:
        raise PromptTooLong(f"prompt is {n} tokens, limit is {max_tokens}")
    return n
