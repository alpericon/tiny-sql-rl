"""Lexical table pruning for the prompt (PLAN-week10).

A table is kept when the question mentions it:
* by name: every content word of the table name appears in the question (``singer_in_concert``
  needs both "singer" and "concert");
* by column: a word of one of its column names appears in the question, counting only column words
  that occur in at most ``max_col_df`` tables of the database (so "id" or "name" do not keep everything);
* by value: one of its stored string values appears in the question as a whole phrase.
Tables on the shortest foreign-key path between kept tables are added as bridges. When fewer than
``min_tables`` tables match, the result is ``None``: use the full schema.

Defaults frozen in P2 by the rule fixed before the grid: among settings with train recall >= 98%,
the one showing the fewest tables (results/prune_recall_train.json: recall 98.3%, 72.5% of tables
shown, full-schema fallback 13%).

Words are lower-cased, split on non-letters and camelCase, and singularised crudely (so "singers"
matches ``singer``). The defaults are fixed on train questions (P2), never on dev.
"""

from __future__ import annotations

import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

STOPWORDS = {"a", "an", "and", "by", "for", "has", "in", "of", "on", "the", "to", "with"}
MAX_ROWS = 50_000
_TABLE = re.compile(r"\b(?:FROM|JOIN)\s+[`\"\[]?(\w+)", re.IGNORECASE)


def gold_tables(sql: str) -> set[str]:
    return {t.lower() for t in _TABLE.findall(sql or "")}


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def words(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [_singular(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def _phrase(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


@dataclass
class Profile:
    names: dict[str, str] = field(default_factory=dict)  # lower -> original
    name_words: dict[str, set[str]] = field(default_factory=dict)
    col_words: dict[str, set[str]] = field(default_factory=dict)
    values: dict[str, set[str]] = field(default_factory=dict)  # phrases like " glasgow "
    fk: dict[str, set[str]] = field(default_factory=dict)


@lru_cache(maxsize=64)
def profile(db_path: str) -> Profile:
    p = Profile()
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"):
            k = t.lower()
            q = '"' + t.replace('"', '""') + '"'
            p.names[k] = t
            p.name_words[k] = {w for w in words(t) if w not in STOPWORDS}
            p.col_words[k] = {w for r in conn.execute(f"PRAGMA table_info({q})") for w in words(r[1])}
            p.fk.setdefault(k, set())
            for r in conn.execute(f"PRAGMA foreign_key_list({q})"):
                other = r[2].lower()
                if other != k:
                    p.fk[k].add(other)
                    p.fk.setdefault(other, set()).add(k)
            vals = set()
            if conn.execute(f"SELECT count(*) FROM {q}").fetchone()[0] <= MAX_ROWS:
                for row in conn.execute(f"SELECT * FROM {q}"):
                    for v in row:
                        if isinstance(v, str):
                            ph = _phrase(v)
                            if len(ph.strip()) >= 3 and not ph.strip().isdigit():
                                vals.add(ph)
            p.values[k] = vals
    finally:
        conn.close()
    p.fk = {k: v & set(p.names) for k, v in p.fk.items() if k in p.names}
    return p


def _path(fk: dict[str, set[str]], a: str, b: str) -> list[str]:
    prev, todo = {a: None}, deque([a])
    while todo:
        x = todo.popleft()
        if x == b:
            out = []
            while x is not None:
                out.append(x)
                x = prev[x]
            return out
        for y in sorted(fk.get(x, ())):
            if y not in prev:
                prev[y] = x
                todo.append(y)
    return []


def _hit(w: str, q_words: set[str], prefix: bool) -> bool:
    if w in q_words:
        return True
    return prefix and len(w) >= 4 and any(len(q) >= 4 and (q.startswith(w) or w.startswith(q)) for q in q_words)


def select_tables(question: str, db_path: str | Path, min_tables: int = 2, max_col_df: int = 3,
                  prefix: bool = True, neighbours: bool = False) -> list[str] | None:
    """prefix: a word also matches when one is a prefix of the other (>= 4 letters: "prereq" /
    "prerequisites"). neighbours: also keep the tables one foreign key away from a matched table."""
    p = profile(str(db_path))
    q_words, q_phrase = set(words(question)) - STOPWORDS, _phrase(question)
    df: dict[str, int] = {}
    for cw in p.col_words.values():
        for w in cw:
            df[w] = df.get(w, 0) + 1
    kept = set()
    for k in p.names:
        by_name = bool(p.name_words[k]) and all(_hit(w, q_words, prefix) for w in p.name_words[k])
        by_col = any(df[w] <= max_col_df and _hit(w, q_words, prefix) for w in p.col_words[k] if w not in STOPWORDS)
        by_value = any(v in q_phrase for v in p.values[k])
        if by_name or by_col or by_value:
            kept.add(k)
    if len(kept) < min_tables:
        return None
    if neighbours:
        kept |= {n for k in kept for n in p.fk.get(k, ())}
    for a in sorted(kept):
        for b in sorted(kept):
            if a < b:
                kept.update(_path(p.fk, a, b))
    return sorted(p.names[k] for k in kept)
