"""Spider loading and the SpokenSpider leakage filter (SPEC §4).

The 100 SpokenSpider questions used in voice-to-sql come from Spider *train*.
They must never be trained on, otherwise the voice evaluation (RQ3, RQ4) leaks.
Matching is by (db_id, normalised question), never by index, so a re-ordered or
re-downloaded Spider cannot silently break the filter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

Key = tuple[str, str]


@dataclass(frozen=True)
class Example:
    db_id: str
    question: str
    query: str


def _norm(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().lower()


def _key(db_id: str, question: str) -> Key:
    return (db_id, _norm(question))


def load_spider_split(path: str | Path) -> list[Example]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Example(r["db_id"], r["question"], r["query"]) for r in rows]


def load_exclusions(path: str | Path) -> set[Key]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {_key(r["db_id"], r["question"]) for r in rows}


def filter_leakage(examples: list[Example], exclusions: set[Key]) -> tuple[list[Example], list[Example]]:
    kept, removed = [], []
    for e in examples:
        (removed if _key(e.db_id, e.question) in exclusions else kept).append(e)
    return kept, removed


def assert_no_leakage(examples: list[Example], exclusions: set[Key]) -> None:
    leaked = [e for e in examples if _key(e.db_id, e.question) in exclusions]
    assert not leaked, f"{len(leaked)} SpokenSpider questions found in training data, e.g. {leaked[0]}"
