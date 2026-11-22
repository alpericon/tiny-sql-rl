"""Spider loading and the SpokenSpider leakage filter (SPEC §4).

The 100 SpokenSpider questions used in voice-to-sql come from Spider *train*.
They must never be trained on, otherwise the voice evaluation (RQ3, RQ4) leaks.

An example is excluded if it matches either
- the same (db_id, normalised question), or
- the same (db_id, normalised gold SQL): Spider pairs paraphrased questions with
  identical SQL, and a paraphrase would leak the answer just as well.
Matching is by content, never by index, so a re-ordered or re-downloaded Spider
cannot silently break the filter. This is deliberately conservative.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

Key = tuple[str, str]


@dataclass(frozen=True)
class Example:
    db_id: str
    question: str
    query: str


@dataclass
class Exclusions:
    questions: set[Key] = field(default_factory=set)
    queries: set[Key] = field(default_factory=set)

    def matches(self, e: Example) -> bool:
        return (e.db_id, _norm(e.question)) in self.questions or (e.db_id, _norm(e.query)) in self.queries


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_spider_split(path: str | Path) -> list[Example]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Example(r["db_id"], r["question"], r["query"]) for r in rows]


def spider_split(spider_dir: str | Path, split: str) -> tuple[Path, Path]:
    """(questions file, databases directory) of an evaluation split: dev, or the held-out test split
    released 02/05/2024 (PLAN-week11)."""
    d = Path(spider_dir)
    if split == "dev":
        return d / "dev.json", d / "database"
    if split == "test":
        return d / "test.json", d / "test_database"
    raise ValueError(f"unknown evaluation split {split!r}; expected dev or test")


def load_exclusions(path: str | Path) -> Exclusions:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return Exclusions(
        questions={(r["db_id"], _norm(r["question"])) for r in rows},
        queries={(r["db_id"], _norm(r.get("sql") or r.get("query"))) for r in rows},
    )


def filter_leakage(examples: list[Example], exclusions: Exclusions) -> tuple[list[Example], list[Example]]:
    kept, removed = [], []
    for e in examples:
        (removed if exclusions.matches(e) else kept).append(e)
    return kept, removed


def assert_no_leakage(examples: list[Example], exclusions: Exclusions) -> None:
    leaked = [e for e in examples if exclusions.matches(e)]
    assert not leaked, f"{len(leaked)} SpokenSpider questions found in training data, e.g. {leaked[0]}"
