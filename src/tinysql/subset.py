"""Fixed, seeded dev subset for checkpoint selection (PLAN-week2 W3).

Ids store the dev.json index plus (db_id, question), so a re-ordered or different
dev.json is detected instead of silently evaluating other questions.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from tinysql.data import Example


def select_subset(dev: Sequence[Example], eligible: Iterable[int], n: int, seed: int) -> list[dict]:
    pool = sorted(set(eligible))
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} eligible examples, need {n}")
    chosen = sorted(random.Random(seed).sample(pool, n))
    return [{"idx": i, "db_id": dev[i].db_id, "question": dev[i].question} for i in chosen]


def load_subset(dev: Sequence[Example], ids: Sequence[dict]) -> list[Example]:
    out = []
    for r in ids:
        e = dev[r["idx"]]
        if (e.db_id, e.question) != (r["db_id"], r["question"]):
            raise ValueError(f"dev.json does not match subset id {r['idx']}: {e.db_id!r} {e.question!r}")
        out.append(e)
    return out
