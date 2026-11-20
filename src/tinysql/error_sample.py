"""Sample for the wrong-result error analysis (PLAN-week9 E1).

The population is the questions whose prediction runs but returns the wrong answer on our metric
and is also wrong on the official metric. The sample is stratified by official hardness, in
proportion to the population (largest-remainder rounding), with a fixed seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence


def wrong_on_both(records: Sequence[dict], official_rows: Sequence[dict]) -> list[int]:
    if len(records) != len(official_rows) or any(o["idx"] != i for i, o in enumerate(official_rows)):
        raise ValueError("records and official rows are not aligned by index")
    return [i for i, (r, o) in enumerate(zip(records, official_rows))
            if r["format_ok"] and r["exec_ok"] and not r["match"] and not o["official"]]


def allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    total = sum(sizes.values())
    exact = {k: n * v / total for k, v in sizes.items()}
    out = {k: int(x) for k, x in exact.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(exact[k] - out[k]), k))
    for k in by_remainder[: n - sum(out.values())]:
        out[k] += 1
    return out


def stratified_sample(strata: Mapping[int, str], n: int, seed: int) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in sorted(strata.items()):
        groups[s].append(i)
    rng = random.Random(seed)
    chosen = []
    for s, k in sorted(allocate({s: len(g) for s, g in groups.items()}, n).items()):
        chosen += rng.sample(groups[s], k)
    return sorted(chosen)
