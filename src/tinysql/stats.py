"""Paired comparison of two models on the same questions (exact two-sided McNemar)."""

from __future__ import annotations

import math


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _outcome(r: dict) -> str:
    if r["match"]:
        return "match"
    if not r["format_ok"]:
        return "format_fail"
    if not r["exec_ok"]:
        return "exec_error"
    return "wrong_result"


def paired_summary(base: list[dict], cand: list[dict]) -> dict:
    """EX of both, discordant counts, where the wins came from, and the candidate's error breakdown."""
    if len(base) != len(cand):
        raise ValueError(f"records not aligned: {len(base)} vs {len(cand)}")
    for i, (b, c) in enumerate(zip(base, cand)):
        key = ("db_id", "question", "gold")
        if any(b.get(k) != c.get(k) for k in key):
            raise ValueError(f"records not aligned at row {i}: {[b.get(k) for k in key]} vs {[c.get(k) for k in key]}")
    errors = ("format_fail", "exec_error", "wrong_result")
    wins = [b for b, c in zip(base, cand) if c["match"] and not b["match"]]
    losses = sum(b["match"] and not c["match"] for b, c in zip(base, cand))
    n = len(base)
    return {
        "n": n,
        "base_ex": sum(r["match"] for r in base) / n,
        "cand_ex": sum(r["match"] for r in cand) / n,
        "wins": len(wins),
        "losses": losses,
        "p": mcnemar_exact(len(wins), losses),
        "wins_by_base_outcome": {e: sum(_outcome(b) == e for b in wins) for e in errors},
        "base_breakdown": {e: sum(_outcome(r) == e for r in base) for e in errors},
        "cand_breakdown": {e: sum(_outcome(r) == e for r in cand) for e in errors},
    }
