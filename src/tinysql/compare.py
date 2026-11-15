"""Execution-result comparison (Spider-style execution accuracy).

Known limitation, shared with the official Spider EX metric: when the gold query is
ordered, rows that tie on the ORDER BY key must appear in the same order as gold.
A prediction that breaks ties differently scores as a mismatch.
"""

from __future__ import annotations

import re
from collections import Counter

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_BRACKET_OR_BACKTICK_IDENT = re.compile(r"\[[^\]]*\]|`[^`]*`")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)

FLOAT_SIGNIFICANT_DIGITS = 12


def _normalise_value(v):
    # 3 and 3.0 are the same answer, and float noise (0.30000000000000004) is ignored.
    # Rounding is relative (significant digits), so tiny but different values stay different.
    if isinstance(v, float):
        v = float(f"{v:.{FLOAT_SIGNIFICANT_DIGITS}g}")
        if v.is_integer():
            return int(v)
    return v


def _normalise_row(row: tuple) -> tuple:
    return tuple(_normalise_value(v) for v in row)


MAX_COLUMN_MAPPINGS = 1_000


def _rows_equal(pred_n: list[tuple], gold_n: list[tuple], ordered: bool) -> bool:
    if ordered:
        return pred_n == gold_n
    # Multiset comparison: row order ignored, duplicates counted. Counter avoids
    # sorting, so rows mixing None / int / str never raise TypeError.
    return Counter(pred_n) == Counter(gold_n)


def _column_mappings(pred_n: list[tuple], gold_n: list[tuple]):
    """Yield column mappings m (gold column j <- pred column m[j]) worth trying.

    A pred column is a candidate for a gold column only if both hold the same multiset
    of values. Pred columns with identical contents are interchangeable, so only one of
    them is tried at each position. At most MAX_COLUMN_MAPPINGS mappings are yielded.
    """
    width = len(gold_n[0])
    pred_cols = [tuple(r[i] for r in pred_n) for i in range(width)]
    gold_bags = [Counter(r[j] for r in gold_n) for j in range(width)]
    pred_bags = [Counter(c) for c in pred_cols]
    candidates = [[i for i in range(width) if pred_bags[i] == gold_bags[j]] for j in range(width)]
    if any(not c for c in candidates):
        return
    budget = [MAX_COLUMN_MAPPINGS]
    used = [False] * width
    mapping: list[int] = []

    def extend(j):
        if budget[0] <= 0:
            return
        if j == width:
            budget[0] -= 1
            yield tuple(mapping)
            return
        tried = set()
        for i in candidates[j]:
            if used[i] or pred_cols[i] in tried:
                continue
            tried.add(pred_cols[i])
            used[i] = True
            mapping.append(i)
            yield from extend(j + 1)
            mapping.pop()
            used[i] = False

    yield from extend(0)


def results_match(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    """Spider-style match: rows compared as a multiset (or in order), columns in any order."""
    pred_n = [_normalise_row(r) for r in pred]
    gold_n = [_normalise_row(r) for r in gold]
    if _rows_equal(pred_n, gold_n, ordered):
        return True
    if not pred_n or not gold_n or len(pred_n) != len(gold_n):
        return False
    if len(pred_n[0]) != len(gold_n[0]) or len(gold_n[0]) < 2:
        return False
    for m in _column_mappings(pred_n, gold_n):
        if _rows_equal([tuple(r[i] for i in m) for r in pred_n], gold_n, ordered):
            return True
    return False


def gold_is_ordered(gold_sql: str) -> bool:
    """True if the *outermost* query has ORDER BY, so row order is part of the answer."""
    sql = _STRING_LITERAL.sub("''", gold_sql)
    sql = _BRACKET_OR_BACKTICK_IDENT.sub("x", sql)
    sql = _BLOCK_COMMENT.sub(" ", sql)
    sql = _LINE_COMMENT.sub(" ", sql)
    depth = 0
    outer = []
    for ch in sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            outer.append(ch)
    return bool(_ORDER_BY.search("".join(outer)))
