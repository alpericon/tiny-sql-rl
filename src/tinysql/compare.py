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

FLOAT_DECIMALS = 6


def _normalise_value(v):
    # 3 and 3.0 are the same answer, and float noise (0.30000000000000004) is ignored.
    if isinstance(v, float):
        v = round(v, FLOAT_DECIMALS)
        if v.is_integer():
            return int(v)
    return v


def _normalise_row(row: tuple) -> tuple:
    return tuple(_normalise_value(v) for v in row)


def results_match(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    pred_n = [_normalise_row(r) for r in pred]
    gold_n = [_normalise_row(r) for r in gold]
    if ordered:
        return pred_n == gold_n
    # Multiset comparison: row order ignored, duplicates counted. Counter avoids
    # sorting, so rows mixing None / int / str never raise TypeError.
    return Counter(pred_n) == Counter(gold_n)


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
