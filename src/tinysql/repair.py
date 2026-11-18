"""Execution-error repair: one retry with the error message (PLAN-week4 addendum Y9).

The rule is fixed in advance: retry at most once, and only when the answer has no ```sql block or
its SQL does not execute. The retry continues the same conversation with the model's own answer and
one fixed user turn. A wrong-but-executing answer is never retried.
"""

from __future__ import annotations

Messages = list[dict[str, str]]

REPAIR_ERROR = ("That SQL failed: `{error}`. Write a corrected SQLite query that answers the question, "
                "in a ```sql block.")
REPAIR_FORMAT = ("Your answer had no ```sql block. Write a SQLite query that answers the question, "
                 "in a ```sql block.")
NO_BLOCK = "no ```sql block"


def repair_messages(messages: Messages, completion: str, error: str | None) -> Messages:
    """The retry prompt. ``error`` is the executor's message, or None when no SQL block was found."""
    turn = REPAIR_FORMAT if error is None else REPAIR_ERROR.format(error=error)
    return [*messages, {"role": "assistant", "content": completion}, {"role": "user", "content": turn}]
