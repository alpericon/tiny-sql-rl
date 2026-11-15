"""TRL GRPOTrainer adapter for the execution reward (PLAN-week2 W2).

TRL calls ``reward_fn(prompts=..., completions=..., **dataset_columns, **extras)`` and expects
one float per completion. Completions are strings, or lists of chat messages for
conversational datasets. The dataset columns ``db_path`` and ``gold`` come from
``grpo_data.build_grpo_records``.
"""

from __future__ import annotations

from tinysql.reward import sql_reward


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        content = completion[-1].get("content") if isinstance(completion[-1], dict) else None
        return content if isinstance(content, str) else ""
    return ""


def make_reward_fn(binary: bool = False, timeout_s: float = 5.0, max_rows: int = 10_000):
    def reward_fn(prompts, completions, db_path, gold, **_ignored) -> list[float]:
        if not (len(completions) == len(db_path) == len(gold)):
            raise ValueError(f"misaligned batch: {len(completions)} completions, {len(db_path)} db_path, {len(gold)} gold")
        # sql_reward never raises on model output; it raises only when the gold query
        # itself fails, which means the training data is broken and must stop the run.
        return [
            sql_reward(_text(c), g, p, binary=binary, timeout_s=timeout_s, max_rows=max_rows)
            for c, p, g in zip(completions, db_path, gold)
        ]

    reward_fn.__name__ = "sql_exec_reward_binary" if binary else "sql_exec_reward"
    return reward_fn
