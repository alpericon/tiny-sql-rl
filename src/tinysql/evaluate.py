"""Model-agnostic Spider evaluation (SPEC §5.4).

``generate`` takes a batch of chat-message lists and returns one completion per item,
so the same harness scores HF models, vLLM, llama.cpp or a fake in tests.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from tinysql.compare import gold_is_ordered, results_match
from tinysql.data import Example
from tinysql.executor import execute_readonly
from tinysql.prompt import build_messages, serialize_schema
from tinysql.repair import NO_BLOCK, repair_messages
from tinysql.reward import extract_sql

Messages = list[dict[str, str]]
Generate = Callable[[list[Messages]], list[str]]

# Evaluation is not a reward: allow large gold answers (e.g. wta_1 in dev) so all 1,034
# dev questions can be scored, matching the standard Spider dev set.
EVAL_MAX_ROWS = 200_000


def score_sql(db_path: str | Path, gold_sql: str, pred_sql: str, timeout_s: float = 30.0) -> dict:
    """EX of one predicted query against gold: {exec_ok, match, error}. Also used to re-score
    SQL produced by other systems (e.g. SQLCoder in voice-to-sql) with the same comparator."""
    gold = execute_readonly(db_path, gold_sql, timeout_s=timeout_s, max_rows=EVAL_MAX_ROWS)
    pred = execute_readonly(db_path, pred_sql, timeout_s=timeout_s, max_rows=EVAL_MAX_ROWS)
    match = False
    if pred.ok and gold.ok and not pred.truncated and not gold.truncated:
        match = results_match(pred.rows, gold.rows, ordered=gold_is_ordered(gold_sql))
    error = pred.error if not pred.ok else (None if gold.ok else f"gold: {gold.error}")
    return {"exec_ok": pred.ok, "match": match, "error": error}


def evaluate(
    examples: Sequence[Example],
    database_dir: str | Path,
    generate: Generate,
    batch_size: int = 8,
    sample_rows: int = 0,
    timeout_s: float = 30.0,
    repair: bool = False,
) -> tuple[list[dict], dict]:
    """With ``repair=True``, an answer with no SQL block or with SQL that does not execute gets one
    retry (tinysql.repair). The first attempt is kept in ``first_*`` fields, so the no-repair result
    of the same run is recoverable."""
    database_dir = Path(database_dir)
    schema_cache: dict[str, str] = {}

    def db_path(db_id: str) -> Path:
        return database_dir / db_id / f"{db_id}.sqlite"

    def messages_for(e: Example) -> Messages:
        if e.db_id not in schema_cache:
            schema_cache[e.db_id] = serialize_schema(db_path(e.db_id), sample_rows=sample_rows)
        return build_messages(schema_cache[e.db_id], e.question)

    records: list[dict] = []
    gen_seconds = 0.0
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        prompts = [messages_for(e) for e in batch]
        t0 = time.perf_counter()
        completions = generate(prompts)
        gen_seconds += time.perf_counter() - t0
        if len(completions) != len(batch):
            raise ValueError(f"generate returned {len(completions)} completions for {len(batch)} prompts")

        def scored(e: Example, completion: str) -> dict:
            pred_sql = extract_sql(completion)
            out = {"completion": completion, "pred_sql": pred_sql, "format_ok": pred_sql is not None,
                   "exec_ok": False, "match": False, "error": None}
            if pred_sql is not None:
                out.update(score_sql(db_path(e.db_id), e.query, pred_sql, timeout_s=timeout_s))
            return out

        batch_records = []
        for e, msgs, completion in zip(batch, prompts, completions):
            rec = {"db_id": e.db_id, "question": e.question, "gold": e.query,
                   "prompt_chars": sum(len(m["content"]) for m in msgs), **scored(e, completion)}
            batch_records.append(rec)

        if repair:
            failed = [i for i, r in enumerate(batch_records) if not r["format_ok"] or not r["exec_ok"]]
            for r in batch_records:
                r.update(repaired=False, first_completion=r["completion"], first_pred_sql=r["pred_sql"],
                         first_match=r["match"],
                         first_error=None if r["format_ok"] and r["exec_ok"] else (r["error"] if r["format_ok"] else NO_BLOCK))
            if failed:
                retry = [repair_messages(prompts[i], batch_records[i]["completion"],
                                         batch_records[i]["error"] if batch_records[i]["format_ok"] else None)
                         for i in failed]
                t0 = time.perf_counter()
                second = generate(retry)
                gen_seconds += time.perf_counter() - t0
                if len(second) != len(retry):
                    raise ValueError(f"generate returned {len(second)} completions for {len(retry)} repair prompts")
                for i, completion in zip(failed, second):
                    batch_records[i].update(repaired=True, **scored(batch[i], completion))
        records.extend(batch_records)

    n = len(records)
    summary = {
        "n": n,
        "execution_accuracy": sum(r["match"] for r in records) / n if n else 0.0,
        "format_failure_rate": sum(not r["format_ok"] for r in records) / n if n else 0.0,
        "execution_error_rate": sum(r["format_ok"] and not r["exec_ok"] for r in records) / n if n else 0.0,
        "max_prompt_chars": max((r["prompt_chars"] for r in records), default=0),
        "median_prompt_chars": statistics.median([r["prompt_chars"] for r in records]) if records else 0,
        "generation_seconds_total": round(gen_seconds, 2),
    }
    if repair:
        summary.update(
            first_attempt_execution_accuracy=sum(r["first_match"] for r in records) / n if n else 0.0,
            repair_attempts=sum(r["repaired"] for r in records),
            repair_rescued=sum(r["match"] and not r["first_match"] for r in records),
        )
    return records, summary
