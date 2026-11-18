"""SpokenSpider transcripts from the voice-to-sql project (PLAN-week3 X6).

voice-to-sql's metadata.csv holds one row per (query, SNR condition): the ASR transcript,
the gold SQL and the SQL its SQLCoder stage produced. ``slim_metadata`` keeps the columns
this project needs; ``voice_examples`` turns one condition into Examples, sorted by query_id
so every condition lists the same queries in the same order (needed for paired tests).
"""

from __future__ import annotations

import csv
from pathlib import Path

from tinysql.data import Example

SNRS = ("clean", "20db", "10db", "0db")
# "text" is the original question text (no ASR); the others are ASR transcripts per SNR.
CONDITIONS = ("text",) + SNRS

_SLIM = {  # slim column <- metadata.csv column
    "query_id": "query_id",
    "db_id": "db_id",
    "snr": "snr_level",
    "question": "question",
    "transcript": "p_transcript",
    "gold": "sql",
    "sqlcoder_sql": "p_generated_sql",
    "wer": "p_asr_wer",
    "latency_asr_s": "latency_asr_s",
    "latency_sql_s": "latency_sql_s",
    "latency_total_s": "latency_total_s",
}


def slim_metadata(metadata_csv: str | Path, out_csv: str | Path) -> None:
    with open(metadata_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(_SLIM))
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["query_id"], SNRS.index(r["snr_level"]))):
            w.writerow({k: r[src] for k, src in _SLIM.items()})


def load_voice(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def voice_rows(rows: list[dict], condition: str) -> list[dict]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    snr = "clean" if condition == "text" else condition
    return sorted((r for r in rows if r["snr"] == snr), key=lambda r: r["query_id"])


def voice_examples(rows: list[dict], condition: str) -> list[Example]:
    field = "question" if condition == "text" else "transcript"
    return [Example(r["db_id"], r[field], r["gold"]) for r in voice_rows(rows, condition)]
