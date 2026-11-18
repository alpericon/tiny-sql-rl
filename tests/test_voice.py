import csv

import pytest

from tinysql.voice import CONDITIONS, load_voice, slim_metadata, voice_examples

META_FIELDS = ["query_id", "db_id", "snr_level", "question", "p_transcript", "sql", "p_generated_sql",
               "p_asr_wer", "latency_asr_s", "latency_sql_s", "latency_total_s", "speaker"]


def meta_row(qid, snr, transcript):
    return {"query_id": qid, "db_id": "toy", "snr_level": snr, "question": f"q {qid} ?",
            "p_transcript": transcript, "sql": f"SELECT {qid[-1]}", "p_generated_sql": "SELECT 1;",
            "p_asr_wer": "0.1", "latency_asr_s": "1.0", "latency_sql_s": "4.0", "latency_total_s": "5.5",
            "speaker": ""}


@pytest.fixture
def voice_csv(tmp_path):
    rows = [meta_row(q, s, f"heard {q} {s}") for s in ["0db", "clean", "10db", "20db"] for q in ["t_1", "t_0"]]
    meta = tmp_path / "metadata.csv"
    with meta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "slim.csv"
    slim_metadata(meta, out)
    return out


def test_slim_keeps_only_needed_columns(voice_csv):
    rows = load_voice(voice_csv)
    assert len(rows) == 8
    assert set(rows[0]) == {"query_id", "db_id", "snr", "question", "transcript", "gold", "sqlcoder_sql",
                            "wer", "latency_asr_s", "latency_sql_s", "latency_total_s"}


def test_conditions_are_aligned_by_query_id(voice_csv):
    rows = load_voice(voice_csv)
    for c in CONDITIONS:
        exs = voice_examples(rows, c)
        assert [e.query for e in exs] == ["SELECT 0", "SELECT 1"]
    assert [e.question for e in voice_examples(rows, "text")] == ["q t_0 ?", "q t_1 ?"]
    assert [e.question for e in voice_examples(rows, "10db")] == ["heard t_0 10db", "heard t_1 10db"]


def test_unknown_condition_raises(voice_csv):
    with pytest.raises(ValueError):
        voice_examples(load_voice(voice_csv), "5db")
