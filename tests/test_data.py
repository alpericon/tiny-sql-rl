import json
from pathlib import Path

import pytest

from tinysql.data import Example, Exclusions, assert_no_leakage, filter_leakage, load_exclusions, load_spider_split

EXCLUDE_FILE = Path(__file__).resolve().parents[1] / "data" / "spokenspider_exclude.json"


def ex(db, q, sql="SELECT 1"):
    return Example(db_id=db, question=q, query=sql)


def test_real_exclusion_file_has_the_100_spokenspider_questions():
    exclusions = load_exclusions(EXCLUDE_FILE)
    assert len(exclusions.questions) == 100
    # Spider pairs ~2 paraphrased questions per SQL: the 100 questions have only 59 distinct queries,
    # which is why the filter also matches on SQL.
    assert len(exclusions.queries) == 59
    raw = json.loads(EXCLUDE_FILE.read_text(encoding="utf-8"))
    assert {r["query_id"] for r in raw} == {f"train_{i:05d}" for i in range(100)}


def test_filter_removes_by_content_not_position():
    exclusions = Exclusions(questions={("concert", "how many singers are there?")}, queries=set())
    data = [ex("concert", "List all stadiums."), ex("concert", "How many singers are there?")]
    kept, removed = filter_leakage(data, exclusions)
    assert [e.question for e in kept] == ["List all stadiums."]
    assert [e.question for e in removed] == ["How many singers are there?"]


def test_filter_normalises_case_and_whitespace():
    exclusions = Exclusions(questions={("concert", "how many singers are there?")}, queries=set())
    kept, removed = filter_leakage([ex("concert", "  How many   singers are there? ")], exclusions)
    assert not kept and len(removed) == 1


def test_same_question_on_another_db_is_kept():
    exclusions = Exclusions(questions={("concert", "how many singers are there?")}, queries=set())
    kept, _ = filter_leakage([ex("music", "How many singers are there?")], exclusions)
    assert len(kept) == 1


def test_assert_no_leakage_raises_on_leak():
    exclusions = Exclusions(questions={("concert", "how many singers are there?")}, queries=set())
    with pytest.raises(AssertionError):
        assert_no_leakage([ex("concert", "How many singers are there?")], exclusions)


def test_load_spider_split_reads_hf_and_original_formats(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            [
                {"db_id": "a", "question": "q1", "query": "SELECT 1"},
                {"db_id": "b", "question": "q2", "query": "SELECT 2", "sql": {"parsed": "ignored"}},
            ]
        ),
        encoding="utf-8",
    )
    rows = load_spider_split(path)
    assert rows == [Example("a", "q1", "SELECT 1"), Example("b", "q2", "SELECT 2")]


def test_paraphrase_with_same_sql_is_also_excluded(tmp_path):
    path = tmp_path / "ex.json"
    path.write_text(
        json.dumps([{"db_id": "concert", "question": "How many singers are there?", "query": "SELECT count(*) FROM singer"}]),
        encoding="utf-8",
    )
    exclusions = load_exclusions(path)
    para = ex("concert", "What is the total number of singers?", "SELECT  COUNT(*)  FROM singer")
    kept, removed = filter_leakage([para], exclusions)
    assert not kept and removed == [para]
