import sqlite3

import pytest

from tinysql.data import Example
from tinysql.evaluate import evaluate


@pytest.fixture
def db_dir(tmp_path):
    d = tmp_path / "database" / "toy"
    d.mkdir(parents=True)
    conn = sqlite3.connect(d / "toy.sqlite")
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Ada', 30), (2, 'Bo', 41), (3, 'Cy', 25);
        """
    )
    conn.commit()
    conn.close()
    return tmp_path / "database"


EXAMPLES = [
    Example("toy", "How many singers?", "SELECT count(*) FROM singer"),
    Example("toy", "Oldest singer?", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"),
    Example("toy", "Names?", "SELECT name FROM singer"),
    Example("toy", "Ages?", "SELECT age FROM singer"),
]

ANSWERS = {
    "How many singers?": "```sql\nSELECT COUNT(id) FROM singer\n```",  # correct, different SQL
    "Oldest singer?": "```sql\nSELECT name FROM singer ORDER BY age LIMIT 1\n```",  # wrong
    "Names?": "SELECT name FROM singer",  # no fence -> format failure
    "Ages?": "```sql\nSELECT agee FROM singer\n```",  # execution error
}


def fake_generate(batch):
    return [ANSWERS[m[-1]["content"].rsplit("Question: ", 1)[1]] for m in batch]


def test_summary_counts(db_dir):
    records, summary = evaluate(EXAMPLES, db_dir, fake_generate, batch_size=3)
    assert summary["n"] == 4
    assert summary["execution_accuracy"] == pytest.approx(0.25)
    assert summary["format_failure_rate"] == pytest.approx(0.25)
    assert summary["execution_error_rate"] == pytest.approx(0.25)
    assert [r["match"] for r in records] == [True, False, False, False]


def test_records_keep_inputs_and_outputs(db_dir):
    records, _ = evaluate(EXAMPLES[:1], db_dir, fake_generate)
    r = records[0]
    assert r["question"] == "How many singers?"
    assert r["pred_sql"] == "SELECT COUNT(id) FROM singer"
    assert r["gold"] == "SELECT count(*) FROM singer"
    assert r["prompt_chars"] > 0


def test_generator_returning_wrong_count_raises(db_dir):
    with pytest.raises(ValueError):
        evaluate(EXAMPLES, db_dir, lambda batch: ["x"])


def test_score_sql_matches_evaluate_semantics(db_dir):
    from tinysql.evaluate import score_sql

    db = db_dir / "toy" / "toy.sqlite"
    assert score_sql(db, "SELECT count(*) FROM singer", "SELECT COUNT(id) FROM singer") == {
        "exec_ok": True, "match": True, "error": None}
    wrong = score_sql(db, "SELECT name FROM singer ORDER BY age DESC LIMIT 1",
                      "SELECT name FROM singer ORDER BY age LIMIT 1")
    assert wrong["exec_ok"] and not wrong["match"]
    bad = score_sql(db, "SELECT age FROM singer", "SELECT agee FROM singer")
    assert not bad["exec_ok"] and not bad["match"] and bad["error"]


def test_sample_rows_reach_the_prompt(db_dir):
    seen = []

    def gen(batch):
        seen.extend(m[-1]["content"] for m in batch)
        return fake_generate(batch)

    evaluate(EXAMPLES[:1], db_dir, gen, sample_rows=0)
    evaluate(EXAMPLES[:1], db_dir, gen, sample_rows=3)
    assert "Example rows" not in seen[0]
    assert "-- Example rows from singer:" in seen[1] and "Ada" in seen[1]


def test_demo_passes_sample_rows(db_dir):
    from tinysql.demo import answer_question

    seen = []

    def gen(batch):
        seen.append(batch[0][-1]["content"])
        return ["```sql\nSELECT count(*) FROM singer\n```"]

    answer_question("How many?", db_dir / "toy" / "toy.sqlite", gen, sample_rows=3)
    answer_question("How many?", db_dir / "toy" / "toy.sqlite", gen)
    assert "Example rows" in seen[0] and "Example rows" not in seen[1]
