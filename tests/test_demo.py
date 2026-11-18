import sqlite3

import pytest

from tinysql.demo import answer_question


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
        "INSERT INTO singer VALUES (1, 'Ada', 30), (2, 'Bo', 41), (3, 'Cy', 25);"
    )
    conn.commit()
    conn.close()
    return path


def fixed(completion):
    calls = []

    def generate(batch):
        calls.append(batch)
        return [completion]
    generate.calls = calls
    return generate


def test_question_to_table_and_answer(db):
    gen = fixed("```sql\nSELECT name, age FROM singer ORDER BY age DESC\n```")
    r = answer_question("Who is oldest?", db, gen)
    assert r.error is None
    assert r.sql == "SELECT name, age FROM singer ORDER BY age DESC"
    assert r.columns == ("name", "age") and r.rows[0] == ("Bo", 41)
    assert r.answer == "Found 3 rows."
    # the model saw the schema and the question in the training prompt format
    assert "singer" in gen.calls[0][0][-1]["content"] and gen.calls[0][0][-1]["content"].endswith("Who is oldest?")


def test_single_value_answer(db):
    r = answer_question("How many?", db, fixed("```sql\nSELECT count(*) FROM singer\n```"))
    assert r.answer == "The answer is 3."


def test_empty_result_and_truncation(db):
    assert answer_question("q", db, fixed("```sql\nSELECT name FROM singer WHERE age > 99\n```")).answer == "No rows match."
    r = answer_question("q", db, fixed("```sql\nSELECT name FROM singer\n```"), max_rows=2)
    assert len(r.rows) == 2 and r.answer == "Showing the first 2 rows."


def test_errors_are_reported_not_raised(db):
    r = answer_question("q", db, fixed("I cannot answer that."))
    assert r.sql is None and r.rows is None and "no SQL" in r.error
    r = answer_question("q", db, fixed("```sql\nDELETE FROM singer\n```"))
    assert r.rows is None and r.error.startswith("rejected")
    r = answer_question("  ", db, fixed("x"))
    assert r.error == "empty question"


def test_single_value_formatting(db):
    assert answer_question("q", db, fixed("```sql\nSELECT avg(age) FROM singer\n```")).answer == "The answer is 32."
    assert answer_question("q", db, fixed("```sql\nSELECT 10.0 / 3\n```")).answer == "The answer is 3.33."
    assert answer_question("q", db, fixed("```sql\nSELECT max(age) FROM singer WHERE age > 99\n```")).answer == \
        "The answer is empty (NULL)."
