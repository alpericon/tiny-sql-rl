import sqlite3

import pytest

from tinysql.reward import extract_sql, sql_reward

GOLD = "SELECT name FROM singer WHERE age > 26"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Ada', 30), (2, 'Bo', 41), (3, 'Cy', 25);
        """
    )
    conn.commit()
    conn.close()
    return path


def wrap(sql):
    return f"Here is the query:\n```sql\n{sql}\n```"


def test_extract_from_fenced_block():
    assert extract_sql(wrap("SELECT 1")) == "SELECT 1"


def test_extract_takes_last_block():
    text = "```sql\nSELECT 1\n```\nActually:\n```sql\nSELECT 2\n```"
    assert extract_sql(text) == "SELECT 2"


def test_extract_none_without_block():
    assert extract_sql("SELECT 1") is None


def test_correct_answer_gets_full_reward(db):
    assert sql_reward(wrap("SELECT name FROM singer WHERE age >= 30"), GOLD, db) == 1.0


def test_valid_but_wrong_gets_partial_credit(db):
    assert sql_reward(wrap("SELECT name FROM singer"), GOLD, db) == 0.1


def test_binary_mode_gives_no_partial_credit(db):
    assert sql_reward(wrap("SELECT name FROM singer"), GOLD, db, binary=True) == 0.0


def test_syntax_error_gets_zero(db):
    assert sql_reward(wrap("SELEC name FROM singer"), GOLD, db) == 0.0


def test_missing_format_gets_zero(db):
    assert sql_reward("SELECT name FROM singer WHERE age > 26", GOLD, db) == 0.0


def test_write_attempt_gets_zero(db):
    assert sql_reward(wrap("DELETE FROM singer"), GOLD, db) == 0.0


def test_order_matters_only_when_gold_orders(db):
    gold_ordered = "SELECT name FROM singer ORDER BY age DESC"
    assert sql_reward(wrap("SELECT name FROM singer ORDER BY age"), gold_ordered, db) == 0.1
    assert sql_reward(wrap("SELECT name FROM singer ORDER BY age DESC"), gold_ordered, db) == 1.0


def test_broken_gold_raises(db):
    # A gold query that does not run is a data bug, not a model failure.
    with pytest.raises(ValueError):
        sql_reward(wrap("SELECT 1"), "SELEC broken", db)
