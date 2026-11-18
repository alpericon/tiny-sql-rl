import hashlib
import sqlite3

import pytest

from tinysql.executor import execute_readonly


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


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_select_returns_rows(db):
    res = execute_readonly(db, "SELECT name FROM singer WHERE age > 26 ORDER BY id")
    assert res.ok
    assert res.rows == [("Ada",), ("Bo",)]
    assert res.error is None


def test_trailing_semicolon_and_whitespace_ok(db):
    res = execute_readonly(db, "  SELECT count(*) FROM singer ;  \n")
    assert res.ok and res.rows == [(3,)]


def test_with_cte_ok(db):
    res = execute_readonly(db, "WITH old AS (SELECT * FROM singer WHERE age > 30) SELECT name FROM old")
    assert res.ok and res.rows == [("Bo",)]


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO singer VALUES (4, 'Di', 50)",
        "UPDATE singer SET age = 0",
        "DELETE FROM singer",
        "DROP TABLE singer",
        "CREATE TABLE x (a)",
        "ATTACH DATABASE 'evil.sqlite' AS evil",
        "PRAGMA writable_schema = ON",
        "WITH t AS (SELECT 1) INSERT INTO singer SELECT 9, 'Z', 1 FROM t",
    ],
)
def test_writes_and_side_effects_rejected(db, sql):
    before = _digest(db)
    res = execute_readonly(db, sql)
    assert not res.ok
    assert res.rows is None
    assert _digest(db) == before


def test_multiple_statements_rejected(db):
    before = _digest(db)
    res = execute_readonly(db, "SELECT 1; DELETE FROM singer")
    assert not res.ok
    assert _digest(db) == before


def test_semicolon_inside_string_is_not_a_second_statement(db):
    res = execute_readonly(db, "SELECT 'a;b'")
    assert res.ok and res.rows == [("a;b",)]


def test_syntax_error_reported(db):
    res = execute_readonly(db, "SELEC name FROM singer")
    assert not res.ok
    assert res.error.startswith("sql_error")


def test_empty_sql_rejected(db):
    assert not execute_readonly(db, "   ;  ").ok


def test_infinite_query_times_out(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    res = execute_readonly(db, sql, timeout_s=0.5)
    assert not res.ok
    assert res.error == "timeout"


def test_row_cap(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 100) SELECT x FROM c"
    res = execute_readonly(db, sql, max_rows=10)
    assert res.ok
    assert len(res.rows) == 10
    assert res.truncated


def test_missing_db_is_not_created(tmp_path):
    path = tmp_path / "nope.sqlite"
    res = execute_readonly(path, "SELECT 1")
    assert not res.ok
    assert not path.exists()


# --- review fixes (CRITICAL / HIGH) ---------------------------------------


def test_huge_blob_is_rejected_quickly(db):
    import time

    t = time.monotonic()
    res = execute_readonly(db, "SELECT zeroblob(1000000000)", timeout_s=2)
    assert not res.ok
    assert time.monotonic() - t < 2


def test_many_large_rows_hit_the_result_size_cap(db):
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 1000) "
        "SELECT zeroblob(500000) FROM c"
    )
    res = execute_readonly(db, sql, timeout_s=5)
    assert not res.ok
    assert res.error.startswith("rejected")


@pytest.mark.parametrize("sql", ["SELECT 1\x00", "SELECT '\ud800'"])
def test_unencodable_sql_is_rejected_not_raised(db, sql):
    res = execute_readonly(db, sql)
    assert not res.ok


def test_result_carries_column_names(db):
    res = execute_readonly(db, "SELECT name, age AS years FROM singer LIMIT 1")
    assert res.columns == ("name", "years")
    assert execute_readonly(db, "SELECT nope FROM singer").columns is None
