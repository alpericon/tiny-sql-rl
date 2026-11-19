import sqlite3

import pytest

from tinysql.ground import build_value_index, ground_sql


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE farm (id INTEGER PRIMARY KEY, theme TEXT, status TEXT, host TEXT, year INTEGER);
        INSERT INTO farm VALUES (1, 'Aliens', 'Village', 'O''Brien', 2014);
        INSERT INTO farm VALUES (2, 'MTV  Cube', 'Town', 'Ann', 2015);
        INSERT INTO farm VALUES (3, 'Codes_A', 'town', 'Bo', 2016);
        CREATE TABLE big (v TEXT);
        """
    )
    conn.executemany("INSERT INTO big VALUES (?)", [("Hidden",)] * 30)
    conn.commit()
    conn.close()
    return path


def idx(db, **kw):
    return build_value_index(db, **kw)


def test_case_and_whitespace_fixed_when_unique(db):
    sql, reps = ground_sql("SELECT host FROM farm WHERE theme != 'aliens' AND status = 'village'", idx(db))
    assert sql == "SELECT host FROM farm WHERE theme != 'Aliens' AND status = 'Village'"
    assert reps == [("aliens", "Aliens"), ("village", "Village")]
    sql, _ = ground_sql("SELECT 1 FROM farm WHERE theme = 'mtv cube'", idx(db))
    assert sql == "SELECT 1 FROM farm WHERE theme = 'MTV  Cube'"


def test_exact_ambiguous_and_unmatched_literals_untouched(db):
    for q in ["SELECT 1 FROM farm WHERE theme = 'Aliens'",      # exact match
              "SELECT 1 FROM farm WHERE status = 'TOWN'",       # 'Town' and 'town' both exist
              "SELECT 1 FROM farm WHERE theme = 'Martians'"]:   # no match at all
        assert ground_sql(q, idx(db)) == (q, [])


def test_like_patterns_numbers_and_identifiers_untouched(db):
    for q in ["SELECT 1 FROM farm WHERE theme LIKE '%alien%'",
              "SELECT 1 FROM farm WHERE theme = 'codes_a'",     # contains _ : treated as a pattern char
              "SELECT 1 FROM farm WHERE year = 2014",
              'SELECT "aliens" FROM farm']:                      # double quotes = identifier
        assert ground_sql(q, idx(db)) == (q, [])


def test_escaped_quotes_are_parsed_and_reescaped(db):
    sql, reps = ground_sql("SELECT 1 FROM farm WHERE host = 'o''brien' AND theme = 'aliens'", idx(db))
    assert sql == "SELECT 1 FROM farm WHERE host = 'O''Brien' AND theme = 'Aliens'"
    assert reps == [("o'brien", "O'Brien"), ("aliens", "Aliens")]


def test_large_tables_are_skipped(db):
    assert ground_sql("SELECT 1 FROM big WHERE v = 'hidden'", idx(db, max_rows=10))[1] == []
    assert ground_sql("SELECT 1 FROM big WHERE v = 'hidden'", idx(db, max_rows=100))[1] == [("hidden", "Hidden")]


def test_idempotent(db):
    once, _ = ground_sql("SELECT 1 FROM farm WHERE theme = 'aliens'", idx(db))
    assert ground_sql(once, idx(db)) == (once, [])
