"""PLAN-week10 P1: lexical table pruning."""

import sqlite3

import pytest

from tinysql.prompt import serialize_schema
from tinysql.prune import gold_tables, select_tables


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "toy.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE stadium (stadium_id INTEGER PRIMARY KEY, name TEXT, location TEXT);
        CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, stadium_id INTEGER, year INTEGER,
                              FOREIGN KEY (stadium_id) REFERENCES stadium(stadium_id));
        CREATE TABLE singer_in_concert (concert_id INTEGER, singer_id INTEGER,
                              FOREIGN KEY (concert_id) REFERENCES concert(concert_id),
                              FOREIGN KEY (singer_id) REFERENCES singer(singer_id));
        CREATE TABLE pets (pet_id INTEGER PRIMARY KEY, pet_type TEXT);
        INSERT INTO singer VALUES (1, 'Joe Sharp', 'Netherlands');
        INSERT INTO stadium VALUES (1, 'Hampden Park', 'Glasgow');
        INSERT INTO concert VALUES (1, 1, 2014);
        INSERT INTO singer_in_concert VALUES (1, 1);
        INSERT INTO pets VALUES (1, 'dog');
    """)
    c.commit()
    c.close()
    return p


def test_table_name_match_keeps_that_table_only(db):
    # "singers" matches singer; singer_in_concert needs both of its words.
    assert select_tables("How many singers are there?", db, min_tables=1, max_col_df=1) == ["singer"]


def test_column_name_match_keeps_the_table(db):
    assert select_tables("List every location.", db, min_tables=1) == ["stadium"]


def test_stored_value_match_keeps_the_table(db):
    assert select_tables("How many are a dog?", db, min_tables=1) == ["pets"]


def test_foreign_key_bridge_tables_are_added(db):
    kept = select_tables("Which singers performed in Glasgow?", db, min_tables=1)
    assert kept == ["concert", "singer", "singer_in_concert", "stadium"]


def test_too_few_matches_fall_back_to_the_full_schema(db):
    assert select_tables("What is the answer?", db, min_tables=1) is None
    assert select_tables("How many singers are there?", db, min_tables=2, max_col_df=1) is None


def test_pruned_schema_has_only_kept_tables_and_none_is_unchanged(db):
    assert serialize_schema(db, sample_rows=3, tables=None) == serialize_schema(db, sample_rows=3)
    pruned = serialize_schema(db, sample_rows=3, tables=["singer", "stadium"])
    assert "CREATE TABLE singer " in pruned and "CREATE TABLE stadium " in pruned
    assert "concert" not in pruned and "pets" not in pruned
    assert "-- Example rows from singer:" in pruned


def test_gold_tables_reads_from_and_join_case_insensitively():
    sql = ("SELECT T1.name FROM Singer AS T1 JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id "
           "WHERE T2.concert_id IN (SELECT concert_id FROM concert WHERE year = 2014)")
    assert gold_tables(sql) == {"singer", "singer_in_concert", "concert"}


def test_frozen_defaults_from_train_tuning():
    import inspect

    d = {k: v.default for k, v in inspect.signature(select_tables).parameters.items() if k not in ("question", "db_path")}
    assert d == {"min_tables": 2, "max_col_df": 3, "prefix": True, "neighbours": False}


def test_prefix_and_neighbour_options(db):
    assert select_tables("List the stadiums.", db, min_tables=1, max_col_df=1, prefix=False) == ["stadium"]
    # "concerts" keeps concert; one-hop FK neighbours add stadium and singer_in_concert.
    assert select_tables("List the concerts.", db, min_tables=1, max_col_df=1, neighbours=True) == [
        "concert", "singer_in_concert", "stadium"]
    # Prefix: "locat" (5 letters) is a prefix of the column word "location".
    assert select_tables("Any locat?", db, min_tables=1, max_col_df=1, prefix=True) == ["stadium"]
