import sqlite3

import pytest

from tinysql.prompt import PromptTooLong, build_messages, check_fits, serialize_schema


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "concert.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE stadium (stadium_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, capacity INTEGER);
        CREATE TABLE concert (
            concert_id INTEGER PRIMARY KEY,
            stadium_id INTEGER,
            year TEXT,
            FOREIGN KEY (stadium_id) REFERENCES stadium(stadium_id)
        );
        CREATE INDEX idx_year ON concert(year);
        INSERT INTO stadium (name, capacity) VALUES ('Recreation Park', 3100), ('Somerset Park', 11998),
            ('Glebe Park', 4000), ('Balmoor', 4000);
        INSERT INTO concert VALUES (1, 1, '2014');
        """
    )
    conn.commit()
    conn.close()
    return path


def test_schema_contains_tables_and_foreign_keys(db):
    schema = serialize_schema(db)
    assert "CREATE TABLE stadium" in schema
    assert "CREATE TABLE concert" in schema
    assert "REFERENCES stadium(stadium_id)" in schema


def test_schema_skips_internal_tables_and_indexes(db):
    schema = serialize_schema(db)
    assert "sqlite_sequence" not in schema
    assert "idx_year" not in schema


def test_schema_tables_in_stable_order(db):
    schema = serialize_schema(db)
    assert schema.index("CREATE TABLE concert") < schema.index("CREATE TABLE stadium")


def test_sample_rows_are_capped(db):
    schema = serialize_schema(db, sample_rows=3)
    assert "Recreation Park" in schema
    assert "Glebe Park" in schema
    assert "Balmoor" not in schema  # 4th row is beyond the cap


def test_no_sample_rows_by_default(db):
    assert "Recreation Park" not in serialize_schema(db)


def test_messages_contain_schema_question_and_format_instruction(db):
    msgs = build_messages(serialize_schema(db), "How many stadiums are there?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    text = msgs[0]["content"] + msgs[1]["content"]
    assert "CREATE TABLE stadium" in text
    assert "How many stadiums are there?" in text
    assert "```sql" in text


def test_check_fits_raises_instead_of_truncating():
    msgs = [{"role": "user", "content": "x" * 100}]
    check_fits(msgs, max_tokens=200, count_tokens=len)
    with pytest.raises(PromptTooLong):
        check_fits(msgs, max_tokens=50, count_tokens=len)
