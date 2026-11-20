"""PLAN-week10 P4: pruned first attempt, full-schema repair, in evaluate() and the demo."""

import sqlite3

import pytest

from tinysql.data import Example
from tinysql.evaluate import evaluate

Q = "Which singers have concerts?"
BAD = "```sql\nSELECT agee FROM singer\n```"
GOOD = "```sql\nSELECT DISTINCT s.name FROM singer s JOIN concert c ON c.singer_id = s.id\n```"


@pytest.fixture
def db_dir(tmp_path):
    d = tmp_path / "database" / "toy"
    d.mkdir(parents=True)
    conn = sqlite3.connect(d / "toy.sqlite")
    conn.executescript("""
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE concert (id INTEGER PRIMARY KEY, singer_id INTEGER, FOREIGN KEY (singer_id) REFERENCES singer(id));
        CREATE TABLE pets (pet_id INTEGER PRIMARY KEY, pet_type TEXT);
        INSERT INTO singer VALUES (1, 'Ada'), (2, 'Bo');
        INSERT INTO concert VALUES (1, 1);
        INSERT INTO pets VALUES (1, 'dog');
    """)
    conn.commit()
    conn.close()
    return tmp_path / "database"


def recorder(answers):
    seen = []

    def gen(batch):
        seen.extend(m[-1]["content"] if m[-1]["role"] == "user" else m[1]["content"] for m in batch)
        return [answers[min(len(seen), len(answers)) - 1] for _ in batch]

    return gen, seen


def schema_part(user_content):
    return user_content.split("Question:")[0]


def test_pruned_first_attempt_and_full_schema_repair(db_dir):
    gen, seen = recorder([BAD, GOOD])
    ex = [Example("toy", Q, "SELECT DISTINCT s.name FROM singer s JOIN concert c ON c.singer_id = s.id")]
    records, summary = evaluate(ex, db_dir, gen, repair=True, prune=True)
    r = records[0]
    assert r["pruned_tables"] == ["concert", "singer"]
    assert r["repaired"] and r["match"]
    # first attempt: pets is pruned away; the repair turn shows the full schema again.
    assert "pets" not in schema_part(seen[0]) and "CREATE TABLE singer" in seen[0]
    assert summary["pruned_queries"] == 1


def test_repair_prompt_is_built_on_the_full_schema(db_dir):
    sent = []

    def gen(batch):
        sent.extend(batch)
        return [BAD] * len(batch)

    evaluate([Example("toy", Q, "SELECT 1")], db_dir, gen, repair=True, prune=True)
    first, retry = sent
    assert "pets" not in first[1]["content"]
    assert "CREATE TABLE pets" in retry[1]["content"]  # the retry conversation starts from the full schema


def test_prune_off_is_unchanged(db_dir):
    ex = [Example("toy", Q, "SELECT 1")]
    a, _ = evaluate(ex, db_dir, lambda b: [GOOD] * len(b), repair=True)
    b, s = evaluate(ex, db_dir, lambda b: [GOOD] * len(b), repair=True, prune=False)
    assert a == b and "pruned_tables" not in a[0] and "pruned_queries" not in s


def test_demo_prunes_first_attempt_and_repairs_on_full_schema(db_dir):
    from tinysql.demo import answer_question

    sent = []

    def gen(batch):
        sent.extend(batch)
        return [BAD if len(sent) == 1 else GOOD]

    res = answer_question(Q, db_dir / "toy" / "toy.sqlite", gen, prune=True)
    assert res.repaired and res.error is None
    assert "pets" not in sent[0][1]["content"] and "CREATE TABLE pets" in sent[1][1]["content"]
    sent.clear()
    answer_question(Q, db_dir / "toy" / "toy.sqlite", gen)
    assert "CREATE TABLE pets" in sent[0][1]["content"]
