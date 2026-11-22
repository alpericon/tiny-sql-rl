"""PLAN-week12 B2: BIRD loader, evidence prompt and BIRD's execution accuracy."""

import json
import sqlite3

import pytest

from tinysql.bird import bird_ex, bird_question, load_bird
from tinysql.data import Example


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE t (a INTEGER, b TEXT);
        INSERT INTO t VALUES (1, 'x'), (1, 'x'), (2, 'y');
    """)
    c.commit()
    c.close()
    return p


def test_question_carries_the_evidence_line():
    assert bird_question("How many?", "count = rows") == "How many?\nEvidence: count = rows"
    assert bird_question("How many?", "") == "How many?"
    assert bird_question("How many?", None) == "How many?"


def test_load_bird_keeps_order_and_metadata(tmp_path):
    rows = [
        {"question_id": 0, "db_id": "d1", "question": "Q0?", "evidence": "E0", "SQL": "SELECT 1", "difficulty": "simple"},
        {"question_id": 1, "db_id": "d2", "question": "Q1?", "evidence": "", "SQL": "SELECT 2", "difficulty": "challenging"},
    ]
    (tmp_path / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
    examples, meta = load_bird(tmp_path / "dev.json")
    assert examples == [Example("d1", "Q0?\nEvidence: E0", "SELECT 1"), Example("d2", "Q1?", "SELECT 2")]
    assert meta == [{"question_id": 0, "difficulty": "simple"}, {"question_id": 1, "difficulty": "challenging"}]


def test_bird_ex_is_set_equality_of_rows(db):
    gold = "SELECT a, b FROM t"
    assert bird_ex(db, gold, "SELECT a, b FROM t ORDER BY a DESC")["match"]  # row order ignored
    assert bird_ex(db, gold, "SELECT DISTINCT a, b FROM t")["match"]  # duplicates collapsed
    assert not bird_ex(db, gold, "SELECT b, a FROM t")["match"]  # column order matters (tuples)
    assert not bird_ex(db, gold, "SELECT a FROM t")["match"]
    assert bird_ex(db, "SELECT 1", "SELECT 1.0")["match"]  # Python equality: 1 == 1.0


def test_bird_ex_errors_and_timeouts_score_wrong(db):
    bad = bird_ex(db, "SELECT a FROM t", "SELECT nope FROM t")
    assert not bad["exec_ok"] and not bad["match"] and bad["error"]
    slow = ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r")
    out = bird_ex(db, "SELECT a FROM t", slow, timeout_s=0.5)
    assert not out["match"] and out["error"] == "timeout"


def test_evaluate_accepts_a_scorer(db):
    from tinysql.evaluate import evaluate

    db_dir = db.parent.parent
    (db.parent.parent / "toy").mkdir(exist_ok=True)
    target = db_dir / "toy" / "toy.sqlite"
    target.write_bytes(db.read_bytes())
    ex = [Example("toy", "Distinct rows?", "SELECT a, b FROM t")]
    answer = lambda batch: ["```sql\nSELECT DISTINCT a, b FROM t\n```"] * len(batch)
    ours, _ = evaluate(ex, db_dir, answer)  # our comparator is multiset-based: duplicates matter
    bird, _ = evaluate(ex, db_dir, answer, scorer=bird_ex)
    assert not ours[0]["match"] and bird[0]["match"]
