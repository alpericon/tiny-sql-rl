import sqlite3

import pytest

from tinysql.data import Example
from tinysql.evaluate import evaluate
from tinysql.repair import REPAIR_ERROR, REPAIR_FORMAT, repair_messages


@pytest.fixture
def db_dir(tmp_path):
    d = tmp_path / "database" / "toy"
    d.mkdir(parents=True)
    conn = sqlite3.connect(d / "toy.sqlite")
    conn.executescript(
        "CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
        "INSERT INTO singer VALUES (1, 'Ada', 30), (2, 'Bo', 41), (3, 'Cy', 25);"
    )
    conn.commit()
    conn.close()
    return tmp_path / "database"


def test_repair_messages_continue_the_conversation():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    out = repair_messages(msgs, "```sql\nSELECT x\n```", "sql_error: no such column: x")
    assert out[:2] == msgs
    assert out[2] == {"role": "assistant", "content": "```sql\nSELECT x\n```"}
    assert out[3] == {"role": "user", "content": REPAIR_ERROR.format(error="sql_error: no such column: x")}
    assert repair_messages(msgs, "no sql here", None)[3]["content"] == REPAIR_FORMAT
    assert msgs == [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]  # not mutated


EXAMPLES = [
    Example("toy", "How many?", "SELECT count(*) FROM singer"),  # correct first time
    Example("toy", "Oldest?", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"),  # wrong but runs
    Example("toy", "Ages?", "SELECT age FROM singer"),  # exec error, repaired
    Example("toy", "Names?", "SELECT name FROM singer"),  # no block, repaired
    Example("toy", "Ids?", "SELECT id FROM singer"),  # exec error, repair fails again
]
FIRST = {
    "How many?": "```sql\nSELECT COUNT(id) FROM singer\n```",
    "Oldest?": "```sql\nSELECT name FROM singer ORDER BY age LIMIT 1\n```",
    "Ages?": "```sql\nSELECT agee FROM singer\n```",
    "Names?": "SELECT name FROM singer",
    "Ids?": "```sql\nSELECT idd FROM singer\n```",
}
SECOND = {
    "Ages?": "```sql\nSELECT age FROM singer\n```",
    "Names?": "```sql\nSELECT name FROM singer\n```",
    "Ids?": "```sql\nSELECT iddd FROM singer\n```",
}


def make_generate(calls):
    def generate(batch):
        calls.append(batch)
        out = []
        for m in batch:
            q = next(x["content"] for x in m if x["role"] == "user").rsplit("Question: ", 1)[1]
            out.append(SECOND[q] if m[-1]["role"] == "user" and len(m) > 2 and m[-2]["role"] == "assistant" else FIRST[q])
        return out
    return generate


def test_evaluate_with_repair(db_dir):
    calls = []
    records, summary = evaluate(EXAMPLES, db_dir, make_generate(calls), batch_size=5, repair=True)
    assert [r["match"] for r in records] == [True, False, True, True, False]
    assert [r["repaired"] for r in records] == [False, False, True, True, True]
    assert [r["first_match"] for r in records] == [True, False, False, False, False]
    assert records[2]["first_pred_sql"] == "SELECT agee FROM singer" and "agee" in records[2]["first_error"]
    assert records[3]["first_pred_sql"] is None and records[3]["first_error"] == "no ```sql block"
    assert records[4]["exec_ok"] is False  # repaired answer is final even if it fails again
    assert len(calls) == 2 and len(calls[1]) == 3  # one retry round, only the three failures
    assert summary["execution_accuracy"] == pytest.approx(3 / 5)
    assert summary["first_attempt_execution_accuracy"] == pytest.approx(1 / 5)
    assert summary["repair_attempts"] == 3 and summary["repair_rescued"] == 2


def test_evaluate_without_repair_is_unchanged(db_dir):
    calls = []
    records, summary = evaluate(EXAMPLES, db_dir, make_generate(calls), batch_size=5)
    assert len(calls) == 1
    assert [r["match"] for r in records] == [True, False, False, False, False]
    assert "repaired" not in records[0] and "repair_attempts" not in summary
