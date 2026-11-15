import sqlite3

import pytest

from tinysql.grpo_reward import make_reward_fn


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Ada', 30), (2, 'Bo', 41);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


GOLD = "SELECT name FROM singer WHERE age > 35"
RIGHT = "```sql\nSELECT name FROM singer WHERE age >= 41\n```"
WRONG = "```sql\nSELECT name FROM singer\n```"
ERROR = "```sql\nSELECT nme FROM singer\n```"
NOFENCE = "SELECT name FROM singer WHERE age > 35"


def chat(text):
    return [{"role": "assistant", "content": text}]


def test_plain_string_completions(db):
    fn = make_reward_fn()
    rewards = fn(prompts=["p"] * 4, completions=[RIGHT, WRONG, ERROR, NOFENCE], db_path=[db] * 4, gold=[GOLD] * 4)
    assert rewards == [1.0, 0.1, 0.0, 0.0]


def test_conversational_completions(db):
    fn = make_reward_fn()
    rewards = fn(prompts=["p"] * 2, completions=[chat(RIGHT), chat(WRONG)], db_path=[db] * 2, gold=[GOLD] * 2)
    assert rewards == [1.0, 0.1]


def test_binary_mode(db):
    fn = make_reward_fn(binary=True)
    assert fn(prompts=["p"] * 2, completions=[RIGHT, WRONG], db_path=[db] * 2, gold=[GOLD] * 2) == [1.0, 0.0]


def test_rewards_stay_aligned_with_their_own_gold(db):
    # Mixed batch: each completion is scored against its own row's gold, not the first one.
    fn = make_reward_fn()
    golds = [GOLD, "SELECT count(*) FROM singer"]
    comps = ["```sql\nSELECT 2\n```", "```sql\nSELECT 2\n```"]
    assert fn(prompts=["p"] * 2, completions=comps, db_path=[db] * 2, gold=golds) == [0.1, 1.0]


def test_extra_trl_kwargs_are_ignored(db):
    fn = make_reward_fn()
    out = fn(prompts=["p"], completions=[RIGHT], db_path=[db], gold=[GOLD], completion_ids=[[1, 2]], trainer_state=None, db_id=["toy"])
    assert out == [1.0]


def test_hostile_completion_never_raises(db):
    fn = make_reward_fn()
    nasty = [
        "```sql\nDROP TABLE singer\n```",
        "```sql\nSELECT 1; DELETE FROM singer\n```",
        "```sql\nSELECT '\x00'\n```",
        "```sql\nWITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT x FROM c\n```",
        "",
        chat(None),
    ]
    rewards = fn(prompts=["p"] * len(nasty), completions=nasty, db_path=[db] * len(nasty), gold=[GOLD] * len(nasty))
    assert all(r in (0.0, 0.1) for r in rewards)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM singer").fetchone()[0] == 2


def test_broken_gold_raises_loudly(db):
    fn = make_reward_fn()
    with pytest.raises(ValueError):
        fn(prompts=["p"], completions=[RIGHT], db_path=[db], gold=["SELECT missing FROM singer"])


def test_length_mismatch_raises(db):
    fn = make_reward_fn()
    with pytest.raises(ValueError):
        fn(prompts=["p"] * 2, completions=[RIGHT, RIGHT], db_path=[db], gold=[GOLD, GOLD])


def test_function_has_a_name_for_trl_logging():
    assert make_reward_fn().__name__ == "sql_exec_reward"
    assert make_reward_fn(binary=True).__name__ == "sql_exec_reward_binary"
