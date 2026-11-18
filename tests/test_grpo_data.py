import sqlite3

import pytest

from tinysql.data import Example, Exclusions, _norm
from tinysql.grpo_data import build_grpo_records
from tinysql.prompt import build_messages, serialize_schema


@pytest.fixture
def db_dir(tmp_path):
    for db_id in ("toy", "zoo"):
        d = tmp_path / db_id
        d.mkdir()
        conn = sqlite3.connect(d / f"{db_id}.sqlite")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'a')")
        conn.commit()
        conn.close()
    return tmp_path


EXAMPLES = [
    Example("toy", "How many rows?", "SELECT count(*) FROM t"),
    Example("zoo", "Names?", "SELECT name FROM t"),
]


def test_record_fields_and_prompt_match_eval(db_dir):
    records = build_grpo_records(EXAMPLES, db_dir, Exclusions())
    assert len(records) == 2
    r = records[0]
    db_path = db_dir / "toy" / "toy.sqlite"
    assert r["prompt"] == build_messages(serialize_schema(db_path), "How many rows?")
    assert r["db_path"] == str(db_path)
    assert r["gold"] == "SELECT count(*) FROM t"
    assert r["db_id"] == "toy"


def test_leaked_example_is_refused(db_dir):
    leaked = Exclusions(questions={("zoo", _norm("Names?"))})
    with pytest.raises(AssertionError):
        build_grpo_records(EXAMPLES, db_dir, leaked)


def test_missing_database_raises(db_dir):
    with pytest.raises(FileNotFoundError):
        build_grpo_records([Example("nope", "q", "SELECT 1")], db_dir, Exclusions())


def test_records_are_plain_json_types(db_dir):
    import json

    records = build_grpo_records(EXAMPLES, db_dir, Exclusions())
    assert json.loads(json.dumps(records)) == records


def test_empty_gold_results_can_be_dropped(db_dir):
    # Any query that returns nothing scores 1.0 against an empty gold answer (W7 reward hack).
    examples = EXAMPLES + [Example("toy", "Nobody?", "SELECT name FROM t WHERE id = 99")]
    kept = build_grpo_records(examples, db_dir, Exclusions(), drop_empty_gold=True)
    assert [r["gold"] for r in kept] == [e.query for e in EXAMPLES]
    assert len(build_grpo_records(examples, db_dir, Exclusions())) == 3  # default unchanged


def test_records_carry_question_for_pass_rate_keys(db_dir):
    records = build_grpo_records(EXAMPLES, db_dir, Exclusions())
    assert [r["question"] for r in records] == ["How many rows?", "Names?"]


def test_filter_by_pass_rate_keeps_only_mixed_groups():
    from tinysql.grpo_data import filter_by_pass_rate

    records = [{"db_id": "a", "question": q, "gold": "G"} for q in ("never", "some", "always")]
    rates = [
        {"db_id": "a", "question": "never", "gold": "G", "n_correct": 0, "n": 8},
        {"db_id": "a", "question": "some", "gold": "G", "n_correct": 3, "n": 8},
        {"db_id": "a", "question": "always", "gold": "G", "n_correct": 8, "n": 8},
    ]
    assert [r["question"] for r in filter_by_pass_rate(records, rates)] == ["some"]


def test_filter_by_pass_rate_refuses_missing_rates():
    from tinysql.grpo_data import filter_by_pass_rate

    with pytest.raises(KeyError):
        filter_by_pass_rate([{"db_id": "a", "question": "q", "gold": "G"}], [])


def _sample(db_dir, completion, question="How many rows?", db_id="toy", gold="SELECT count(*) FROM t"):
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    return {"db_id": db_id, "question": question, "gold": gold, "db_path": str(db_path),
            "prompt": build_messages(serialize_schema(db_path), question), "completion": completion}


def test_repair_records_only_from_failures_with_inference_prompt(db_dir):
    from tinysql.grpo_data import build_repair_records
    from tinysql.repair import repair_messages

    samples = [
        _sample(db_dir, "```sql\nSELECT count(*) FROM t\n```"),          # runs: skipped
        _sample(db_dir, "```sql\nSELECT name FROM t\n```"),              # runs but wrong: skipped
        _sample(db_dir, "```sql\nSELECT nope FROM t\n```"),              # exec error: kept
        _sample(db_dir, "no block here"),                                 # no block: kept
    ]
    recs = build_repair_records(samples)
    assert len(recs) == 2
    err = recs[0]
    assert "no such column" in err["first_error"]
    assert err["prompt"] == repair_messages(samples[2]["prompt"], samples[2]["completion"], err["first_error"])
    assert recs[1]["prompt"] == repair_messages(samples[3]["prompt"], samples[3]["completion"], None)
    assert recs[1]["first_error"] == "no ```sql block"
    for r in recs:
        assert r["kind"] == "repair" and r["gold"] == "SELECT count(*) FROM t" and r["db_id"] == "toy"
        assert r["db_path"] == samples[0]["db_path"] and r["question"] == "How many rows?"


def test_repair_records_dedup_and_cap_per_question(db_dir):
    from tinysql.grpo_data import build_repair_records

    samples = [_sample(db_dir, f"```sql\nSELECT nope{i} FROM t\n```") for i in range(4)]
    samples.append(_sample(db_dir, "```sql\nSELECT nope0 FROM t\n```"))  # duplicate failed SQL
    samples.append(_sample(db_dir, "```sql\nSELECT bad FROM t\n```", question="Names?", db_id="zoo",
                           gold="SELECT name FROM t"))
    recs = build_repair_records(samples, max_per_question=2)
    per_q = {}
    for r in recs:
        per_q[r["question"]] = per_q.get(r["question"], 0) + 1
    assert per_q == {"How many rows?": 2, "Names?": 1}
    first_sqls = [r["prompt"][-2]["content"] for r in recs if r["question"] == "How many rows?"]
    assert len(set(first_sqls)) == 2
