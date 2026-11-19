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


def test_sample_rows_reach_training_prompts_like_eval(db_dir):
    # PLAN-week8 R1: training with example rows must see exactly the inference prompt.
    records = build_grpo_records(EXAMPLES, db_dir, Exclusions(), sample_rows=3)
    for r, e in zip(records, EXAMPLES):
        db_path = db_dir / e.db_id / f"{e.db_id}.sqlite"
        assert r["prompt"] == build_messages(serialize_schema(db_path, sample_rows=3), e.question)
        assert "-- Example rows from t:" in r["prompt"][1]["content"]


def test_sample_rows_zero_reproduces_schema_only_records(db_dir):
    default = build_grpo_records(EXAMPLES, db_dir, Exclusions())
    assert build_grpo_records(EXAMPLES, db_dir, Exclusions(), sample_rows=0) == default
    assert not any("Example rows" in r["prompt"][1]["content"] for r in default)


def test_sample_rows_keep_the_same_question_set(db_dir):
    # The pass-rate filter is keyed by (db_id, question), so it selects the same questions either way.
    key = lambda rs: [(r["db_id"], r["question"], r["gold"], r["db_path"]) for r in rs]
    assert key(build_grpo_records(EXAMPLES, db_dir, Exclusions(), sample_rows=3, drop_empty_gold=True)) == key(
        build_grpo_records(EXAMPLES, db_dir, Exclusions(), drop_empty_gold=True)
    )


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


def _rec(i, kind):
    return {"prompt": [{"role": "user", "content": f"{kind}{i}"}], "db_path": "p", "gold": "g",
            "db_id": "d", "question": f"q{i}", "kind": kind, "first_error": "x"}


def test_mix_records_hits_the_fraction_and_keeps_common_columns():
    from tinysql.grpo_data import mix_records

    first = [{k: v for k, v in _rec(i, "first").items() if k != "first_error"} for i in range(10)]
    repair = [_rec(i, "repair") for i in range(25)]
    mixed = mix_records(first, repair, repair_frac=0.5, seed=0)
    kinds = [r["kind"] for r in mixed]
    assert kinds.count("first") == 10 and kinds.count("repair") == 10
    assert all(set(r) == {"prompt", "db_path", "gold", "db_id", "question", "kind"} for r in mixed)
    assert mixed == mix_records(first, repair, repair_frac=0.5, seed=0)  # deterministic
    assert kinds != sorted(kinds)  # shuffled, not blocked


def test_mix_records_downsamples_first_turn_when_repair_is_scarce():
    from tinysql.grpo_data import mix_records

    first = [{k: v for k, v in _rec(i, "first").items() if k != "first_error"} for i in range(10)]
    mixed = mix_records(first, [_rec(i, "repair") for i in range(3)], repair_frac=0.5, seed=0)
    kinds = [r["kind"] for r in mixed]
    assert kinds.count("repair") == 3 and kinds.count("first") == 3


def test_kind_reward_logger_only_scores_its_kind():
    from tinysql.grpo_data import kind_reward_logger

    base = lambda prompts, completions, **kw: [1.0, 0.1, 0.0]  # noqa: E731
    log_repair = kind_reward_logger(base, "repair")
    out = log_repair([None] * 3, ["a", "b", "c"], kind=["first", "repair", "repair"])
    assert out == [None, 0.1, 0.0]
    assert log_repair.__name__ == "reward_repair"
