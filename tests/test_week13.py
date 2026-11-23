"""PLAN-week13 K1: record names for another model, BIRD paired comparisons, EX where both have a block."""

import pytest

from tinysql.bird import bird_pairs
from tinysql.official import record_stem
from tinysql.stats import ex_where_both_formatted


def test_record_stem_defaults_to_qwen3_and_accepts_another_model():
    assert record_stem("") == "Qwen__Qwen3-1.7B"
    assert record_stem("_v2-checkpoint-900") == "Qwen__Qwen3-1.7B_v2-checkpoint-900"
    coder = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert record_stem((coder, "")) == "Qwen__Qwen2.5-Coder-1.5B-Instruct"
    assert record_stem((coder, "_repair_ground_rows3_prune")) == "Qwen__Qwen2.5-Coder-1.5B-Instruct_repair_ground_rows3_prune"


def r(qid, db, match, fmt=True, ok=True):
    return {"question_id": qid, "db_id": db, "question": f"q{qid}", "gold": f"g{qid}",
            "match": match, "format_ok": fmt, "exec_ok": ok}


A = [r(0, "x", True), r(1, "x", False), r(2, "formula_1", True), r(3, "formula_1", False), r(4, "y", False)]
B = [r(0, "x", False), r(1, "x", False), r(2, "formula_1", False), r(3, "formula_1", True), r(4, "y", True)]


def test_bird_pairs_overall_and_without_a_database():
    out = bird_pairs(A, B, exclude_db="formula_1")
    assert (out["all"]["n"], out["all"]["wins"], out["all"]["losses"]) == (5, 2, 2)  # B over A
    assert (out["without_formula_1"]["n"], out["without_formula_1"]["wins"], out["without_formula_1"]["losses"]) == (3, 1, 1)


def test_bird_pairs_refuses_a_different_question_order():
    with pytest.raises(ValueError):
        bird_pairs(A, list(reversed(B)), exclude_db="formula_1")
    shifted = [dict(b, question_id=b["question_id"] + 100) for b in B]
    with pytest.raises(ValueError):
        bird_pairs(A, shifted, exclude_db="formula_1")


def test_ex_where_both_formatted():
    fmt_a, match_a = [True, True, False, True], [True, False, False, True]
    fmt_b, match_b = [True, False, True, True], [False, False, True, True]
    out = ex_where_both_formatted(fmt_a, match_a, fmt_b, match_b)
    assert out == {"n": 2, "a_ex": 1.0, "b_ex": 0.5}  # rows 0 and 3 only
    with pytest.raises(ValueError):
        ex_where_both_formatted(fmt_a, match_a, fmt_b[:3], match_b[:3])
