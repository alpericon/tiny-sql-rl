import pytest

from tinysql.stats import mcnemar_exact, paired_summary


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(11, 3) == pytest.approx(2 * 470 / 2**14)  # sum C(14, 0..3) = 470
    assert mcnemar_exact(5, 5) == 1.0  # capped at 1
    assert mcnemar_exact(3, 11) == mcnemar_exact(11, 3)


def test_paired_summary_counts_and_breakdown():
    base = [
        {"match": True, "format_ok": True, "exec_ok": True},
        {"match": False, "format_ok": False, "exec_ok": False},
        {"match": False, "format_ok": True, "exec_ok": False},
        {"match": True, "format_ok": True, "exec_ok": True},
    ]
    cand = [
        {"match": True, "format_ok": True, "exec_ok": True},
        {"match": True, "format_ok": True, "exec_ok": True},
        {"match": False, "format_ok": True, "exec_ok": True},
        {"match": False, "format_ok": True, "exec_ok": True},
    ]
    s = paired_summary(base, cand)
    assert (s["n"], s["base_ex"], s["cand_ex"]) == (4, 0.5, 0.5)
    assert (s["wins"], s["losses"]) == (1, 1)
    assert s["wins_by_base_outcome"] == {"format_fail": 1, "exec_error": 0, "wrong_result": 0}
    assert s["cand_breakdown"] == {"format_fail": 0, "exec_error": 0, "wrong_result": 2}
    assert s["p"] == 1.0


def test_paired_summary_rejects_unaligned():
    with pytest.raises(ValueError):
        paired_summary([{"match": True}], [])


def test_paired_summary_rejects_misordered_questions():
    a = {"match": True, "format_ok": True, "exec_ok": True, "db_id": "d", "question": "q1", "gold": "G"}
    b = dict(a, question="q2")
    with pytest.raises(ValueError):
        paired_summary([a, b], [b, a])
