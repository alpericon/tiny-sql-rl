"""PLAN-week9 E1: the error-analysis sample (stratified, reproducible, wrong on both metrics only)."""

from collections import Counter

import pytest

from tinysql.error_sample import allocate, stratified_sample, wrong_on_both


def _rec(match=False, fmt=True, ex=True):
    return {"match": match, "format_ok": fmt, "exec_ok": ex}


def test_wrong_on_both_keeps_only_running_wrong_answers_the_official_metric_also_rejects():
    recs = [_rec(), _rec(match=True), _rec(ex=False), _rec(fmt=False), _rec()]
    off = [{"idx": i, "official": o, "hardness": "easy"} for i, o in enumerate([False, False, False, False, True])]
    assert wrong_on_both(recs, off) == [0]


def test_wrong_on_both_refuses_misaligned_rows():
    with pytest.raises(ValueError):
        wrong_on_both([_rec()], [{"idx": 1, "official": False, "hardness": "easy"}])


def test_allocate_is_proportional_by_largest_remainder():
    # The plan's numbers: medium 117, extra 86, hard 70, easy 28 -> 39 / 29 / 23 / 9.
    assert allocate({"medium": 117, "extra": 86, "hard": 70, "easy": 28}, 100) == {
        "medium": 39, "extra": 29, "hard": 23, "easy": 9}


def test_stratified_sample_is_reproducible_and_stratified():
    strata = {i: ("a" if i < 60 else "b") for i in range(100)}
    s1 = stratified_sample(strata, 10, seed=3407)
    assert s1 == stratified_sample(strata, 10, seed=3407)
    assert s1 != stratified_sample(strata, 10, seed=1)
    assert Counter(strata[i] for i in s1) == {"a": 6, "b": 4}
    assert s1 == sorted(set(s1))
