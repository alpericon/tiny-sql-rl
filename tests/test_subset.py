import pytest

from tinysql.data import Example
from tinysql.subset import load_subset, select_subset

DEV = [Example(f"db{i % 5}", f"q{i}", f"SELECT {i}") for i in range(50)]


def test_selection_is_deterministic_sorted_and_sized():
    a = select_subset(DEV, eligible=range(50), n=10, seed=3407)
    b = select_subset(DEV, eligible=range(50), n=10, seed=3407)
    assert a == b
    assert [r["idx"] for r in a] == sorted(r["idx"] for r in a)
    assert len({r["idx"] for r in a}) == 10


def test_selection_respects_eligibility():
    ids = select_subset(DEV, eligible=range(0, 50, 2), n=10, seed=1)
    assert all(r["idx"] % 2 == 0 for r in ids)


def test_selection_records_content_for_verification():
    r = select_subset(DEV, eligible=range(50), n=1, seed=0)[0]
    assert r == {"idx": r["idx"], "db_id": DEV[r["idx"]].db_id, "question": DEV[r["idx"]].question}


def test_too_few_eligible_raises():
    with pytest.raises(ValueError):
        select_subset(DEV, eligible=range(5), n=10, seed=0)


def test_load_subset_returns_examples_in_order():
    ids = select_subset(DEV, eligible=range(50), n=5, seed=7)
    assert load_subset(DEV, ids) == [DEV[r["idx"]] for r in ids]


def test_load_subset_detects_reordered_dev():
    ids = select_subset(DEV, eligible=range(50), n=5, seed=7)
    with pytest.raises(ValueError):
        load_subset(list(reversed(DEV)), ids)
