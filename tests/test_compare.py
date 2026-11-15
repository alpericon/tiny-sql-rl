import pytest

from tinysql.compare import gold_is_ordered, results_match


def test_unordered_match_ignores_row_order():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)


def test_ordered_match_respects_row_order():
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert results_match([(1,), (2,)], [(1,), (2,)], ordered=True)


def test_duplicates_are_counted():
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)


def test_column_order_matters():
    assert not results_match([(1, "a")], [("a", 1)], ordered=False)


def test_extra_column_is_a_mismatch():
    assert not results_match([(1, "a")], [(1,)], ordered=False)


def test_empty_results_match():
    assert results_match([], [], ordered=False)


def test_int_and_float_with_same_value_match():
    # SQLite may return 3 or 3.0 for the same aggregate depending on the query form.
    assert results_match([(3,)], [(3.0,)], ordered=False)


def test_mixed_types_do_not_crash_sorting():
    assert results_match([(None,), (1,), ("x",)], [("x",), (None,), (1,)], ordered=False)


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT name FROM t ORDER BY age", True),
        ("select name from t order   by age desc limit 3", True),
        ("SELECT name FROM t", False),
        ("SELECT name FROM (SELECT * FROM t ORDER BY age) GROUP BY name", False),
        ("SELECT 'order by' FROM t", False),
    ],
)
def test_gold_is_ordered(sql, expected):
    assert gold_is_ordered(sql) is expected


def test_float_noise_is_tolerated():
    assert results_match([(0.30000000000000004,)], [(0.3,)], ordered=False)


def test_small_but_different_floats_do_not_match():
    assert not results_match([(1e-7,)], [(2e-7,)], ordered=False)
    assert not results_match([(1234567.5,)], [(1234567.0,)], ordered=False)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t -- order by a",
        "SELECT a FROM t /* order by a */",
        "SELECT [order by] FROM t",
        "SELECT `order by` FROM t",
    ],
)
def test_order_by_in_comments_or_identifiers_is_ignored(sql):
    assert gold_is_ordered(sql) is False
