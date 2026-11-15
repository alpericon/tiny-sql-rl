import pytest

from tinysql.compare import gold_is_ordered, results_match


def test_unordered_match_ignores_row_order():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)


def test_ordered_match_respects_row_order():
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert results_match([(1,), (2,)], [(1,), (2,)], ordered=True)


def test_duplicates_are_counted():
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)


def test_column_order_is_ignored_but_values_must_match():
    # Changed in week 2 (reviewer finding): Spider EX accepts permuted columns.
    assert results_match([(1, "a")], [("a", 1)], ordered=False)
    assert not results_match([(1, "b")], [("a", 1)], ordered=False)


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


# Column order is not part of the answer (official Spider / test-suite EX accept permutations).
def test_permuted_columns_match_unordered():
    gold = [("dog", 12.0), ("cat", 3.5)]
    pred = [(3.5, "cat"), (12.0, "dog")]
    assert results_match(pred, gold, ordered=False)


def test_permuted_columns_match_ordered():
    gold = [("a", 1), ("b", 2)]
    pred = [(1, "a"), (2, "b")]
    assert results_match(pred, gold, ordered=True)
    assert not results_match([(2, "b"), (1, "a")], gold, ordered=True)


def test_permutation_must_be_consistent_across_rows():
    # Each column alone has the same values, but no single column mapping reproduces the rows.
    gold = [(1, 2), (2, 1), (1, 1)]
    pred = [(1, 2), (2, 1), (2, 2)]
    assert not results_match(pred, gold, ordered=False)


def test_permutation_with_equal_columns():
    gold = [(1, 1, "x"), (2, 2, "y")]
    pred = [("x", 1, 1), ("y", 2, 2)]
    assert results_match(pred, gold, ordered=False)


def test_different_column_count_never_matches():
    assert not results_match([(1, 2)], [(1,)], ordered=False)


def test_wide_result_does_not_blow_up():
    # 12 identical columns: candidate mapping is ambiguous for every column; must stay fast.
    gold = [tuple([1] * 12)]
    pred = [tuple([1] * 11 + [2])]
    assert not results_match(pred, gold, ordered=False)
