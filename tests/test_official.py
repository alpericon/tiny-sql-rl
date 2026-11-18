import pytest

from tinysql.data import Example
from tinysql.official import FORMAT_FAILURE_SQL, eval_lines

DEV = [
    Example("db_a", "How many?", "SELECT count(*)\nFROM t"),
    Example("db_b", "Names?", "SELECT name FROM s"),
]


def rec(e, pred):
    return {"db_id": e.db_id, "question": e.question, "gold": e.query, "pred_sql": pred}


def test_lines_are_one_per_example_in_order():
    gold, pred = eval_lines([rec(DEV[0], "SELECT\n  count(*) FROM t;"), rec(DEV[1], "SELECT name FROM s")], DEV)
    assert gold == ["SELECT count(*) FROM t\tdb_a", "SELECT name FROM s\tdb_b"]
    assert pred == ["SELECT count(*) FROM t;", "SELECT name FROM s"]


def test_format_failure_becomes_placeholder_not_dropped():
    _, pred = eval_lines([rec(DEV[0], None), rec(DEV[1], "  ")], DEV)
    assert pred == [FORMAT_FAILURE_SQL, FORMAT_FAILURE_SQL]


def test_tabs_inside_sql_cannot_break_the_columns():
    gold, pred = eval_lines([rec(DEV[0], "SELECT\tcount(*)\tFROM t"), rec(DEV[1], "SELECT name FROM s")], DEV)
    assert "\t" not in pred[0]
    assert gold[0].count("\t") == 1


def test_misaligned_records_raise():
    with pytest.raises(ValueError):
        eval_lines([rec(DEV[1], "x"), rec(DEV[0], "y")], DEV)
    with pytest.raises(ValueError):
        eval_lines([rec(DEV[0], "x")], DEV)
