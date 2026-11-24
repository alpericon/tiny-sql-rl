"""PLAN-week14 M0: the seed runs repeat the v2 recipe, the selection rule, and the record names."""

from pathlib import Path

import pytest

from tinysql.seeds import adapter_suffix, check_same_recipe, seed_entries, select_checkpoint

V2 = {"model": "Qwen/Qwen3-1.7B", "run_name": "v2", "max_steps": 1000, "lr": 5e-06, "seed": 3407,
      "pass_rate": "data/train_pass_rate.jsonl"}
# Arguments added to train_grpo.py after v2 ran, at their "off" values.
LATER_ARGS = {"init_adapter": None, "repair_data": None, "repair_frac": 0.5, "sample_rows": 0}


def test_adapter_suffix_matches_the_eval_output_name():
    assert adapter_suffix(Path("checkpoints/v2/checkpoint-900")) == "_v2-checkpoint-900"
    assert adapter_suffix(Path("checkpoints/v2-s1/checkpoint-700")) == "_v2-s1-checkpoint-700"


def test_same_recipe_allows_only_run_name_and_seed():
    check_same_recipe(V2, {**V2, "run_name": "v2-s1", "seed": 1, **LATER_ARGS})


def test_same_recipe_refuses_another_difference():
    with pytest.raises(ValueError, match="lr"):
        check_same_recipe(V2, {**V2, "run_name": "v2-s1", "seed": 1, "lr": 1e-05, **LATER_ARGS})


def test_same_recipe_refuses_a_later_argument_switched_on():
    with pytest.raises(ValueError, match="sample_rows"):
        check_same_recipe(V2, {**V2, "run_name": "v2-s1", "seed": 1, **LATER_ARGS, "sample_rows": 3})


def test_same_recipe_refuses_an_unknown_missing_argument():
    with pytest.raises(ValueError, match="pass_rate"):
        check_same_recipe(V2, {k: v for k, v in V2.items() if k != "pass_rate"} | {"run_name": "v2-s1", "seed": 1})


def test_select_checkpoint_best_dev200_ties_to_the_later_step():
    # v2's own curve: 400, 500 and 900 tie at 60.0, so the rule picks 900.
    v2 = {100: .585, 200: .585, 300: .595, 400: .6, 500: .6, 600: .595, 700: .59, 800: .585, 900: .6, 1000: .585}
    assert select_checkpoint(v2) == 900
    assert select_checkpoint({100: .5, 200: .62, 300: .61}) == 200


def test_seed_entries_names_and_pairs():
    models, pairs = seed_entries({"v2-s1": 700, "v2-s2": 900})
    assert models == {"B-v2-s1": "_v2-s1-checkpoint-700", "B-v2-s1-900": "_v2-s1-checkpoint-900",
                      "B-v2-s2": "_v2-s2-checkpoint-900"}
    assert ("B-v2-s1", "A-base") in pairs and ("B-v2-s2", "A-base") in pairs  # S1, S2
    assert ("B-v2-s1", "B-v2-900") in pairs and ("B-v2-s2", "B-v2-s1") in pairs  # descriptive
    assert ("B-v2-s1-900", "A-base") in pairs


def test_seed_entries_with_one_run_selected_so_far():
    models, pairs = seed_entries({"v2-s1": 900})
    assert models == {"B-v2-s1": "_v2-s1-checkpoint-900"}
    assert pairs == [("B-v2-s1", "A-base"), ("B-v2-s1", "B-v2-900")]


def test_same_recipe_ignores_resume_and_the_guard_itself():
    # what train_grpo writes to checkpoints/<run>/run_args.json for a resumed, guarded seed run
    written = {**V2, "run_name": "v2-s1", "seed": 1, **LATER_ARGS, "resume": True,
               "same_recipe_as": "checkpoints/v2/run_args.json"}
    check_same_recipe({**V2, "resume": False}, written)


def test_same_recipe_against_the_real_v2_run_args():
    import json

    v2 = json.loads((Path(__file__).resolve().parents[1] / "checkpoints" / "v2" / "run_args.json").read_text(encoding="utf-8"))
    check_same_recipe(v2, {**v2, "run_name": "v2-s2", "seed": 2, **LATER_ARGS,
                           "same_recipe_as": "checkpoints/v2/run_args.json"})
    with pytest.raises(ValueError, match="beta"):
        check_same_recipe(v2, {**v2, "run_name": "v2-s2", "seed": 2, "beta": 0.0})


def test_selected_adapter_check_for_the_bird_tags():
    from tinysql.seeds import check_selected_adapter

    check_selected_adapter("checkpoints/v2-s1/checkpoint-700", "v2-s1", 700)
    check_selected_adapter("checkpoints/v2-s1/checkpoint-700/", "v2-s1", 700)
    with pytest.raises(ValueError, match="checkpoint-900"):
        check_selected_adapter("checkpoints/v2-s1/checkpoint-900", "v2-s1", 700)
    with pytest.raises(ValueError):
        check_selected_adapter("checkpoints/v2-s2/checkpoint-700", "v2-s1", 700)
