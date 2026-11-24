"""PLAN-week14: the v2 recipe repeated with new seeds.

- `check_same_recipe`: a seed run's run_args.json may differ from v2's only in run_name and seed.
- `select_checkpoint`: the release rule (PLAN-week3), best dev-200 EX, ties to the later step.
- `adapter_suffix` / `seed_entries`: record names and Spider test pairs for the selected checkpoints.
"""

from __future__ import annotations

from pathlib import Path

ALLOWED = ("run_name", "seed")
# train_grpo.py arguments added after the v2 run, with the values that leave the v2 recipe unchanged.
LATER_ARGS = {"init_adapter": None, "repair_data": None, "repair_frac": 0.5, "sample_rows": 0}
_MISSING = object()


def check_same_recipe(ref: dict, new: dict) -> None:
    diffs = []
    for key in sorted(set(ref) | set(new)):
        if key in ALLOWED:
            continue
        a, b = ref.get(key, _MISSING), new.get(key, _MISSING)
        if a is _MISSING or b is _MISSING:
            present = b if a is _MISSING else a
            if key not in LATER_ARGS or present != LATER_ARGS[key]:
                diffs.append(f"{key}: {'missing' if a is _MISSING else a!r} vs {'missing' if b is _MISSING else b!r}")
        elif a != b:
            diffs.append(f"{key}: {a!r} vs {b!r}")
    if diffs:
        raise ValueError("not the same recipe: " + "; ".join(diffs))


def select_checkpoint(ex_by_step: dict[int, float]) -> int:
    best = max(ex_by_step.values())
    return max(step for step, ex in ex_by_step.items() if ex == best)


def adapter_suffix(adapter: Path) -> str:
    """The part eval_baseline.py adds to an output name for a LoRA checkpoint."""
    return f"_{adapter.parent.name}-{adapter.name}"


def seed_entries(selected: dict[str, int]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """TEST_MODELS entries and (candidate, base) pairs for the seed runs selected so far.

    Each run gets its selected checkpoint ("B-<run>") and, when that is not step 900, step 900 too
    ("B-<run>-900", descriptive). The first pairs per run are S1/S2 (vs zero-shot A).
    """
    models, pairs = {}, []
    for run, step in selected.items():
        models[f"B-{run}"] = adapter_suffix(Path(run) / f"checkpoint-{step}")
        pairs += [(f"B-{run}", "A-base"), (f"B-{run}", "B-v2-900")]
        if step != 900:
            models[f"B-{run}-900"] = adapter_suffix(Path(run) / "checkpoint-900")
            pairs.append((f"B-{run}-900", "A-base"))
    runs = list(selected)
    pairs += [(f"B-{b}", f"B-{a}") for i, a in enumerate(runs) for b in runs[i + 1:]]
    return models, pairs
