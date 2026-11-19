# Z4: checkpoint selection on dev-200 (v3: repair-turn GRPO from v2-900, 500 steps)

The command is the week-3 command (greedy, 4-bit, batch 8, 256 tokens) plus `--repair` (one retry,
PLAN-week4 Y9). The reference is v2-900 with the same command:

| Model | EX with repair | First-attempt EX | Retries | Rescued | vs v2-900 with repair (w / l, p) | First attempt vs v2-900 (w / l, p) |
|---|---:|---:|---:|---:|---:|---:|
| v2-900 (reference) | 62.5 | 60.0 | 21 | 5 | | |
| v3-100 | 62.0 | 60.5 | 20 | 3 | 3 / 4, 1.0 | 5 / 4, 1.0 |
| v3-200 | 62.0 | 59.5 | 22 | 5 | 5 / 6, 1.0 | 5 / 6, 1.0 |
| v3-300 | 65.0 | 63.0 | 18 | 4 | 8 / 3, 0.23 | 8 / 2, 0.11 |
| **v3-400** | **65.0** | 62.5 | 21 | 5 | 7 / 2, 0.18 | 7 / 2, 0.18 |
| v3-500 | 63.5 | 61.5 | 18 | 4 | 8 / 6, 0.79 | 9 / 6, 0.61 |

Sources: `results/dev200_Qwen__Qwen3-1.7B_v3-checkpoint-*_repair.json`,
`results/dev200_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair.json`.

**Selection.** The rule was fixed in PLAN-week5 before training: best dev-200 EX with repair, ties
to the later step. Steps 300 and 400 tie at 65.0, so the choice is **checkpoint-400**.

**Reading.**
- On dev-200 the gain over v2-900 comes from **first attempts** (62.5 vs 60.0), not from repair.
  Rescued counts stay at 3–5 of ~20 retries at every checkpoint, the same as v2-900's 5 of 21.
- No difference is significant; dev-200 selects, it does not test.
- The training log is consistent with this:
  - repair-turn reward rose only from ~0.23 to ~0.29 per 100-step mean;
  - first-turn reward stayed at ~0.74;
  - about 57% of groups had no reward variance (full-run mean).

**Control for the paired repair-turn measure.** `scripts/eval_repair_turn.py` replays v2-900's 133
failed first attempts from its full-dev repair run through v2-900's own repair turn. It gives 20
rescued and 54 identical repeats, against 19 and 53 in the original run. The 1-row difference is
batch-composition noise.
