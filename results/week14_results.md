# Week 14 results: the GRPO gain with two new training seeds (PLAN-week14)

**Setup.**
- **New runs:** the release recipe (v2) trained twice more, identical except the seed:
  - v2-s1, seed 1;
  - v2-s2, seed 2.
  - The release is v2 with seed 3407.
- **Same recipe, checked:** each run's `run_args.json` was checked against `checkpoints/v2/run_args.json`
  before training (`train_grpo.py --same-recipe-as`). Only `run_name` and `seed` differ.
- **What the seed changes:** LoRA init, question order and the sampled completions.
  - The training data is fixed. The pass-rate filter (1,688 questions) was computed once with the base
    model and is part of the recipe.
- **Selection:** the release rule, unchanged. Best dev-200 EX of the 10 checkpoints, ties to the later
  step.
- **Scoring:** the selected checkpoint, bare (schema-only prompt), nf4, greedy, 256 new tokens.
  - Spider test (2,147 questions): official EX, batch 8, guard 3,072, as week 11.
  - BIRD dev (1,534 questions): BIRD EX, batch 4, guard 5,500, as week 12.
  - A (zero-shot) and B (v2-900, the release) are reused, not rerun.
- **Tests:** four pre-registered comparisons, exact McNemar, two-sided, Bonferroni over four
  (p < 0.0125). Each run was trained and scored once.

## Results

| # | Data | Comparison | EX | Difference | w / l | p | Label |
|---|---|---|---|---|---|---|---|
| S1 | Spider test | v2-s1 vs A | 61.9% vs 56.3% | +5.6 | 164 / 43 | <0.0001 | **GRPO gain replicated** |
| S2 | Spider test | v2-s2 vs A | 61.7% vs 56.3% | +5.4 | 158 / 42 | <0.0001 | **GRPO gain replicated** |
| S3 | BIRD dev | v2-s1 vs A | 23.3% vs 20.5% | +2.8 | 70 / 27 | <0.0001 | **GRPO gain replicated** |
| S4 | BIRD dev | v2-s2 vs A | 23.9% vs 20.5% | +3.3 | 71 / 20 | <0.0001 | **GRPO gain replicated** |

**Overall reading (decided in the plan): replicated on both benchmarks.** Both new seeds are labelled
"replicated" on Spider test and on BIRD dev.

**Spread over the three seeds** (gain over zero-shot, points):

| Seed | Selected step | Spider test (official EX) | gain | BIRD dev (BIRD EX) | gain |
|---|---|---|---|---|---|
| 3407 (release, v2) | 900 | 61.7% | +5.4 | 23.6% | +3.1 |
| 1 (v2-s1) | 800 | 61.9% | +5.6 | 23.3% | +2.8 |
| 2 (v2-s2) | 700 | 61.7% | +5.4 | 23.9% | +3.3 |
| **mean (range)** | | | **+5.5 (5.4–5.6)** | | **+3.1 (2.8–3.3)** |

- **The release seed is not the lucky one.** On Spider test it is the lowest of the three, tied with
  seed 2 (both 61.67%). On BIRD dev it is in the middle.
- **The release does not change** (one-shot rule). Picking the seed that scores best on test would be
  selecting on the test set.

BIRD without `formula_1` (1,360 questions; secondary, as in weeks 12 and 13):
- S3: 24.9% vs 21.9%, 65 / 25, p<0.0001.
- S4: 25.4% vs 21.9%, 67 / 19, p<0.0001.

Sources:
- `results/official_ex_test.json` and `results/official_test/B-v2-s*`;
- `results/bird_compare_week14.json` and `results/bird_B-v2-s*.json`;
- `results/selection_v2-s*.json` and `results/checkpoint_eval_v2-s*.txt`;
- descriptive numbers: `scripts/week14_extras.py`.

## Reading

**The +5.4-point gain on Spider test is a property of the recipe (with the pass-rate filter held
fixed), not of one run.** All three seeds land within 0.25 points of each other on Spider test and within 0.6 on BIRD dev. Every one of them is
clearly above zero-shot on both benchmarks.

**No detectable difference between the seed runs and the release, or between the seed runs**
(descriptive McNemar, not pre-registered):

| Pair | Spider test, w / l | p | BIRD dev, w / l | p |
|---|---|---|---|---|
| v2-s1 vs B (release) | 38 / 33 | 0.64 | 16 / 20 | 0.62 |
| v2-s2 vs B (release) | 36 / 36 | 1.0 | 20 / 16 | 0.62 |
| v2-s2 vs v2-s1 | 37 / 42 | 0.65 | 24 / 16 | 0.27 |

- About 70–80 questions flip between any two seeds on Spider test, and they cancel out. That is about
  3.5% of the questions, so individual answers depend on the run while the average does not.

**The selected step depends on the seed (900, 800, 700), and it hardly matters.**
- The dev-200 curves are flat after step 100: every checkpoint of every run is between 58.0% and
  61.5%, against 55.0% for zero-shot.
- Step 900 of each seed run, scored on test without selection: 61.9% (v2-s1) and 62.1% (v2-s2),
  against 61.7% for the release. So, descriptively, the gain does not hinge on the selection rule.

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 (3407) | 58.5 | 58.5 | 59.5 | 60.0 | 60.0 | 59.5 | 59.0 | 58.5 | **60.0** | 58.5 |
| v2-s1 (1) | 58.5 | 60.0 | 58.5 | 59.0 | 58.5 | 59.5 | 59.0 | **61.5** | 60.0 | 60.5 |
| v2-s2 (2) | 59.0 | 59.0 | 58.0 | 60.5 | 59.5 | 60.5 | **61.5** | 60.0 | 60.0 | 59.5 |

Dev-200 EX (%), the selected step in bold.

**Training behaved the same way in all three runs:**

| | v2 (3407) | v2-s1 (1) | v2-s2 (2) |
|---|---|---|---|
| reward, steps 1–100 / 451–550 / 901–1000 | 0.576 / 0.685 / 0.710 | 0.592 / 0.664 / 0.714 | 0.587 / 0.685 / 0.732 |
| KL, last 100 steps | 0.049 | 0.050 | 0.043 |
| groups with no reward variance | 35.6% | 36.0% | 35.8% |
| format failures, Spider test / BIRD dev | 6 / 17 | 3 / 16 | 3 / 17 |
| training step time, hours (wall) | 5.3 | 6.1 (6.3) | 6.2 (6.3) |
| peak reserved VRAM, max GPU temperature | 5.75 GiB, 87 °C | 6.36 GiB, 87 °C | 5.86 GiB, 87 °C |

The seed runs took ~0.8–0.9 h more step time than v2 for the same 1,000 steps. They ran on the same laptop weeks
later; the cause was not investigated, and it does not affect the scores.

## Limits

- **Three seeds give a range, not a variance estimate.** They show that the gain is not one lucky run.
  They do not pin down the run-to-run spread exactly.
- **The pass-rate filter was held fixed.** Variation from re-running the filter is not measured.
- **Only the bare model was retrained.** The inference stack (repair, grounding, example rows,
  pruning) was tuned on v2-900 and was not rerun on the seed runs (option E, left out of the plan).
  The demo's 66.3% on Spider test is still a one-seed number.
- **Earlier one-seed comparisons are not revisited.** Filtered vs unfiltered training (+1.5 points,
  not significant), the reward ablation, v3 and v4 all remain single runs. This week's spread (0.2 points
  on Spider test between the selected checkpoints, 5 questions) suggests seed noise is small for this recipe, but it
  was measured for one recipe only.
- **The training stack is not fully deterministic** (vLLM sampling, bf16 kernels), so even seed 3407 would
  not reproduce v2 exactly. That variation is part of what was measured.

## Notes

- **Run logs:** `results/v2-s1_runs.log`, `results/v2-s2_runs.log` (`scripts/run_seed.sh`, one seed
  unattended: training, dev-200 selection, Spider test, BIRD dev). Official Spider scoring:
  `scripts/official_eval.py --split test`.
- **Run times (RTX 3070 Ti Laptop), per seed:** training ~6.3 h wall, dev-200 selection ~40 min, Spider test
  ~27 min per checkpoint, BIRD dev ~50 min.
- **Training logs:** `results/train_log_v2-s1.csv`, `results/train_log_v2-s2.csv`.
