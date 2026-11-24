# Week 14 plan: does the GRPO gain survive a new seed? Two more training runs of the release recipe

**Status: APPROVED 2026-11-24, as written (option E not added).**

## Why

Every training result in this project comes from one run with seed 3407. The McNemar tests cover
which questions were asked, not how the training run happened to go. So "GRPO adds +5.4 points on
Spider test" may be partly a lucky run. This is the largest open gap in the project's claims. The
release recipe (v2: pass-rate-filtered data, partial reward, 1,000 steps, best dev-200 checkpoint)
is repeated with two new seeds, and each result is scored on the same held-out sets as the release.

## Facts checked before writing

- **The recipe is fully recorded** in `checkpoints/v2/run_args.json`: Qwen3-1.7B, 4-bit QLoRA r=16,
  8 generations x 4 prompts per step, 1,000 steps, lr 5e-6, beta 0.04, partial reward, max prompt
  2,304, `--pass-rate data/train_pass_rate.jsonl`, seed 3407. `train_grpo.py --seed` sets the LoRA
  init (`random_state`), the data order and the trainer seed (`GRPOConfig.seed`).
- **Timings from the v2 run** (`results/train_log_v2.csv`, RTX 3070 Ti Laptop):
  - training: 5.3 h for 1,000 steps, peak reserved VRAM 5.75 GiB;
  - one dev-200 checkpoint eval: ~3.5 min (10 per run, ~35 min);
  - Spider test, bare: 34 min; BIRD dev, bare: 79 min.
- **Disk:** a run's checkpoints take 1.2 GB; C: has 35 GB free.
- **Nothing is downloaded.** All data and the base model are local.
- **What a seed does not change:** the training data. The pass-rate filter (1,688 questions) was
  computed once with the base model and is kept fixed, as it is part of the recipe. Seed variance
  here means LoRA init, question order and the sampled completions.

## Question

**Is the release's GRPO gain over zero-shot Qwen3-1.7B a property of the recipe or of one run?**
Measured on Spider test (official EX) and BIRD dev (BIRD EX), bare model, no inference steps.

## Method (fixed in advance; each run once)

**Two new training runs, identical to v2 except the seed:**

| Run | Seed | Command |
|---|---|---|
| v2-s1 | 1 | `train_grpo.py --run-name v2-s1 --seed 1 --pass-rate data/train_pass_rate.jsonl` |
| v2-s2 | 2 | `train_grpo.py --run-name v2-s2 --seed 2 --pass-rate data/train_pass_rate.jsonl` |

All other arguments are the defaults, which match `checkpoints/v2/run_args.json`. The script writes
its own `run_args.json`, and the task checks that it differs from v2's only in `run_name` and `seed`.

**Checkpoint selection: the release rule, unchanged.** Best dev-200 EX among the 10 checkpoints,
ties to the later step (PLAN-week3), via `compare_checkpoints.py`. This replicates the whole recipe,
selection included. Step 900 of each run is also scored on test, as a descriptive check that does
not depend on selection.

**Scoring:** the selected checkpoint of each run, bare (schema-only prompt), nf4, greedy, 256 new
tokens, exactly as week 11 (Spider test, batch 8, guard 3,072) and week 12 (BIRD dev, batch 4,
guard 5,500). A (zero-shot) and B (v2-900, seed 3407) are reused, not rerun.

**Pre-registered comparisons** (paired exact McNemar, two-sided; Bonferroni over four, p < 0.0125):

| # | Data | Comparison |
|---|---|---|
| S1 | Spider test | v2-s1 vs A |
| S2 | Spider test | v2-s2 vs A |
| S3 | BIRD dev | v2-s1 vs A |
| S4 | BIRD dev | v2-s2 vs A |

**Labels per comparison:** "GRPO gain replicated" (significant, seed run higher), "GRPO worse"
(significant, lower), "no detectable difference".

**Overall reading, decided now:**
- Replicated on a benchmark if both new seeds are labelled "replicated" there.
- Partly replicated if one of the two is.
- Not replicated if neither is.

**Reported but not tested:**
- Mean and range of the GRPO gain over the three seeds (3407, 1, 2), per benchmark.
- Where the release seed sits: lowest, middle or highest of the three.
- Seed runs against each other and against B (e.g. v2-s1 vs B), as descriptive McNemar.
- The selected step per run, the dev-200 curve, training reward and KL, as in week 3.
- Format failures per run.

**One-shot rules:**
- Each run is trained and scored once. A crashed run is resumed from its last checkpoint
  (`--resume`); a run is never restarted to get a different result.
- **The release does not change**, whatever the result. Switching to the seed that scores best on
  test would be selecting on the test set. The README and the model card report the spread.
- No change to the recipe, the selection rule or the evaluation because of these results.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| M0 | Registries, test-first | `scripts/official_eval.py` (`TEST_MODELS` and `TEST_PAIRS` entries for the seed runs), `scripts/compare_bird.py` (S3/S4 pairs), `tests/` | Tests: the seed-run record names resolve; a check that a run's `run_args.json` differs from v2's only in `run_name` and `seed` (passes on toy files, refuses an extra difference). |
| — | **Review gate** (M0) | | Reviewer approves before training starts. |
| M1 | Train v2-s1 (overnight) | `checkpoints/v2-s1/`, `results/train_log_v2-s1.csv` | 1,000 steps; `run_args.json` check passes. |
| M2 | Select and score v2-s1 | `results/checkpoint_eval_v2-s1.md`, `results/test_*v2-s1*`, `results/bird_*v2-s1*` | 10 dev-200 evals, the selected checkpoint per the rule, Spider test (official EX) and BIRD dev for it, Spider test for step 900. |
| M3 | Train v2-s2 (second night) | `checkpoints/v2-s2/`, `results/train_log_v2-s2.csv` | as M1 |
| M4 | Select and score v2-s2 | as M2 | as M2 |
| M5 | Write-up + README + model card + final review | `results/week14_results.md`, README, `release/model_card.md`, `scripts/check_readme.py` | S1–S4 labelled, the overall reading, the spread table; `check_readme.py` covers the new numbers. Upload and push only after you say so. |

**Cost:** ~1–2 h code. GPU ~15–16 h over two nights:
- per seed: training 5.3 h + dev-200 selection ~35 min + Spider test ~35 min (+ ~35 min for step 900
  if it is not the selected one) + BIRD dev ~80 min, so ~7.5–8 h.

## Risks

- **Two new seeds is a small sample.** Three runs give a range, not a variance estimate. It can show
  that the gain is not one lucky run, but not pin down its spread exactly.
- **The training stack is not fully deterministic** (vLLM sampling, bf16 kernels). Even a rerun with
  seed 3407 would not reproduce v2 exactly. That is part of what is being measured.
- **Heat and long runs.** v2 peaked at 87 °C and ran 5.3 h without trouble. A crash is handled with
  `--resume` (see the one-shot rules), and the gap is noted.
- **The pass-rate filter is held fixed**, so this does not measure variation from re-running the
  filter. That is stated in the write-up.

## Option to decide (not in the plan unless you add it)

**Also score the recommended configuration (E: repair + grounding + rows + pruning) for each seed**,
descriptive only. It would show whether the demo's 66.3% on test also depends on the seed. It adds
~1 h (Spider test) and ~2 h (BIRD dev) per seed, so ~6 h in total. My recommendation is to leave it
out: the inference steps were tuned on v2-900 and do not change with the seed, and the training
question is answered by the bare model.

## Not in this plan

Changing the release, other recipes (unfiltered data, binary reward, repair or rows training) with
new seeds, re-running the pass-rate filter, more than two seeds, voice evaluation, and Coder.
