# Week 5 plan: teaching the model to use error feedback (multi-turn repair GRPO)

**Status: APPROVED 2026-11-18.**

This extends the 4-week SPEC. It comes first in README "Future work".

## Why

The Y9 repair step gives the model one retry when its SQL fails, with the error message
(`results/week4_results.md`, Y9):
- 133 retries on full dev rescued 19 questions (14%).
- **In 53 of them (40%) the model wrote the identical SQL again.**

The model was trained only on single-turn prompts and has never been rewarded for using an error
message. This plan trains exactly that turn and nothing else changes.

## Question and success criterion

**Does GRPO on repair turns raise the repair success rate, without hurting first-attempt accuracy?**

Each quantity has its own test, all fixed now, before training:

| Quantity | Measured on | Test |
|---|---|---|
| **Repair success rate** (primary): of the retries, how many end correct | full dev, Y9 protocol | v2-900's own failures re-run through the new model's repair turn (paired on the same 133 inputs, McNemar). Also on the new model's own failures. |
| **First-attempt EX** (guard rail) | full dev, ours + official | Paired McNemar vs v2-900. A significant loss means the run failed, whatever the repair rate. |
| **End-to-end EX with repair** (headline) | full dev, ours + official; voice with repair | Paired vs v2-900 + repair. |

**A null result is a valid outcome.** It is reported as such, as in weeks 3 and 4.

## Design

**Repair-turn data (training questions only, no dev, no SpokenSpider):**
1. Sample v2-900 on the 6,638 training questions (empty gold already dropped) with the training
   decoding: vLLM, temperature 1.0, 8 samples. This is the same machinery as `scripts/pass_rate.py`,
   plus the LoRA adapter.
2. Keep samples whose SQL does not execute or that have no ```sql block. Their executor error becomes
   the repair prompt, built with `tinysql.repair.repair_messages`, so it is exactly the turn used at
   inference.
3. Deduplicate per (question, failed SQL). Keep at most 2 failures per question so a few hard
   questions cannot dominate.
4. **Stop rule:** if fewer than ~800 repair prompts result, stop and decide with the user (options:
   more samples per question, or include wrong-result failures with a different prompt).

**Training:**
- Continue from the v2-900 adapter: same LoRA and settings as v2; fresh optimizer and scheduler.
- **Data mix per step:** 50% repair prompts, 50% first-turn prompts from the v2 filtered set (1,688).
  The first-turn half guards against forgetting.
- **Reward:** the existing execution reward (1.0 / 0.1 / 0), computed on the repair turn's answer.
  - Repeating the failed SQL scores 0, so it is discouraged with no new reward term.
  - No reward shaping is added, so there is nothing new to hack. Re-read the top-reward repair
    samples by hand, as in W7.
- **500 steps:** about 3.5–4 h (repair prompts are longer; the smoke run measured 18–53 s/step), one overnight run started by you, with checkpoints every 100 steps.
- **Logged:** reward split by prompt type (repair vs first-turn) and `frac_reward_zero_std` per type.

**Selection** (fixed now): best dev-200 **EX with repair** among checkpoints 100–500, ties to the
later step. Then the evaluation in the table above, once, on full dev.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| Z1 | Repair-prompt builder, test-first | `src/tinysql/grpo_data.py` (`build_repair_records`), `tests/test_grpo_data.py` | Tests: only non-executing / no-block samples become records; the prompt equals `repair_messages(original, completion, error)`; dedup and the 2-per-question cap; gold and db_path carried. |
| Z2 | Sample v2-900 failures on train | `scripts/sample_failures.py` → `data/train_repair_candidates.jsonl` | Counts logged: samples, failures by type, records after dedup and cap. Stop rule applied. |
| Z3 | Mixed training | `scripts/train_grpo.py --init-adapter --repair-data --repair-frac 0.5` (+ tests for the mixing/record schema where it is pure code) | 2-step smoke run passes. The reward is logged per prompt type. Then you start the 500-step run. |
| — | **Review gate** (Z1–Z3 code) | | Reviewer approves before the overnight run. |
| Z4 | Checkpoint selection + evaluation | `results/checkpoint_eval_repair.md`, `results/week5_results.md` | dev-200 with repair for checkpoints 100–500. The selected one gets: full dev with repair (ours + official), the first-attempt guard rail, the paired repair-success comparison on v2-900's 133 failures, voice with repair, and a hand read of 20 top-reward repair samples. |
| Z5 | Release (only if it beats v2-900 + repair) | merge, Q8_0 GGUF, demo-path check, HF upload, README / card | Same pipeline as week 4. Each upload is asked first. If it does not beat v2-900 + repair, nothing is released and the README reports the null result. |
| — | **Final review gate** | | Reviewer approves; README numbers verified (`scripts/check_readme.py`). |

## Risks

- **Too few repair prompts.** At greedy decoding the model fails to execute ~12% of the time; at
  temperature 1.0 it should fail more often. Handled by the Z2 stop rule.
- **Forgetting first-turn skill.** Handled by the 50/50 mix and the first-attempt guard rail.
- **Overfitting to train errors.** The same ~1,000 prompts are seen ~1.25× in 500 steps (2,000 repair
  slots). Lower than v2's 2.4×.
- **Single seed.** As before, McNemar covers question sampling, not training variance.
- **Partial-reward bias on the repair turn.** 0.1 for "now executes but wrong" might teach the model
  to make SQL run rather than be right. Watch the share of repair outcomes that become wrong results:
  Y9 baseline 25 of 133. Do not change the reward mid-run.

## Not in this plan

ASR value normalisation, more seeds, schema grounding and BIRD stay in "Future work".
