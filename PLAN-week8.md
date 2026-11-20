# Week 8 plan: training with example rows in the prompt

**Status: APPROVED 2026-11-19 (approach A). Done 2026-11-20: primary null (official 63.4 -> 63.9, p=0.58); v2-900 stays the release.**

This is README "Future work" item 4, next step after week 7.

## Why

Week 7 put 3 example rows per table in the prompt, at inference only (`results/week7_results.md`):
- official EX 60.9 → 63.4 (72 / 46, p=0.021); voice +5..+8 in every condition;
- the gain is mostly fewer `no such column` errors (74 → 60);
- but **112 questions changed outcome**: on this off-distribution prompt, greedy answers move a lot.

The model was trained only on schema-only prompts. It has never been rewarded for reading example
rows. This plan trains on exactly the prompt the demo now uses, and changes nothing else.

## Question and success criterion

**Does GRPO training with example rows beat v2-900 when both are evaluated with example rows?**

| Quantity | Measured on | Test |
|---|---|---|
| **Primary: EX in the release configuration** (rows, repair, grounding) | full dev 1,034, official (decision) + ours | paired McNemar vs v2-900's week-7 records (same prompt, same config; not re-run) |
| Voice EX per condition (guard rail) | 100 SpokenSpider × 5, same config | paired McNemar per condition, not pooled |
| Error mix | full dev | counts: wrong result, `no such column`, other exec error, format |
| Schema-only EX (secondary) | full dev, no rows, repair + grounding | paired vs v2-900's week-6 records. Does the new model still work with rows turned off (`TINYSQL_SAMPLE_ROWS=0`)? Reported, not part of the decision. |

**Decision rule (fixed now):** the new model replaces v2-900 as the release **only if** the
official-EX gain is significant (p<0.05) **and** no voice condition is significantly worse.
Otherwise v2-900 + rows stays, and the result is reported. **A null result is a valid outcome**, as in
weeks 3 and 5.

## Approaches considered

| | Approach | For | Against |
|---|---|---|---|
| **A (recommended)** | **Re-run the v2 recipe from the base model with rows in the prompt** (run name `v4`): same 1,688 filtered questions, same reward, LoRA, LR, β, 1,000 steps, seed; selection rule as in week 3 | Clean comparison: v4 and v2-900 differ only in the training prompt, and both are selected the same way from 10 checkpoints | One overnight run, maybe longer (prompts ~2× longer) |
| B | Continue from v2-900 for ~500 steps with rows | Cheaper | Confounded with extra training, the week-5 problem (v3-400 also gained on dev from more steps alone). A fair control needs a second run without rows, so it is not cheaper after all |
| C | A, but 50% of prompts with rows, 50% without | One model for both demo settings | Two changes at once; halves the with-rows signal. Only worth it if the schema-only secondary shows a real loss |

**The filter is reused, not recomputed.** The 1,688 questions come from base-model pass rates on
schema-only prompts (`data/train_pass_rate.jsonl`). Re-sampling with rows would change the data
as well as the prompt. Cost: some questions may become all-solved or all-failed with rows, so they give
no signal. This shows in `frac_reward_zero_std` (v2 started at 0.25) and is reported.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| R1 | Plumbing, test-first | `scripts/train_grpo.py --sample-rows`, `scripts/compare_checkpoints.py --suffix _rows3 --base <v2-900 rows3 dev-200 records>`, `tests/test_grpo_data.py` | Tests: `build_grpo_records(..., sample_rows=3)` puts `-- Example rows from` in every prompt; `sample_rows=0` gives today's records byte for byte; the question set (db_id, question) is the same for both. |
| R2 | Train prompt lengths | `scripts/prompt_lengths.py` on the 1,688 train questions → `results/prompt_lengths_train_rows3.json` | Median / max tokens with rows. `--max-prompt-length` set just above the max (prompts are never truncated). **Stop rule:** if the max exceeds ~3,000 tokens, stop and decide with you (options: cap long cell values, fewer rows for very wide tables). |
| R3 | 50-step pre-flight, longest prompts first | `train_grpo.py --sample-rows 3 --longest-first --max-steps 50 --run-name preflight-rows` | Peak VRAM, s/step, GPU temperature logged. **Stop rule:** OOM, or >40 s/step (1,000 steps > 11 h): stop and decide with you (options: 600 steps, G=6). |
| — | **Review gate** (R1–R3) | | Reviewer approves before the overnight run. |
| R4 | `v4` run | `train_grpo.py --pass-rate data/train_pass_rate.jsonl --sample-rows 3 --run-name v4` | **You start it** (overnight). 1,000 steps, checkpoints every 100. Logged: reward, `frac_reward_zero_std`, KL, completion length, VRAM. |
| R5 | Selection + evaluation | `results/checkpoint_eval_v4.md`, `results/baseline_*v4*`, `results/voice_*v4*`, `results/official/` | **Selection (fixed now):** best dev-200 EX with rows, first attempt, greedy (same protocol as week 3's selection, plus rows); ties go to the later step. The selected checkpoint, once: the measures table above, plus `scripts/hack_check.py` on top-reward samples. |
| R6 | Docs + decision | `results/week8_results.md`, README (+ `check_readme.py`) | The decision rule applied as written. |
| R7 | Release (only if the rule is met) | merge, Q8_0 GGUF, demo-path check, HF upload, README / model card | Week-4 pipeline. Each upload is asked first. Latency is not re-measured: the prompts are the same as week 7. |
| — | **Final review gate** | | Reviewer approves; README numbers verified. |

**Cost:** ~1 h code (R1–R3). GPU: ~30 min pre-flight, one overnight run (~8–10 h estimated from
week 5's 28 s/step on similarly long repair prompts; R3 measures it), ~1 h checkpoint selection,
~30 min full dev + voice. CPU: ~10 min official EX.

## Risks

- **VRAM on 8 GB.** Prompts roughly double (dev median 304 → 647 tokens). Week 5's longer repair prompts
  peaked at 5.87 GiB, so it should fit; R3 checks the worst case first.
- **Single seed.** As before, McNemar covers question sampling, not training variance. Week 3 showed
  that two 1,000-step runs can differ by ~1–1.5 points by chance. A small gain here is in that range.
- **Power.** Beating v2-900 + rows (63.4%) needs a clear net win count; week 7 needed 72 / 46 for
  p=0.021. A real gain of ~1 point would likely not reach significance.
- **Selection bias.** Selecting on dev-200 inflates dev-200, as for v2-900; both are selected the same
  way, and full dev is the test.
- **Copying values.** With rows, the model may copy example values into `WHERE` clauses instead of
  values from the question. The error mix and the hack check look for it; the reward is unchanged.
- **Loss without rows.** The demo can turn rows off. The schema-only secondary measures this; if it
  drops clearly, approach C becomes the follow-up.

## Decision at the R2 stop rule (2026-11-19, user)

With 3 rows, training prompts reach 6,239 tokens (median 792); 58 of 1,688 exceed 3,000, all from
baseball_1, sakila_1 and cre_Drama_Workshop_Groups (`results/prompt_lengths_train_rows3.json`). The
cause is table count, not long cells. **Decision: those prompts use the schema-only prompt**
(`rows_or_schema_only`, threshold = `--max-prompt-length 3000`). The question set is unchanged. On
dev the longest prompt with rows is 1,897 tokens, so the same rule would change nothing there, and the
comparison with v2-900's week-7 records stays valid.

## Decision at the R3 stop rule (2026-11-19, user)

The pre-flight (50 steps, longest prompts first) ran without OOM: 6.73 GiB allocated, 7.78 GiB
reserved, 87 °C with software thermal slowdown. The median was 109 s/step, against 33 s for the
schema-only pre-flight (`results/train_log_preflight-rows.csv`). Scaling by the old pre-flight/full-run
ratio and the 2× longer typical prompt gives roughly 45–60 s/step, i.e. 12–17 h for 1,000 steps (an
estimate; the first steps of R4 measure it). **Decision: keep the recipe unchanged (1,000 steps, G=8)
and split the run over two nights with `--resume`.**

## Not in this plan

More seeds, ASR word-error normalisation, repair-focused training and BIRD stay in "Future work".
The demo video is recorded only when the project is finished.
