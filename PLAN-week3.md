# Week 3 implementation plan: evaluation, v2 run, ablation, voice swap (from SPEC.md §2, §5.3–5.4, §6)

**Status: APPROVED 2026-11-16.**

Goal of the week: fill the results table for RQ1–RQ4 with numbers that trace back to result files.

## Where week 2 left off

- **Main run, checkpoint-1000:** 59.0% on dev-200 vs 55.0% base. The gain looks mostly like format
  fixes (`results/checkpoint_eval_main.md`).
- **Training signal:** in ~65% of groups all 8 samples got the same reward, so they gave no learning signal.
- **Reward hack found in W7:** questions whose gold result is empty reward wrong queries. It is fixed
  in code: `drop_empty_gold` is the default (`results/hack_check.md`).

## Decision (2026-11-16, user)

Do **v2 run + ablation**:
- **v2:** train on filtered data. Drop empty-gold questions, and drop questions the base model solves
  in 0 or 8 of 8 samples. Partial reward.
- **Ablation:** the same filtered data with the binary reward, so the two runs differ only in the reward.
- The week-2 main run stays as the "unfiltered" reference point.

Same rules as before:
- CPU code is test-first.
- GPU work runs in WSL with `~/venvs/tinysql`.
- Every comparison of two models is paired (exact McNemar) on the same questions, with the same
  command, greedy, 4-bit. Differences under ~1.5pp on dev-200 are noise.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| X1 | Full-dev EX of week-2 checkpoint-1000 (RQ1, first answer) | `scripts/paired_compare.py` (general version of `compare_checkpoints.py`: any two records files), `results/baseline_*_main-checkpoint-1000.json` | All 1,034 dev questions are evaluated with the baseline command plus `--adapter`. EX, format/exec breakdown and McNemar vs base 53.7% are written to `results/week3_results.md`. |
| X2 | Difficulty filter | `src/tinysql/grpo_data.py` (`filter_by_pass_rate`), `tests/test_grpo_data.py`, `scripts/pass_rate.py`, `data/train_pass_rate.json` | Test-first: the pure filter keeps 1–7/8 and is keyed by (db_id, question). The base model (no adapter) samples 8 completions per training question with the training decoding via vLLM (`fast_generate`). The pass-rate histogram and filtered set size are logged. **If fewer than ~1,500 questions remain, stop and decide with the user** (1,000 steps × 4 prompts would then be >2.5 epochs). |
| — | **Review gate** (X1–X2 code) | | Reviewer approves |
| X3 | v2 run: filtered data, partial reward | `scripts/train_grpo.py --pass-rate data/train_pass_rate.json --run-name v2` | 1,000 steps, same settings as week 2 otherwise. Logged: `frac_reward_zero_std` (expected to drop well below 0.65), reward curve, VRAM, temperature. Checkpoints 100..1000 evaluated on dev-200 (`results/checkpoint_eval_v2.md`). |
| X4 | Ablation: filtered data, binary reward | `--binary-reward --run-name v2-binary` | Same as X3. Selection rule for both runs, fixed in advance: **best dev-200 EX, ties go to the later step.** |
| X5 | Full-dev EX of the selected v2 and v2-binary checkpoints (RQ1 + ablation) | `results/week3_results.md` | Table rows: base, main-1000, v2, v2-binary. Paired McNemar for: v2 vs base, v2 vs main-1000, v2 vs v2-binary. Error breakdown (format / exec error / wrong result) per model. |
| X6 | Voice EX: our model vs SQLCoder on the same transcripts (RQ2, RQ4) | `data/spokenspider_transcripts.csv` (slim copy of voice-to-sql `metadata.csv`: id, db, SNR, question, ASR transcript, gold, SQLCoder SQL), `scripts/eval_voice.py` | The best model generates SQL for all 400 ASR transcripts and for the 100 clean question texts (no ASR). SQLCoder's stored SQL is re-scored with **our** comparator, so both use the same EX. Output: EX per SNR for both models, paired McNemar per condition. |
| X7 | Latency (RQ3) | `scripts/bench_latency.py`, `results/latency.json` | Our model and SQLCoder-7B, both nf4 4-bit, HF generate, batch 1, same GPU. 100 clean transcripts after 5 warm-up queries. Median and p95 text-to-SQL latency. SQLCoder uses its voice-to-sql prompt (copied, with a pointer to the source). End-to-end estimate: the voice-to-sql per-row totals with the SQL stage replaced by the measured latency. |
| — | **Review gate** after X7 | | Reviewer approves; `results/week3_results.md` answers RQ1–RQ4 |

Order: X1 and X2 are tonight's work. X3 runs overnight, then X4 the next night. X6 and X7 are
independent of training and can run with week-2 checkpoint-1000 while X3/X4 train. They are
re-run with the final model, which costs about 30 minutes.

## Risks

- **The filter leaves too few questions.** This is handled by the stop rule in X2. Fallback: keep 1/8–8/8
  (drop only never-solved), or lower `--max-steps`.
- **v2 is not better than main-1000.** That is a valid result (SPEC §2). Report it with the
  zero-signal numbers that motivated it.
- **SQLCoder-7B in 4-bit leaves no room alongside anything else on 8 GB.** Run X7's two models in
  separate processes, never at the same time as training.
- **voice-to-sql EXA ≠ our EX.** Its published table used its own matcher and a SQL-repair step.
  We re-score its stored SQL with our comparator and state the difference. The repair step is part of
  its pipeline, so it stays in.
- **The pass-rate sampling is slow.** It needs 6,638 × 8 generations. With vLLM this is estimated at
  1–2 h; the X2 log measures it. Without vLLM it is not feasible, so the fallback is filtering on
  100-step training logs instead (to be designed only if needed).
