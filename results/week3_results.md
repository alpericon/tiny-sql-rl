# Week 3 results (PLAN-week3)

All EX numbers come from the same command:
- greedy decoding, nf4 4-bit, batch 8, 256 new tokens, thinking off;
- column-permutation-tolerant multiset EX (SPEC §12).

Paired tests are exact two-sided McNemar on the same questions (`scripts/paired_compare.py`).

## RQ1: GRPO vs zero-shot, full Spider dev (1,034)

| Model | EX | Official EX (week 4, Y1) | Format fail | Exec error | Wrong result | vs base: wins / losses | p |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B zero-shot | 53.7% | 54.1% | 51 | 130 | 298 | | |
| + GRPO main (week 2), ckpt-1000 | 57.1% | 58.1% | 12 | 119 | 313 | 54 / 19 | 0.0001 |
| + GRPO v2 (filtered, partial reward), ckpt-900 | **58.5%** | **59.0%** | 10 | 123 | 296 | 83 / 33 | <0.0001 |
| + GRPO v2-binary (filtered, binary reward), ckpt-800 | 58.2% | 58.8% | 7 | 124 | 301 | 78 / 31 | <0.0001 |

The "Official EX" column is Spider's official evaluation script re-scoring the same predictions.
See the "Week 4 Y1" section at the end for how it was run and why it differs from ours.

Sources:
- `results/baseline_Qwen__Qwen3-1.7B.json`
- `results/baseline_Qwen__Qwen3-1.7B_main-checkpoint-1000.json`

**+3.4pp, significant for this run.** This is a single training run with one seed: McNemar covers the noise from which dev questions were sampled, not run-to-run training variance. The 54 wins break down by the base model's outcome on the same question:
- 29 were format failures;
- 10 were execution errors;
- 15 were wrong results.

So roughly half the gain is format and half is SQL. The dev-200 estimate (8 of 11 wins from format)
overstated the format share. Some fixed format failures became wrong results instead, which is why
wrong results rose from 298 to 313.

## X5: v2 and the reward ablation, full dev

Checkpoints were chosen on dev-200 with the rule fixed in advance (`results/checkpoint_eval_v2.md`,
`results/checkpoint_eval_v2-binary.md`). Sources: `results/baseline_Qwen__Qwen3-1.7B_v2-checkpoint-900.json`,
`results/baseline_Qwen__Qwen3-1.7B_v2-binary-checkpoint-800.json`.

The three planned v2 comparisons, plus v2-binary vs base. Bonferroni over 3 comparisons gives α = 0.017.

| Comparison (n = 1,034) | EX | wins / losses | p |
|---|---|---|---|
| v2 vs base | 58.5 vs 53.7 (+4.8pp) | 83 / 33 | <0.0001 |
| v2 vs main-1000 | 58.5 vs 57.1 (+1.5pp) | 42 / 27 | 0.091 |
| v2 vs v2-binary | 58.5 vs 58.2 (+0.3pp) | 19 / 16 | 0.74 |
| v2-binary vs base | 58.2 vs 53.7 (+4.5pp) | 78 / 31 | <0.0001 |

**Held-out check against selection bias.** dev-200 chose the checkpoints, so the same comparisons were
repeated on the 834 dev questions outside dev-200. (The full-dev runs score dev-200 questions slightly
differently from the dev-200 runs, e.g. base 55.5% vs 55.0%. This is the batch-composition noise.)

| 834 held-out questions | base | main-1000 | v2-900 | v2-binary-800 |
|---|---|---|---|---|
| EX | 53.2% | 56.8% | 58.3% | 58.3% |

- On the held-out questions, v2 vs base is 68 / 26 (p<0.0001) and v2 vs main is 35 / 23 (p=0.15).
- **v2 vs v2-binary is 15 / 15**, identical.
- So selection bias did not inflate the result: the held-out gains match the full-dev ones.

### Reading

1. **GRPO on filtered data beats the base model clearly: +4.8pp, p<0.0001.** Both v2 variants do. As
   with main, this is one seed per run, and McNemar does not cover training variance.
2. **Filtering is not a significant improvement over the unfiltered main run.** The difference is
   +1.5pp (42 / 27, p=0.09; held-out p=0.15). It points the right way, but a single seed per run
   cannot separate it from run-to-run variance. The confounds listed below also apply.
   - The kind of gain differs. Of v2's 83 wins over base, 33 were format failures, 17 execution errors
     and 33 wrong results; main's 54 wins were 29 / 10 / 15.
   - Against main, 29 of v2's 42 wins were questions main got wrong with valid SQL. Format failures
     are already near zero in both (12 → 10).
   - Read with the training logs, this fits v2 learning more SQL and less format. This is the
     only place where v2 and main clearly differ.
3. **The reward ablation found no effect.** The 0.1 partial credit for executable SQL changed nothing
   measurable: 19 / 16 on full dev and 15 / 15 held-out. The two runs differ on only 35 of 1,034
   questions. With pass-rate filtering, the binary reward gives about the same learning signal.
4. **What still fails is mostly SQL.** For v2 the failures are 10 format, 123 execution errors and 296
   wrong results. Execution errors barely moved in any run (130 → 119–124). Remaining gains would
   have to come from schema grounding, not output format.

## Caveat for the v2 comparison (from the X1–X2 review)

The v2 filter drops questions the base model solves 0/8 or 8/8, so v2 trains on a smaller,
mid-difficulty subset of train. v2 vs main is therefore confounded:
- by dataset size and composition, not only by learning-signal density;
- by the pass rates being measured on the base model while the policy drifts during training.

X5 reports the kept fraction and the pass-rate histogram next to the result.

## X2: base-model pass@8 on the training set

`data/train_pass_rate.jsonl` comes from `scripts/pass_rate.py`:
- Qwen3-1.7B base, vLLM, training decoding, 8 samples per question;
- 6,638 questions (empty-gold already dropped);
- 78 min at 0.70 s/question.

| correct of 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| questions | 2,214 | 299 | 208 | 172 | 173 | 177 | 267 | 392 | 2,736 |

Results:
- **1,688 questions (25.4%) are kept** for v2. This is above the ~1,500 stop threshold.
- 1,000 steps × 4 prompts is about 2.4 epochs over the kept set.
- 33% of training questions are never solved and 41% are always solved. Those 74% give no GRPO
  signal at the base policy. This matches the ~65% zero-variance groups seen in the main run's log.

## X6: voice EX vs SQLCoder (RQ2, RQ4)

**Data.** The data is the 100 SpokenSpider queries from voice-to-sql, each with ASR transcripts at 4 SNR
conditions: `data/spokenspider_transcripts.csv`, a slim copy of voice-to-sql's `data/metadata.csv`.
- All 100 are Spider *train* questions on 3 databases (student_assessment 44, farm 40,
  department_management 16).
- All 100 are in `data/spokenspider_exclude.json`, so none was trained on.

**Method** (`scripts/eval_voice.py`):
- Our models get each transcript in the same prompt and decoding as Spider dev (greedy, 4-bit).
- "text" is the original typed question (no ASR).
- SQLCoder-7B's stored SQL (voice-to-sql pipeline, including its repair step) is re-scored with
  **our** comparator (`tinysql.evaluate.score_sql`).
  - Our comparator gives SQLCoder 2 more matches per condition than voice-to-sql's own matcher (52 vs 50
    clean). It tolerates column permutation.
  - SQLCoder only ever saw transcripts, so its "text" number is its clean-transcript result.

Sources: `results/voice_Qwen__Qwen3-1.7B_v2-checkpoint-900.json`, `results/voice_Qwen__Qwen3-1.7B.json`.

| Condition | mean WER | Qwen3-1.7B base | **v2-900** | SQLCoder-7B | v2 vs SQLCoder (w / l, p) | v2 vs base (w / l, p) |
|---|---|---|---|---|---|---|
| text (no ASR) | — | 46% | **57%** | (52%) | 18 / 13, 0.47 | 15 / 4, 0.019 |
| clean | 1.3% | 43% | 52% | 52% | 12 / 12, 1.0 | 11 / 2, 0.022 |
| 20 dB | 1.7% | 43% | 55% | 50% | 15 / 10, 0.42 | 13 / 1, 0.002 |
| 10 dB | 4.5% | 43% | 52% | 48% | 15 / 11, 0.56 | 11 / 2, 0.022 |
| 0 dB | 16.9% | 27% | 38% | 44% | 5 / 11, 0.21 | 11 / 0, 0.001 |

The base model's format failures are 9–10% per condition; v2's are 0–2%.

### Reading

- **RQ2: on these 100 queries, the 1.7B GRPO model matches SQLCoder-7B at clean to 10 dB.**
  None of the per-condition differences is significant: 12–15 wins vs 10–12 losses. With n=100 on 3
  databases, "matches" means "no detectable difference", not "equal".
- **GRPO helps on voice too.** v2 beats its own base model in every condition, by +9 to +12pp
  (p ≤ 0.022 everywhere). The base model is significantly behind SQLCoder at 0 dB (4 / 21, p=0.001)
  and trails at every other condition.
- **RQ4: v2 degrades like its base, and directionally more than SQLCoder. The difference is not tested.**
  - Clean → 0 dB costs v2 14pp (1 gained / 15 lost, p=0.0005) and the base 16pp. SQLCoder loses 8pp
    (2 / 10, p=0.039).
  - The 6pp gap between the drops was not tested, and on n=100 it may be noise.
  - At 0 dB, SQLCoder is ahead (44 vs 38, p=0.21, not significant).
  - The questions v2 loses at 0 dB have a mean WER of 27% vs 17% overall. They are real ASR errors
    ("departments" → "apartments", "acting" → "action", "ID" → "unit"). SQLCoder's
    repair step and larger model recover some of these.
- **Even clean ASR costs our model 5pp vs typed text** (57 → 52), at 1.3% WER. 7 of the 9 flips are
  losses, and they come from lost formatting, not wrong words:
  - quoted, capitalised literals become lower case (`"Aliens"` → `aliens`, `"Village"` → `village`), so
    string filters no longer match;
  - "1000" becomes "1,000".
  - One flip had an identical question text: that is the batch-composition noise seen on dev.

  Value normalisation before the SQL stage, or case-insensitive literal matching, would target exactly this.
- **Caveats:**
  - These are Spider *train* questions. Our model never saw them, but SQLCoder-7B-2's training data is
    not documented here and may include Spider train, which would favour it.
  - SQLCoder's numbers include voice-to-sql's SQL-repair step, which our model does not get.

## X7: latency (RQ3)

**Setup** (`scripts/bench_latency.py`):
- Same GPU (RTX 3070 Ti Laptop, WSL2) for every system, one system per process.
- nf4 4-bit, HF `generate`, batch 1, greedy.
- 100 clean transcripts, after 5 untimed warm-up queries. Timed from prompt rendering to decoded SQL,
  with CUDA synchronised.
- SQLCoder uses voice-to-sql's prompt, schema format, quantisation (double-quant nf4), 300-token limit
  and SQL extraction, copied with a pointer to the source.
  - **Reproduction check:** it produced the same SQL that voice-to-sql stored for 93 of 100 transcripts.
- The end-to-end estimate takes voice-to-sql's recorded per-row total, removes its SQL stage and adds the latency measured here.

Sources: `results/latency_ours_Qwen__Qwen3-1.7B_v2-checkpoint-900.json`, `results/latency_ours_Qwen__Qwen3-1.7B.json`, `results/latency_sqlcoder.json`.

| System | Text-to-SQL median | p95 | output tokens (median) | ms / token | peak VRAM | End-to-end est. median (p95) |
|---|---|---|---|---|---|---|
| v2-900 (4-bit base + LoRA adapter, unmerged) | 2.22 s | 5.70 s | 31 | 71 | 1.5 GiB | 2.91 s (6.08) |
| Qwen3-1.7B, no adapter (lower bound for a merged v2) | 1.32 s | 2.73 s | 34 | 39 | 1.4 GiB | 1.98 s (3.36) |
| SQLCoder-7B (re-measured here) | 1.76 s | 3.36 s | 41.5 | 41 | 4.0 GiB | 2.34 s (4.14) |
| SQLCoder-7B (as recorded by voice-to-sql) | 13.78 s | 26.53 s | — | — | — | 14.30 s (27.23) |

### Reading

- **RQ3: on this serving stack, the smaller model gives no latency gain as deployed.** v2-900 is
  *slower* than SQLCoder-7B (2.22 vs 1.76 s median).
  - Measured: the unmerged LoRA adapter nearly doubles the time per token, from 39 ms (same model, no
    adapter) to 71 ms. That accounts for the gap.
- **Model size matters little here.** Per token, 4-bit Qwen3-1.7B (39 ms) and 4-bit SQLCoder-7B (41 ms)
  are almost equal.
  - Hypothesis, not measured: at batch 1 with bitsandbytes 4-bit, decoding is bound by per-layer
    kernel overhead rather than by weight size.
  - The only measured saving is memory: 1.4–1.5 GiB vs 4.0 GiB peak.
  - The no-adapter run suggests a merged 1.7B model might be ~25% faster (1.32 vs 1.76 s). This is
    unmeasured: merging and re-quantising change the weights, and possibly the kernel path.
- **Timing caveats (small, and both favour SQLCoder):**
  - Our timed path also re-tokenises the output to count tokens.
  - Our timed path builds the schema text on the first query of a database, if the 5 warm-ups did not
    cover it.
  - The systems ran one after another on a laptop GPU that reaches ~87 °C, so thermal drift is not
    controlled.
- **voice-to-sql's ~14 s SQL stage was not the model.** The same SQLCoder, with the same prompt, settings
  and outputs (93/100 identical SQL), takes 1.76 s here, about 8× faster.
  - The slowdown came from how voice-to-sql ran it, likely the transformers `pipeline`, `device_map="auto"`
    and the native-Windows runtime. This was not isolated here.
  - So the conclusion in its README ("SQLCoder dominates end-to-end latency") is about that runtime,
    not about 7B vs 1.7B.
  - With SQLCoder re-measured, the whole voice pipeline is ~2.3 s median, not ~14 s.
- **What would make the small model pay off:**
  - merging the adapter into the weights (then re-checking EX after re-quantisation);
  - serving with a runtime that removes per-token overhead (vLLM is already in this project's stack for training).
  Neither was measured; both are follow-ups, not results.

## Week 4 Y1: official Spider EX

**Method** (`scripts/official_eval.py`):
- The stored full-dev predictions are re-scored, with no regeneration.
- Scorer: Spider's official evaluation script (`third_party/test-suite-sql-eval`, commit e97acc5),
  `--etype exec`, on the standard Spider dev databases.
- Values are predicted (no `--plug_value`), and the official default for DISTINCT is used
  (no `--keep_distinct`, so DISTINCT is removed before execution).
- The test-suite (distilled) databases were not downloaded, so this is official **exec** EX, not test-suite EX.
- **Per-row reproduction:** each row is re-scored with the script's own `eval_exec_match`. The per-row
  totals reproduce the CLI's printed accuracy for all four models.
- **Run-time cap:** the official script's 60 s timeout never fires (it wraps a blocking sqlite call).
  So predictions that do not finish within 120 s are scored as wrong:
  - base: 1 row; main: 3; v2: 2; v2-binary: 3 (rows 459/460 on wta_1, 730 on world_1).
  - For base and v2, the uncapped runs (`results/official/uncapped/`, ~25 min per slow query) gave
    identical rows, so the cap changes no score.

Source: `results/official_ex.json`. Per-row files are in `results/official/`.

| Model | Ours | Official | easy / medium / hard / extra (official) |
|---|---|---|---|
| base | 53.7 | 54.1 | 73.4 / 61.9 / 32.8 / 26.5 |
| main-1000 | 57.1 | 58.1 | 82.3 / 64.6 / 35.6 / 28.3 |
| v2-900 | 58.5 | **59.0** | 82.7 / 63.7 / 42.5 / 28.3 |
| v2-binary-800 | 58.2 | 58.8 | 81.9 / 63.9 / 40.8 / 29.5 |

**Paired comparisons with the official metric:**

| Comparison | Wins / losses | p |
|---|---|---|
| v2 vs base | 84 / 33 | <0.0001 |
| main vs base | 58 / 16 | <0.0001 |
| v2 vs main | 39 / 30 | 0.34 |
| v2 vs v2-binary | 19 / 17 | 0.87 |

**Every week-3 conclusion holds under the official metric.** The v2-vs-main gap is, if anything,
weaker (p=0.34 vs 0.09).

**Why the two metrics disagree** (13–23 rows per model; all were classified):
- **Almost all of them come from the official default of removing DISTINCT.**
  - Removing it from `count(DISTINCT x)` turns the gold into `count(x)`, so a wrong `COUNT(*)`
    prediction is scored correct (official-only: 11–16 rows per model).
  - Removing it from a joined gold query brings duplicate rows back. A correct, duplicate-free
    prediction (e.g. an `IN` subquery) is then scored wrong (ours-only: 6–8 rows).
  - So the official default is the more lenient of the two here, by +0.4 to +1.0pp.
- **1 row per model is an error in our executor.** Question 455 (wta_1) returns names with invalid
  UTF-8, which our executor fails to decode, so it is unscorable for every model. Question 456 has the
  same problem, but there all models are wrong under both metrics. Since every model loses the same
  rows, the comparisons are unaffected. It is a known limitation of our EX (−0.1 to −0.2pp).
- Hardness view: v2's gain over main is mostly on *hard* questions (42.5 vs 35.6). Each hardness
  bucket is small (174 hard questions), so this is not tested.
