# Week 5 results: repair-turn GRPO (PLAN-week5)

## Training

v3 continues from v2-900 for 500 steps (`checkpoints/v3/run_args.json`).
- Data mix: 1,688 first-turn prompts (the v2 filtered set) and 1,688 of 2,281 repair-turn prompts,
  sampled from v2-900's own failures on train (`data/train_repair.jsonl`); 3 over-long repair prompts dropped.
- The KL reference is the base model, as for v2.
- Peak reserved VRAM 5.87 GiB; ~28 s/step.

Source: `results/train_log_v3.csv`, rewards logged per prompt type:

| steps | first-turn reward | repair-turn reward | zero-variance groups |
|---|---|---|---|
| 1–100 | 0.740 | 0.233 | 0.56 |
| 201–300 | 0.710 | 0.244 | 0.51 |
| 401–500 | 0.735 | 0.286 | 0.59 |

The repair-turn reward barely moves, and over half of the groups give no learning signal.

**Selection** (rule fixed in advance; `results/checkpoint_eval_repair.md`): best dev-200 EX with
repair, ties to the later step. **checkpoint-400** was chosen (65.0%, tied with 300).

## Results (full Spider dev, 1,034; same command as week 3 plus `--repair`)

These are the three pre-registered measures.

| Measure | v2-900 | **v3-400** | Paired test |
|---|---|---|---|
| **Primary: repair success on v2-900's 133 failed first attempts** (both models repair the same inputs) | 20 rescued (re-run control; 19 in the original run) | 23 rescued | 5 / 2, **p=0.45** |
| **Guard rail: first-attempt EX** | 58.5% | 59.3% | 32 / 24, p=0.35 (no loss) |
| **Headline: EX with repair**, ours | 60.3% | **62.0%** | 37 / 20, p=0.033 |
| **Headline: EX with repair**, official | 60.8% | **62.8%** | 39 / 19, **p=0.012** |

Sources:
- `results/repair_turn_v3-checkpoint-400.json` and `results/repair_turn_v2-checkpoint-900.json`;
- `results/baseline_Qwen__Qwen3-1.7B_v3-checkpoint-400_repair.json`;
- `results/official/v3-400-repair.json`;
- `results/official_ex.json`.

**v3-400's own retries:** 131 retries (on its own failures). 28 became correct, 23 became wrong
results and 80 still fail. 43 repeated the identical SQL, vs 53 of 133 for v2-900.

**Remaining failures:** 3 format, 77 execution error, 313 wrong result (v2-900 + repair: 4 / 85 / 321).

### Reading

1. **The primary hypothesis is not supported.** On identical inputs, v3's repair turn rescues 23 vs 20
   (p=0.45). Repeats of the failed SQL drop (45 vs 54 on those inputs), but it does not turn into
   significantly more correct answers.
   - The training log is consistent with this:
     - the repair-turn reward's 100-step means stayed between 0.23 and 0.31;
     - about 57% of all groups had no reward variance.
2. **v3-400 + repair is better than v2-900 + repair overall:** 62.8% vs 60.8% official (p=0.012). That
   is +17 correct answers (641 vs 624).
   - **This p-value is optimistic.** v3-400 was picked as the best of 5 checkpoints on dev-200, which
     is part of full dev, and there are several measures (3 pre-registered, plus voice).
   - On the 834 questions outside dev-200, the gain is 61.8% vs 60.1% (30 / 16, **p=0.054**).
   - Either way it should not be attributed to repair training, for two reasons:
   - about half of the gain is on first attempts (+8: 613 vs 605). The other half is repair on v3's own,
     different failures (28 rescued vs 19);
   - v3 is v2-900 plus 500 more GRPO steps, half on first-turn prompts. No control run did those
     steps without repair data. "More training" explains the gain as well as "repair data" does.
   - With one seed, this is a single observation.
3. **Hand read, deviating from the plan.** The plan asked for 20 top-reward repair samples from
   *training*. The trainer does not log rollouts, so the 28 repairs v3 rescued on dev were read instead,
   as a reward-hacking check (`results/repair_rescues_v3-400_dev.md`). This is a different sample.
   - Most are fixes that act on the error: adding the missing join, using the right column or table,
     fixing syntax.
   - 2 were format-only (the first answer had no ```sql block). A few change more than the reported
     error (e.g. rewriting a join, dropping a DISTINCT).
   - No degenerate or trivially-true queries were seen.

## Voice with repair (100 SpokenSpider queries per condition)

| Input | v2-900 + repair | v3-400 + repair | v3 vs v2 (w / l, p) | SQLCoder-7B | v3 vs SQLCoder (w / l, p) |
|---|---|---|---|---|---|
| typed question | 58% | **61%** | 5 / 2, 0.45 | (52%) | 21 / 12, 0.16 |
| clean | 55% | 54% | 4 / 5, 1.0 | 52% | 14 / 12, 0.85 |
| 20 dB | 57% | 53% | 2 / 6, 0.29 | 50% | 15 / 12, 0.70 |
| 10 dB | 54% | 50% | 2 / 6, 0.29 | 48% | 15 / 13, 0.85 |
| 0 dB | 39% | 36% | 2 / 5, 0.45 | 44% | 6 / 14, 0.12 |

Source: `results/voice_Qwen__Qwen3-1.7B_v3-checkpoint-400_repair.json`.

- **On ASR transcripts v3 is directionally worse than v2-900 in all four conditions.** No condition is
  significant.
  - Pooled, it is 10 / 22 (p=0.050). The same 100 queries repeat across conditions, though, so this
    pooled test overstates the evidence.
  - On typed questions it is better, consistent with its Spider dev gain.
- A possible reading, not tested: v3's extra training on clean Spider text (first-turn and repair
  prompts) traded some robustness to noisy input for accuracy on clean input.

## Conclusion

- **Repair-turn GRPO, at this data size and step count, did not make the model measurably better at
  using error feedback.** This is the question the week was designed to answer, and the answer is a
  null result.
- The resulting model is more accurate on Spider dev: official 62.8% with repair, p=0.012 vs v2-900,
  though only p=0.054 outside the selection subset. It is somewhat less robust on noisy speech
  (direction only). The cause of the gain is confounded with extra training.