# Week 8 results: training with example rows in the prompt (PLAN-week8)

**Setup.** `v4` is the v2 recipe run again from the base model. The only change is the training
prompt: 3 example rows per table (`--sample-rows 3`), as the demo uses since week 7. Everything else
is the same as v2: the same 1,688 pass-rate-filtered questions, reward, LoRA, LR, β, seed and 1,000
steps.
- **58 of the 1,688 prompts use the schema-only prompt.** With rows they exceed 3,000 tokens
  (baseball_1, sakila_1, cre_Drama_Workshop_Groups; longest 6,239). This was the user's decision at the
  R2 stop rule. No dev prompt is that long (longest 1,897), so on dev every prompt has rows.
- **Run:** 1,000 steps in 10.75 h, median 38 s/step, peak reserved VRAM 8.04 GiB, no OOM.
- **Selection** (rule fixed in the plan): best dev-200 EX with rows, ties to the later step →
  checkpoint-900. See `results/checkpoint_eval_v4.md`.

The comparison is v4-900 against v2-900. Both are evaluated with rows, repair and grounding, so they
differ only in the prompt used during training. v2-900's week-7 records are reused, not re-run.

## Primary: Spider dev, 1,034

| Metric | v2-900 + rows | **v4-900 + rows** | w / l, p |
|---|---|---|---|
| **Official EX (decision metric)** | 63.4% | **63.9%** | 29 / 24, **p=0.58** |
| Ours | 62.2% | 62.8% | 28 / 22, p=0.48 |
| First attempt, ours | 59.9% | 60.7% | 28 / 19, p=0.24 |

Sources: `results/baseline_Qwen__Qwen3-1.7B_v4-checkpoint-900_repair_ground_rows3.json`,
`results/official/v4-900-repair-ground-rows3.json` and `results/official_ex.json`.

**Not significant.** The direction is positive on both metrics (+0.5 official, +0.6 ours), but 29
against 24 is well within chance.

**Error mix, failures:**

| | wrong result | `no such column` | other exec error | format |
|---|---|---|---|---|
| v2-900 + rows | 318 | 60 | 12 | 1 |
| v4-900 + rows | 313 | 55 | 15 | 2 |

- Small changes everywhere. `no such column` drops a little further (60 → 55).
- Repair: 112 retries, 21 rescued (v2-900: 111 and 24). Grounding fires on 18 queries (v2-900: 16).
- **Churn is lower than in week 7.** 50 questions change outcome between v2-900 and v4-900 (both with
  rows). Adding rows to v2-900 changed 112.

## Secondary: schema-only prompt (rows turned off)

| Metric | v2-900 | v4-900 | w / l, p |
|---|---|---|---|
| Official EX | 60.9% | 62.0% | 37 / 26, p=0.21 |
| Ours | 60.4% | 61.5% | 35 / 24, p=0.19 |

- **v4 does not lose the schema-only skill.** Without rows it is directionally better than v2-900 (not
  significant), so `TINYSQL_SAMPLE_ROWS=0` would still work with v4.
- **The gain from rows at inference is smaller for v4:** 62.0 → 63.9 official, against 60.9 → 63.4 for
  v2-900. Training with rows did not make the model use them more.

## Guard rail: voice (100 SpokenSpider queries per condition, rows + repair + grounding)

| Input | v2-900 | **v4-900** | w / l, p | SQLCoder-7B (grounded) | v4 vs SQLCoder (w / l, p) |
|---|---|---|---|---|---|
| typed question | 66% | **72%** | 8 / 2, 0.11 | (55%) | 22 / 5, 0.002 |
| clean | 65% | **70%** | 7 / 2, 0.18 | 55% | 19 / 4, 0.003 |
| 20 dB | 65% | **69%** | 8 / 4, 0.39 | 53% | 20 / 4, 0.002 |
| 10 dB | 61% | **70%** | 11 / 2, **0.022** | 51% | 22 / 3, <0.001 |
| 0 dB | 46% | **49%** | 4 / 1, 0.38 | 45% | 10 / 6, 0.45 |

Source: `results/voice_Qwen__Qwen3-1.7B_v4-checkpoint-900_repair_ground_rows3.json`.

- **No voice condition gets worse**, so the guard rail holds. All five are higher (+3 to +9).
- 10 dB reaches p=0.022. That is one test of five; with a Bonferroni correction (α = 0.01) it is not
  significant. Read it as directional.
- Against SQLCoder-7B, v4 is significantly ahead in four of five conditions and level at 0 dB.
- Summed over the five conditions, v4 has 38 wins and 11 losses against v2-900. The plan tests each
  condition separately and fixes no pooled test, and the conditions share the same 100 questions, so
  this sum is descriptive only.

## Reward hacking

No new hack (`results/hack_check_v4.md`). Rewarded wrong answers come from empty or buggy gold, as in
W7. No sample copies example-row values in place of values from the question. On inn_1 the model wrote
the database's real date format, seen in the example rows, which a buggy gold query then contradicts.

## Decision (rule fixed in the plan)

The rule: v4 replaces v2-900 only if the official-EX gain is significant (p<0.05) **and** no voice
condition is significantly worse.
- **The first condition fails (p=0.58), so v2-900 stays the release.** The demo is unchanged: v2-900
  with 3 example rows. The second condition holds.
- **The primary result is null**, as in weeks 3 and 5. Training with example rows gives no measurable
  gain on Spider dev over using them at inference only.

## Caveats

- **Less learning signal than v2.** Zero-variance groups ran at 0.43–0.51 against v2's 0.25–0.41
  (`results/checkpoint_eval_v4.md`). The pass-rate filter was computed on schema-only prompts and was
  kept so that the data stayed the same. A filter recomputed with rows might give a stronger run, but
  that changes the data as well as the prompt.
- **Single seed.** McNemar covers question sampling, not training variance. Week 3 showed that two
  1,000-step runs can differ by ~1–1.5 points by chance, which is larger than the gain here.
- **The voice gains are consistent but small-sample** (100 questions per condition, one significant
  at p<0.05 before correction). They are not enough to release by the rule fixed in advance.
