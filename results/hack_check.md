# W7 — reward-hacking inspection (main run, checkpoint-1000)

## Method

The trainer did not log completions, so rollouts were re-created: `scripts/sample_policy.py`
sampled 8 completions for each of 100 random training prompts (seed 3407) from checkpoint-1000 with
the training decoding (temperature 1.0, no top-p/top-k, 256 tokens, thinking off). Each one is scored
with the training reward → `results/hack_samples.jsonl`.

Rewards: 480 × 1.0, 259 × 0.1, 61 × 0.0. All "highest-reward" samples tie at 1.0, so
`scripts/hack_check.py` ranks the reward-1.0 ones by suspicion:
- the gold result is empty or a single 0/1;
- the prediction has no FROM;
- the prediction reads different tables;
- ties are broken by the SQL length gap.

The top 20 (deduplicated) were read by hand → `results/hack_check_top20.txt`.

## Verdicts on the 20

| # | Verdict |
|---|---|
| 1 | **Wrong, rewarded.** Filters project *code* by project *name* → empty = empty gold. |
| 2–6, 9–10 | **Wrong, rewarded.** college_2 prerequisite question: compares course_id to a title, invents filters, or takes the course instead of its prerequisite. All return nothing, as does the gold. |
| 7 | **Wrong, rewarded.** Picks the alphabetically first district instead of the most populous. |
| 12–14 | **Wrong, rewarded.** Wrong date format ('Sep 21, 2010'), and first/last name swapped in 13–14. |
| 15–16 | Correct by intent (`IS NULL`), but the gold is buggy (`= "null"`); both are empty. |
| 11, 20 | Correct. |
| 8, 17–19 | Correct result, with redundant filters (`IS NOT NULL`, `> 0`) that do not change it on this DB. |

**12 of 20 are wrong queries that earned full reward, and every one of them is the same
exploit: the gold result is empty, so any query that returns nothing matches it.** They come from only 5 distinct prompts; 7 of the 12 are the same college_2 question.

No other exploit was found among the flagged candidates. This is not an exhaustive search: the
ranking puts empty-gold cases first, and the ~430 unflagged full-reward samples were not read.
Across all 480 full-reward samples:
- no literal-only or FROM-less queries;
- no matches on a single 0/1 gold answer;
- no truncation tricks;
- 13 read different tables than the gold query. The ones read by hand were correct or empty-gold.

## How big is it

- 252 of 6,890 training examples (3.7%) have an empty gold result. In the sample, 32 of 480
  full rewards (6.7%) came from them.
- No evidence that the policy *learned* to exploit it. The evidence is indirect, and no base-model
  rollout sample exists to compare the 6.7% against:
  - training reward stayed flat;
  - the redundant-filter habit did not grow (dev-200 greedy, mean `AND` per query: 0.23 base → 0.15 ckpt-1000;
    `IS NOT NULL`: 3 → 1).
- It is still a wrong gradient on ~4% of prompts, so it is fixed before week 3.

## Fix

`build_grpo_records(..., drop_empty_gold=True)` skips examples whose gold query returns no rows
(test: `test_empty_gold_results_can_be_dropped`). `train_grpo.py` now drops them by default;
`--keep-empty-gold` reproduces the week-2 main run. Evaluation is unchanged: Spider EX keeps
empty-gold dev questions, so baselines stay comparable.

## Side observation (not a hack)

70 of 100 prompts gave all 8 samples the same reward (no learning signal), matching the
~65% zero-std fraction in the training log. This, not reward hacking, explains the flat curve.
