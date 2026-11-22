# Week 11 results: one-shot confirmation on the Spider test set (PLAN-week11)

**Setup.** The Spider 1.0 test set was released on 02/05/2024. It is 2,147 questions on 40 databases,
none of which appear in train or dev (`results/test_split_check.json`). No setting was tuned on it.

Five configurations ran once each, with the exact settings used on dev (nf4, greedy, 256 new tokens):
- A: Qwen3-1.7B zero-shot;
- B: v2-900;
- C: B + repair + grounding;
- D: C + 3 example rows;
- E: D + schema pruning.

The decision metric is official EX (`results/official_ex_test.json`, `results/official_test/`). No
prediction hit the 120 s cap.

Four comparisons were registered before the runs. Their labels:
- **confirmed**: p < 0.0125 (Bonferroni over 4) and the same sign as on dev;
- **directionally consistent**: same sign, not significant;
- **not reproduced**: the sign flips.

## Results

| | Configuration | Official EX, **test** | Official EX, dev | Ours, test |
|---|---|---|---|---|
| A | zero-shot | 56.3% | 54.1% | 55.3% |
| B | v2-900 (GRPO) | 61.7% | 59.0% | 60.6% |
| C | + repair + grounding | 64.2% | 60.9% | 63.1% |
| D | + 3 example rows | 65.7% | 63.4% | 65.2% |
| E | + schema pruning (**the demo**) | **66.3%** | 65.9% | 66.2% |

| # | Comparison | Test: Δ, w / l, p (official) | Dev: Δ, w / l, p (official) | Label |
|---|---|---|---|---|
| T1 | GRPO (B vs A) | +5.4, 160 / 44, p<0.0001 | +4.9, 84 / 33, p<0.0001 | **confirmed** |
| T2 | example rows (D vs C) | +1.5, 166 / 134, p=0.073 | +2.5, 72 / 46, p=0.021 | **directionally consistent** |
| T3 | schema pruning (E vs D) | +0.6, 67 / 54, p=0.28 | +2.4, 42 / 17, p=0.0015 | **directionally consistent** |
| T4 | whole stack (E vs A) | +10.1, 308 / 92, p<0.0001 | +11.8 | **confirmed** |

On our own metric the signs are the same: T1 158 / 45, T2 173 / 128 (p=0.011), T3 76 / 54 (p=0.065),
T4 324 / 91. The labels follow the pre-registered official metric.

## Reading

- **The main claims hold on unseen databases.**
  - GRPO gains 5.4 points on test, against 4.9 on dev.
  - The full configuration is 10.1 points above zero-shot (66.3% vs 56.3%).
  - Every individual step has the same sign as on dev.
- **The two latest inference steps gain less on test than on dev.**
  - **Example rows:** +1.5 instead of +2.5. Official p is 0.073. Our metric happens to fall under the
    Bonferroni line (p=0.011), but the label follows the official metric.
  - **Schema pruning:** +0.6 instead of +2.4, not significant.
  - Each step was adopted because it was significant on dev, so dev overstates it. That is the
    selection effect this test was meant to measure. **Pruning's gain is much smaller on test than on
    dev.** The dev-vs-test difference itself was not tested, and there is one run per configuration.
- **Repair + grounding (C vs B, not one of the four tests):** +2.6 on test (61.7 → 64.2) against +1.9
  on dev. Descriptive only.
- **Pruning by hardness** (official EX, D → E): easy 83.2 → 84.3, medium 74.1 → 75.1, hard 49.9 → 51.0,
  extra 43.1 → 41.5. Mixed, and within noise.

## What changes

- **Per the plan, nothing in the demo or the release changes.** The demo keeps example rows and
  pruning. Both are still in the right direction on test, and pruning still roughly halves the cold
  CPU latency added by rows (week 10).
- **The README's headline numbers should now be the test numbers**, with dev kept for the history of
  decisions: **66.3% official EX on held-out Spider test for the demo configuration, vs 56.3% zero-shot.**

## Notes and corrections

- **H1 said 2 gold queries fail to execute. That was wrong.** The two aan_1 queries (test 1021 and 1022)
  took 6.0 s and 5.8 s when re-timed over `/mnt/c` and 0.2 s on the WSL filesystem (commit b57ae81).
  They only exceeded the check script's 5 s default timeout. The evaluation allows 30 s, and our
  comparator scores the gold query against itself as a match. The official evaluator reads them from
  the WSL copy. All five configurations score these two rows wrong, because the predictions differ
  from the gold, not because of a timeout.
  `results/test_split_check.json` is left as committed; this note corrects it.
- The overlap check in `scripts/test_split_check.py` covered train_spider.json and dev. The reviewer
  also checked train_others.json (1,659 questions): no shared databases or questions.
- The frozen pruner's recall on test was 98.5% (`results/prune_recall_test.json`), close to dev's 98.9%.
  The lower test gain is therefore not explained by more missed tables.
- Run times were longer than on dev: C took 61 min and E 38 min for 2,147 questions
  (`results/w11_test_runs.log`).
- The plan listed `prune_recall.py --split test` for H3. A separate `scripts/test_prompt_report.py` was
  written instead (reporting only).
- **Single run, single seed**, as everywhere in this project.
