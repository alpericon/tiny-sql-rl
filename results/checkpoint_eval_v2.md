# X3 — checkpoint selection on dev-200 (v2 run: filtered data, partial reward, 1000 steps)

Same command as the baseline and W6 (greedy, 4-bit, batch 8, 256 new tokens); only `--adapter` differs.
Baseline: 55.0% (110/200). Noise floor: ~1.5pp.
Produced by `python scripts/compare_checkpoints.py v2`. Breakdowns come from `scripts/paired_compare.py`.

v2 trains on the 1,688 questions the base model solves 1–7 of 8 times (X2). Empty-gold questions are
dropped. All other settings match the week-2 main run.

| ckpt | EX | Δ pp | format fail | hit 256 | win | loss | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 58.5 | +3.5 | 2.5% | 2.0% | 7 | 0 | 0.016 |
| 200 | 58.5 | +3.5 | 2.5% | 2.0% | 8 | 1 | 0.039 |
| 300 | 59.5 | +4.5 | 2.0% | 1.5% | 10 | 1 | 0.012 |
| 400 | 60.0 | +5.0 | 2.0% | 1.5% | 15 | 5 | 0.041 |
| 500 | 60.0 | +5.0 | 2.5% | 2.0% | 15 | 5 | 0.041 |
| 600 | 59.5 | +4.5 | 1.5% | 1.0% | 15 | 6 | 0.078 |
| 700 | 59.0 | +4.0 | 2.0% | 1.5% | 14 | 6 | 0.115 |
| 800 | 58.5 | +3.5 | 1.5% | 1.0% | 14 | 7 | 0.189 |
| 900 | **60.0** | +5.0 | 2.0% | 1.5% | 15 | 5 | 0.041 |
| 1000 | 58.5 | +3.5 | 2.5% | 2.0% | 15 | 8 | 0.210 |

## Training signal (per-100-step means, `results/train_log_v2.csv` vs `train_log_main.csv`)

| steps | v2 reward | v2 zero-std | main reward | main zero-std |
|---|---:|---:|---:|---:|
| 1–100 | 0.576 | 0.25 | 0.568 | 0.71 |
| 401–500 | 0.690 | 0.37 | 0.607 | 0.68 |
| 901–1000 | 0.710 | 0.41 | 0.620 | 0.69 |

- **The filter did what X2 intended.** Groups with zero reward variance fell from ~0.65 to 0.25 at
  the start. They climb to ~0.41 as the policy learns to solve its 1,688 questions.
- **Training reward rises:** 0.58 → 0.71, where main stayed flat around 0.57–0.62. This is measured on
  a set seen ~2.4 times, so part of the rise can be memorisation.
- KL ended about 2× higher than main (0.049 vs 0.025). Mean completion length stayed ~40–44 tokens.
- Peak reserved VRAM was 5.75 GiB, and the GPU peaked at 87 °C, as in the main run.

## Reading

- The dev-200 gain is about the same size as main's, at +3.5 to +5.0pp vs +2.0 to +4.0pp for main.
  The extra training reward did not carry over to dev clearly.
- **Head to head, v2-900 vs main-1000 on dev-200: 60.0 vs 59.0, 8 wins / 6 losses, p=0.79.** This is
  within noise.
- **The wins have a different mix than main's.** For v2-900, 6 of 15 wins were baseline
  wrong-result answers, against 2 of 11 for main-1000. Format failures fixed: 8 of 15. Execution
  errors went 21 → 17. The trend points toward SQL rather than format gains, but n is small (indicative only).
- From step 400 on, losses grow (5 → 8 by step 1000) while wins stay at ~15. dev EX does not keep
  rising with training reward, which fits mild overfitting to the 1,688 questions.
- After correcting for 10 comparisons (Bonferroni α = 0.005), no checkpoint is significant against the base.

## Selection (rule fixed in PLAN-week3 before the run)

The rule is the best dev-200 EX, with ties going to the later step. 400, 500 and 900 tie at 60.0%,
so the choice is **checkpoint-900**. As noted in week3_results.md, selecting on dev-200 biases
dev-200 numbers upward. The full-dev evaluation (X5) is the unbiased test, and dev-200 is part of it.
