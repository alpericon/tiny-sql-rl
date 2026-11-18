# X4 — checkpoint selection on dev-200 (v2-binary ablation: filtered data, binary reward, 1000 steps)

Same command as W6 and X3 (greedy, 4-bit, batch 8, 256 new tokens); only `--adapter` differs.
Baseline: 55.0% (110/200). Noise floor: ~1.5pp.
Produced by `python scripts/compare_checkpoints.py v2-binary`. Breakdowns come from `scripts/paired_compare.py`.

This run is identical to v2 (same 1,688 filtered questions, same seed and settings) except for the
reward. Here it is binary: 1 when the result matches, else 0. v2 also gave 0.1 for SQL that executes
but returns the wrong result.

| ckpt | EX | Δ pp | format fail | hit 256 | win | loss | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 58.5 | +3.5 | 1.5% | 1.0% | 8 | 1 | 0.039 |
| 200 | 58.5 | +3.5 | 2.0% | 1.5% | 8 | 1 | 0.039 |
| 300 | 58.5 | +3.5 | 2.5% | 2.0% | 9 | 2 | 0.065 |
| 400 | 58.5 | +3.5 | 2.0% | 1.5% | 12 | 5 | 0.143 |
| 500 | 59.0 | +4.0 | 2.5% | 2.0% | 14 | 6 | 0.115 |
| 600 | 58.0 | +3.0 | 2.5% | 2.0% | 12 | 6 | 0.238 |
| 700 | 59.0 | +4.0 | 1.5% | 1.0% | 15 | 7 | 0.134 |
| 800 | **59.0** | +4.0 | 2.0% | 1.5% | 13 | 5 | 0.096 |
| 900 | 58.0 | +3.0 | 2.0% | 1.5% | 14 | 8 | 0.286 |
| 1000 | 58.0 | +3.0 | 2.0% | 1.5% | 14 | 8 | 0.286 |

## Training signal (per-100-step means, `results/train_log_v2-binary.csv` vs `train_log_v2.csv`)

| steps | binary reward | binary zero-std | v2 reward | v2 zero-std |
|---|---:|---:|---:|---:|
| 1–100 | 0.552 | 0.30 | 0.576 | 0.25 |
| 401–500 | 0.672 | 0.42 | 0.690 | 0.37 |
| 901–1000 | 0.690 | 0.43 | 0.710 | 0.41 |

- Binary reward is the fraction of fully correct samples. v2's reward also includes the 0.1 partial
  credits, so the two columns are not on the same scale. Both rise by about 0.13 over the run.
- Zero-std starts higher with the binary reward: 0.30 vs 0.25. Without the 0.1 credit, a group of
  wrong-but-executable and failing samples has no variance.
- KL is 0.046 at the end (v2: 0.049), and peak reserved VRAM is 5.69 GiB. The GPU peaked at 87 °C.
  Steps were ~23 s vs ~19 s for v2, with similar completion lengths. The cause was not investigated;
  the reward function itself costs the same either way.

## Reading

- **The reward choice makes almost no difference on dev-200.** v2-binary-800 vs v2-900:
  59.0 vs 60.0, **1 win / 3 losses, p=0.63**. The two models disagree on only 4 of 200 questions.
- v2-binary-800 vs main-1000: 59.0 vs 59.0, 6 wins / 6 losses, p=1.0.
- vs base: +4.0pp, 13 wins / 5 losses, p=0.096. Of the 13 wins, 7 were baseline format failures,
  1 an execution error and 5 wrong-result answers. Execution errors are unchanged at 21.
  The partial reward's 21 → 17 drop in execution errors (v2-900) is the only visible difference, and
  at n=4 discordant pairs it is not evidence.
- Like v2, losses rise after step ~400 (5 → 8 by step 1000) while wins plateau.
- After correcting for 10 comparisons (Bonferroni α = 0.005), no checkpoint is significant against the base.

## Selection (rule fixed in PLAN-week3 before the run)

The rule is the best dev-200 EX, with ties going to the later step. 500, 700 and 800 tie at 59.0%,
so the choice is **checkpoint-800**. As for v2, selecting on dev-200 biases dev-200 numbers upward.
The full-dev evaluation (X5) is the test.
