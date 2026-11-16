# W6 — checkpoint selection on dev-200 (main run, 1000 steps)

Same command as the baseline (greedy, 4-bit, batch 8, 256 new tokens); only `--adapter` differs.
Baseline: 55.0% (110/200). Noise floor from batch-composition nondeterminism: ~1.5pp.
Produced by `python scripts/compare_checkpoints.py`.

| ckpt | EX | Δ pp | format fail | hit 256 | win | loss | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 57.5 | +2.5 | 3.5% | 2.0% | 6 | 1 | 0.125 |
| 200 | 58.5 | +3.5 | 1.5% | 1.0% | 8 | 1 | 0.039 |
| 300 | 57.5 | +2.5 | 2.5% | 2.0% | 8 | 3 | 0.227 |
| 400 | 57.0 | +2.0 | 2.5% | 2.0% | 8 | 4 | 0.388 |
| 500 | 58.5 | +3.5 | 2.0% | 1.5% | 10 | 3 | 0.092 |
| 600 | 57.0 | +2.0 | 2.0% | 1.5% | 9 | 5 | 0.424 |
| 700 | 59.0 | +4.0 | 2.5% | 2.0% | 10 | 2 | 0.039 |
| 800 | 57.5 | +2.5 | 2.0% | 1.5% | 9 | 4 | 0.267 |
| 900 | 58.0 | +3.0 | 1.0% | 1.0% | 9 | 3 | 0.146 |
| 1000 | 59.0 | +4.0 | 2.0% | 2.0% | 11 | 3 | 0.057 |

## Reading

- Every checkpoint is above baseline (+2.0 to +4.0pp, all wins > losses), but the gain is already
  there at step 100 and then plateaus. Training reward was flat too (~0.57 → ~0.62 per-100-step mean),
  and 60–70% of groups had zero reward variance (no learning signal) throughout.
- **The gain looks mostly like format, not SQL** (only 11 wins, so this is indicative). Baseline: 13 format failures; ckpt-1000: 4. Of
  ckpt-1000's 11 wins, 8 were baseline format failures, 1 a baseline execution error, and only
  2 were baseline wrong-result answers. Execution errors are unchanged (21 → 20).
- No single checkpoint is significant after correcting for 10 comparisons (Bonferroni α = 0.005).
  dev-200 selects the checkpoint; the unbiased test is the full-dev evaluation in week 3.

## Selection

**checkpoint-1000**: it is the run's end state, so no early-stopping point is tuned on dev-200.
It ties with 700 at 59.0%, and the two are statistically indistinguishable
(700: 10 wins / 2 losses, p=0.039; 1000: 11/3, p=0.057).

## Implications for week 3

- Expect the full-dev gain to be small (~2–4pp) and likely format-driven; report it that way.
- The zero-variance groups are the bottleneck. Filtering training questions the base model
  always or never solves (pass@8 = 0 or 8) is the most likely lever, ahead of more steps.
