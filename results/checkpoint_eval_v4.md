# R5: checkpoint selection on dev-200 (v4 run: v2 recipe with 3 example rows in the prompt)

Protocol (fixed in PLAN-week8): greedy, nf4, first attempt only (no repair, no grounding), 3 example
rows per table in the prompt (`eval_baseline.py --subset data/dev200_ids.json --sample-rows 3`).
The baseline is v2-900 evaluated the same way: **60.0% (120/200)**. Without rows, v2-900 also scores
60.0% on dev-200.
Produced by `python scripts/compare_checkpoints.py v4 --suffix _rows3 --base dev200_Qwen__Qwen3-1.7B_v2-checkpoint-900_rows3.records.jsonl`.

| ckpt | EX | Δ pp vs v2-900 | format fail | hit 256 | win | loss | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 52.5 | −7.5 | 4.5% | 0.5% | 1 | 16 | <0.001 |
| 200 | 54.5 | −5.5 | 3.5% | 0.5% | 1 | 12 | 0.003 |
| 300 | 57.0 | −3.0 | 3.0% | 0.5% | 2 | 8 | 0.109 |
| 400 | 57.0 | −3.0 | 2.0% | 0.0% | 2 | 8 | 0.109 |
| 500 | 58.0 | −2.0 | 1.5% | 0.0% | 4 | 8 | 0.388 |
| 600 | 60.0 | 0.0 | 1.0% | 0.5% | 4 | 4 | 1.000 |
| 700 | 60.0 | 0.0 | 1.5% | 0.0% | 5 | 5 | 1.000 |
| 800 | 58.0 | −2.0 | 1.5% | 0.0% | 2 | 6 | 0.289 |
| 900 | **60.0** | 0.0 | 1.5% | 1.0% | 4 | 4 | 1.000 |
| 1000 | 59.0 | −1.0 | 1.0% | 0.5% | 2 | 4 | 0.688 |

## Training signal (per-100-step means, `results/train_log_v4.csv` vs `train_log_v2.csv`)

| steps | v4 reward | v2 reward | v4 zero-std | v2 zero-std | v4 KL | v2 KL |
|---|---:|---:|---:|---:|---:|---:|
| 1–100 | 0.591 | 0.576 | 0.43 | 0.25 | 0.002 | 0.011 |
| 401–500 | 0.689 | 0.690 | 0.51 | 0.37 | 0.040 | 0.043 |
| 901–1000 | 0.685 | 0.710 | 0.49 | 0.41 | 0.045 | 0.049 |

- **Less learning signal than v2.** Groups with zero reward variance are higher throughout (0.43–0.51
  vs 0.25–0.41). The pass-rate filter was computed on schema-only prompts (kept on purpose, see the
  plan), so with rows more of the 1,688 questions are all-solved or all-failed.
- Run: 1,000 steps in 10.75 h, median 38 s/step, peak reserved VRAM 8.04 GiB, no OOM.

## Reading

- **No v4 checkpoint beats v2-900 + rows on dev-200.** The best ones tie (60.0%, 4–5 wins and losses
  each). Early checkpoints are clearly worse, as expected for 100–200 steps against 900.
- dev-200 is small (the noise floor is ~1.5 points), and the primary test is full dev in the release
  configuration (rows + repair + grounding), run once on the selected checkpoint.

## Selection (rule fixed in PLAN-week8 before the run)

Best dev-200 EX, ties to the later step: 600, 700 and 900 tie at 60.0%, so **checkpoint-900**.
