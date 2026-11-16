# Week 3 results (PLAN-week3)

All EX numbers come from the same command:
- greedy decoding, nf4 4-bit, batch 8, 256 new tokens, thinking off;
- column-permutation-tolerant multiset EX (SPEC §12).

Paired tests are exact two-sided McNemar on the same questions (`scripts/paired_compare.py`).

## RQ1: GRPO vs zero-shot, full Spider dev (1,034)

| Model | EX | Format fail | Exec error | Wrong result | vs base: wins / losses | p |
|---|---|---|---|---|---|---|
| Qwen3-1.7B zero-shot | 53.7% | 51 | 130 | 298 | | |
| + GRPO main (week 2), ckpt-1000 | **57.1%** | 12 | 119 | 313 | 54 / 19 | 0.0001 |

Sources:
- `results/baseline_Qwen__Qwen3-1.7B.json`
- `results/baseline_Qwen__Qwen3-1.7B_main-checkpoint-1000.json`

**+3.4pp, significant.** The 54 wins break down by the base model's outcome on the same question:
- 29 were format failures;
- 10 were execution errors;
- 15 were wrong results.

So roughly half the gain is format and half is SQL. The dev-200 estimate (8 of 11 wins from format)
overstated the format share. Some fixed format failures became wrong results instead, which is why
wrong results rose from 298 to 313.

Rows for v2 and v2-binary follow in X5.
