# Week 7 results: example rows in the prompt (PLAN-week7)

**Setup.** The released configuration: v2-900, nf4, greedy, repair, grounding. The only change is the
prompt: after each `CREATE TABLE`, the first 3 rows of the table, as `-- Example rows from <table>:`
(`serialize_schema(sample_rows=3)`). k=3 is from SPEC; no other value was tried. No training: the
model never saw example rows during training.

- Prompts get longer: median 304 → 647 tokens, longest 930 → 1,897 (the longest prompt passed to the
  model was 2,168 tokens, under the 3,072-token guard).
- With `sample_rows=0`, all 1,034 prompts are byte-for-byte the week-6 length (`prompt_chars` equal).

## Primary: Spider dev, 1,034

| Metric | without rows | **with 3 rows** | w / l, p |
|---|---|---|---|
| **Official EX (decision metric)** | 60.9% | **63.4%** | 72 / 46, **p=0.021** |
| Ours | 60.4% | 62.2% | 65 / 47, p=0.11 |
| First attempt, ours | 58.6% | 59.9% | 65 / 52, p=0.27 |

Sources: `results/baseline_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3.json`,
`results/official/v2-900-repair-ground-rows3.json` and `results/official_ex.json`.

**The two metrics disagree on significance.** The pre-registered decision metric (official) gives
p=0.021, while our comparator gives p=0.11. The direction and size agree (+2.5 vs +1.8 points). Both
are reported, and the decision follows the rule as written.

**Error mix, failures with vs without rows:**

| | wrong result | `no such column` | other exec error | format |
|---|---|---|---|---|
| without rows | 320 | 74 | 11 | 4 |
| with rows | 318 | **60** | 12 | 1 |

- **The gain is mostly fewer schema errors:** `no such column` drops from 74 to 60. Wrong-result
  answers barely change.
- Grounding now fires on 16 queries instead of 34: with real values in the prompt, the model writes
  the right case more often by itself.
- Repair: 111 retries, 24 rescued (without rows: 133 and 19).
- **There is a lot of churn:** 112 questions change outcome in one direction or the other. On this
  off-distribution prompt, greedy answers move a lot. A single run, so the net gain is the
  measurement, not each flip.
- Run-time cap (official EX): row 483 joined the capped set and row 459 left it. Both are slow wta_1
  queries, and the capped prediction was wrong in our metric as well.

## Secondary: voice (100 SpokenSpider queries per condition)

| Input | without rows | **with 3 rows** | w / l, p | SQLCoder-7B (grounded) | with rows vs SQLCoder (w / l, p) |
|---|---|---|---|---|---|
| typed question | 58% | **66%** | 18 / 10, 0.19 | (55%) | 20 / 9, 0.06 |
| clean | 59% | **65%** | 15 / 9, 0.31 | 55% | 17 / 7, 0.06 |
| 20 dB | 60% | **65%** | 14 / 9, 0.41 | 53% | 18 / 6, 0.02 |
| 10 dB | 56% | **61%** | 14 / 9, 0.41 | 51% | 18 / 8, 0.08 |
| 0 dB | 40% | **46%** | 10 / 4, 0.18 | 45% | 8 / 7, 1.0 |

Source: `results/voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3.json`.

- **All five conditions improve** (+5 to +8 points). None is significant alone, and **none gets worse**.
- **Against SQLCoder-7B** (also grounded), the 1.7B model is now ahead in every condition:
  - including 0 dB, where it was behind before (46 vs 45);
  - 20 dB reaches p=0.02. That is one of five tests, so read it as directional, not as a confirmed win.

## Cost: CPU demo path (Q8_0, 2 threads, 30 questions)

| | without rows | with 3 rows |
|---|---|---|
| text-to-SQL, warm prefix cache (median) | 1.9 s | 2.8 s |
| text-to-SQL, cold (median) | 5.0 s | 9.3 s |
| end-to-end with speech, cold (median) | 6.5 s | 10.8 s |

Sources: `results/demo_latency_*_rows3*.json`, `results/demo_latency_none_t2*_now.json` (same session).

- **Cold first questions about double in latency**, but stay under the ~15 s demo limit.
- Follow-up questions on the same database cost ~1 s more.

## Decision (rule fixed in the plan)

The rule: the dev gain must be significant on the official metric (p<0.05), and no voice condition
may be significantly worse. **Both conditions hold**, so the demo now puts 3 example rows in the prompt
by default (`TINYSQL_SAMPLE_ROWS=3` in `demo/app.py`; 0 turns it off). The model weights are unchanged.

## Caveats

- **Single run, no seed variance for inference.** The 112-question churn shows how sensitive the model
  is to this prompt change.
- **Our own metric does not reach significance** (p=0.11). The decision rests on the pre-registered
  official metric.
- **The model was not trained with example rows.** Training with them (a separate plan) might gain
  more, or remove the churn.
