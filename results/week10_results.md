# Week 10 results: schema pruning (PLAN-week10, inference-only)

**Setup.** This is the released configuration (v2-900, nf4, greedy, 3 example rows, repair,
grounding) with one change: the first attempt shows only the tables a lexical pruner keeps
(`tinysql.prune.select_tables`). The repair turn shows the full schema again. Model weights are
unchanged.

- **Pruner** (`src/tinysql/prune.py`):
  - A table is kept when the question mentions its name, one of its less common column words, or one
    of its stored string values.
  - Tables on the foreign-key path between kept tables are added as bridges.
  - When fewer than 2 tables match, the full schema is used.
- **Tuning on train only** (`results/prune_recall_train.json`, 6,890 questions):
  - The rule was fixed before the final grid: among settings with train gold-table recall ≥ 98%,
    take the one showing the fewest tables.
  - Chosen: `min_tables=2, max_col_df=3`, prefix matching. Train recall 98.3%, 72.5% of tables shown.
  - Prefix matching and the neighbour option were added after reading train misses, which were
    paraphrases such as "followers" vs `follows`.
- **Dev recall, measured once after freezing** (`results/prune_recall_dev.json`):
  - **98.9%**, above the 95% stop line. The reviewer reproduced it independently, with gold tables
    taken from SQLite's authorizer.
  - 480 of 1,034 questions get fewer tables, and 11 lose a gold table (9 of them on car_1).
  - Median prompt 647 → 529 tokens.

## Primary: Spider dev, 1,034

| Metric | full schema (week 7) | **pruned** | w / l, p |
|---|---|---|---|
| **Official EX (decision metric)** | 63.4% | **65.9%** | 42 / 17, **p=0.0015** |
| Ours | 62.2% | 64.1% | 39 / 19, p=0.012 |
| First attempt, ours | 59.9% | 61.5% | 40 / 23, p=0.043 |

Sources: `results/baseline_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3_prune.json`,
`results/official/v2-900-repair-ground-rows3-prune.json`, `results/official_ex.json`. The full-schema
side is the week-7 run, reused, not re-run.

**Where the gain comes from:**
- **Questions that got fewer tables** (480): 37 wins, 18 losses. **The rest** (554, full schema
  either way): 2 wins, 1 loss, i.e. noise.
- **Databases with 5+ tables** (256 questions): 11 / 9. Most of the gain is on 3–4 table databases.
  This differs from the week-9 expectation, which pointed at larger schemas (descriptive, not tested).
- 58 questions change outcome; adding example rows in week 7 changed 112.

**Error mix, failures:**

| | wrong result | `no such column` | other exec error | format |
|---|---|---|---|---|
| full schema | 318 | 60 | 12 | 1 |
| pruned | 310 | **47** | 12 | 2 |

- Fewer `no such column` errors (60 → 47) and slightly fewer wrong results (318 → 310).
- Repair: 103 retries, 27 rescued (full schema: 111 and 24). The retry uses the full schema.

## Guard rail: voice (100 SpokenSpider queries per condition)

| Input | full schema | **pruned** | w / l, p |
|---|---|---|---|
| typed question | 66% | **70%** | 9 / 5, 0.42 |
| clean | 65% | **68%** | 10 / 7, 0.63 |
| 20 dB | 65% | **67%** | 10 / 8, 0.82 |
| 10 dB | 61% | **65%** | 10 / 6, 0.45 |
| 0 dB | 46% | 46% | 5 / 5, 1.0 |

Source: `results/voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground_rows3_prune.json`.

- **No condition gets worse.** Four are higher and 0 dB is level; none is significant on its own.
- Pruner recall on ASR transcripts was not measured separately. Only text dev recall had a stop rule.

## Cost: CPU demo path (Q8_0, 2 threads, 30 questions, same session)

| | full schema | pruned |
|---|---|---|
| text-to-SQL, cold (median) | 9.1 s | **6.8 s** |
| text-to-SQL, warm follow-up on the same database (median) | 2.2 s | 2.6 s |

Sources: `results/demo_latency_w10_rows3*_now.json` (full schema) and `results/demo_latency_w10_rows3_prune*.json`.

- **Cold first questions are ~26% faster**, because the prompts are shorter. This recovers about half of
  week 7's slowdown (5.0 → 9.3 s).
- **Warm follow-ups are ~0.4 s slower.** The schema now changes from question to question, so the cached
  prompt prefix is reused less.
- **All 30 questions answered without error** in every run.
- **Unexplained:** on the GPU evaluation (HF generate, batch 8), total generation time roughly doubled
  (2,577 s vs ~1,234 s for comparable runs). The cause was not investigated. It does not show up on the
  CPU demo path, which runs one question at a time.

## Decision (rule fixed in the plan)

The rule: pruning becomes the demo default only if the official-EX gain is significant (p<0.05)
**and** no voice condition is significantly worse.
- **Both hold** (p=0.0015; no voice condition worse).
- **The demo now prunes by default** (`TINYSQL_PRUNE` in `demo/app.py`; 0 turns it off). If the pruner
  raises, the demo falls back to the full schema.
- The model weights are unchanged.

## Caveats

- **Single run, no seed variance at inference.** The prompt changed for 480 questions; the net count
  is the measurement.
- **Tuned on Spider train, measured on Spider dev.** The pruner relies on lexical overlap between
  questions and schema names, which Spider has a lot of. It may do worse on schemas with cryptic names.
- **Known limits** (review, LOW):
  - The gold-table regex misses comma joins and quoted names with spaces. This has no effect on dev:
    the reviewer found the regex set equal to the true set on all 1,034 questions. Train recall was not
    re-checked the same way.
  - Prefix matching can over-match (e.g. "count" vs `country`). That only weakens pruning.
  - Views are never kept. Spider has none.
