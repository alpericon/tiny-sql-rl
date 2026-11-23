# Week 12 plan: does it transfer? One-shot evaluation on BIRD dev

**Status: APPROVED 2026-11-22, including the dev.zip download. Done 2026-11-23: GRPO (+3.1) and the inference stack (+6.6) both confirmed on BIRD dev, with and without formula_1.**

## Why

Everything so far is Spider: training, every decision and the held-out test (week 11). Spider schemas
are small, clean and named in plain words. **BIRD** is the other standard text-to-SQL benchmark:
- the databases are large and messy (real values, dirty data, cryptic column names);
- each question comes with expert "evidence", a hint of external knowledge such as
  "eligible free rate = `Free Meal Count` / `Enrollment`".

It tests whether the GRPO gain and the inference stack transfer beyond Spider. A 1.7B model will score
far lower on BIRD than on Spider. That is expected and is not the question here; the comparisons
between configurations are.

## Data (facts checked before writing)

- **BIRD dev:** 1,534 questions, with an `evidence` field, on SQLite databases; license CC BY-SA 4.0.
- **Download:**
  - **`dev.zip`, 346 MB** (`Content-Length` 346,207,293; last modified 2024-06-29), from
    `https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip`, the "Dev Set" link on the official page
    bird-bench.github.io. It contains a further `dev_databases.zip`.
  - Stored in the WSL home folder (`~/bird`), not in the repo: Windows C: has only 42 GB free, and
    SQLite over `/mnt/c` is slow.
- **Not used:**
  - The cleaner "bird-sql-dev-1106" split (Nov 2025, on Hugging Face) is newer. The original dev set
    is still the one most results are reported on, so it is used here. Mixing versions would make the
    numbers hard to compare.
  - Mini-Dev is not used either.
- **No overlap with our training:** BIRD databases are not Spider databases. This is checked after
  download (B1).

## Question

**On BIRD dev, does GRPO still beat zero-shot, and does the inference stack (repair, grounding, example
rows, pruning) still help?**

## Method (fixed in advance; run once, like week 11)

**Prompt:** the same as on Spider. The question is followed by the evidence, as `Question: <question>`
and a new line with `Evidence: <evidence>`, for every configuration. This is the standard BIRD setting
(the "oracle knowledge" column on the leaderboard). The model never saw an evidence line in training;
that is equally true for all configurations. Without-evidence runs are not in this plan.

**Configurations** (as released, nothing tuned on BIRD):

| | Configuration |
|---|---|
| A | Qwen3-1.7B zero-shot, schema-only |
| B | v2-900, schema-only |
| E | v2-900 + repair + grounding + 3 example rows + schema pruning (the demo configuration) |

**Metric: BIRD execution accuracy**, as BIRD defines it: the set of predicted result rows equals the
set of gold result rows. That is order-insensitive and duplicate-insensitive. Our comparator also
ignores column order and is multiset-based, so it differs slightly; BIRD's rule is the decision metric.
- It is implemented in `tinysql.bird` and checked against the official `bird/llm/src/evaluation.py`
  (read from the DAMO-ConvAI repository, B2).
- Each query has a 30 s execution timeout, and a timeout scores wrong. The cap is the same for
  every configuration, and the capped rows are reported.

**Pre-registered comparisons** (paired exact McNemar, BIRD EX):

| # | Comparison | Question |
|---|---|---|
| T1 | B vs A | does GRPO transfer? |
| T2 | E vs B | does the inference stack transfer? |
| T3 | E vs A | the whole configuration |

The labels are the same as in week 11, with Bonferroni over three tests (p < 0.0167):
- "confirmed": significant, with the same sign as on Spider;
- "directionally consistent": same sign, not significant;
- "not reproduced": the sign flips.

Also reported, descriptively, by BIRD difficulty (simple / moderate / challenging): pruner recall,
prompt lengths and error mix.

**One-shot rules:** each configuration runs once. Nothing in the model, the prompt format above, the
pruner or the demo changes because of BIRD results. A negative result is reported as such.

**Prompt length:** BIRD schemas are much wider than Spider's. Before generation, prompt lengths are
measured for A and E.
- Prompts are never truncated.
- **Stop rule:** if more than 5% of prompts exceed the 3,072-token guard, stop and decide with you.
  Options would be raising the guard (if memory allows) or reporting those questions separately.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| B1 | **Download (needs your approval)**, unzip, check | `scripts/download_bird.sh` (records the ETag and the sha256 of what was downloaded) → `~/bird/dev/`; `results/bird_split_check.json` | 1,534 questions, every database present, every gold query executes within the timeout (failures listed), no database name shared with Spider. |
| B2 | BIRD loader, evidence prompt and EX, test-first | `src/tinysql/bird.py`, `tests/test_bird.py` | Tests: the evidence line in the prompt; set-based EX (order and duplicates ignored); timeout scored wrong. Semantics compared line by line with the official evaluation.py (reading the file on GitHub; nothing is installed). |
| B3 | Evaluation script + pre-generation report | `scripts/eval_bird.py` (reuses `evaluate` with a BIRD scorer), `results/bird_prompt_report.json` | Prompt lengths for A and E, pruner recall on BIRD gold, and the stop rule applied. |
| — | **Review gate** (B1–B3) | | Reviewer approves before any generation. |
| B4 | Three runs | `results/bird_*` | A, B and E, once each. |
| B5 | Write-up + README + final review | `results/week12_results.md` | T1–T3 labelled, with numbers by difficulty. |

**Cost:** ~2 h code. Download ~350 MB, with ~1–2 GB unzipped in WSL. GPU: ~3–5 h for three runs on
1,534 questions (BIRD queries on large databases are slower to execute). No training.

## Risks

- **Very low scores** (a 1.7B model on BIRD). Expected; the plan measures differences, not rank.
- **Long prompts** on wide schemas: handled by the stop rule. Pruning helps E, but A and B use the
  full schema.
- **Slow gold queries** on large tables: a 30 s cap, the same for all configurations, reported.
- **Evidence is off-distribution** for the model: this affects all configurations equally, but may
  lower absolute scores.
- **Grounding and pruner value scans** skip tables over 50,000 rows (existing behaviour). Some BIRD
  tables are larger, so these steps see fewer values there.
- **Single run, single seed.**

## Decision after B1 (2026-11-22, user, before any generation)

BIRD's `formula_1` database (174 dev questions) has the same 13-table schema as Spider's `formula_1`.
Spider train has 80 questions on it, and 11 of them are in v2's training set. No question text is
shared (`results/bird_split_check.json`). **Decision: the primary analysis stays the full 1,534
questions.** T1–T3 are also reported, with the same labels, on the 1,360 questions outside `formula_1`.
This is a secondary analysis, registered here before any BIRD generation.

Also from B1: 2 gold queries (question_id 518, 701) do not finish within 30 s on the WSL filesystem.
They count as wrong for every configuration and are listed in the write-up.

## Decision at the B3 stop rule (2026-11-22, user, before any generation)

19.9% of the full-configuration prompts exceed 3,072 tokens (longest 4,547), all on card_games and
european_football_2. The cause is wide tables, not long cells (`results/bird_prompt_report.json`).
**Decision: raise the prompt guard to 5,000 tokens for all three configurations.**
- The extra room is for repair prompts, which add the failed answer and the error message.
- Configuration E stays exactly the demo configuration.
- A and B are unaffected: their longest prompt is 2,199.
- GPU memory at batch 8 is checked first with synthetic prompts. If it does not fit, only the batch size
  is lowered, and that is reported.
- **Memory check** (8 synthetic prompts of 4,757 tokens, v2-900, nf4; no BIRD data):
  - batch 8 needs 8.96 GiB allocated (10.77 reserved), more than the 8 GiB card;
  - batch 4 needs 5.18 GiB (6.13 reserved);
  - batch 2 needs 3.28 GiB.
- **All three configurations therefore run at batch 4** (`--batch-size 4 --max-prompt-tokens 5000`).
  Using the same batch size everywhere avoids batch-size effects between them. Spider runs used batch 8.

## Change after the B1–B3 review (before any generation)

The guard is raised again, from 5,000 to **5,500 tokens**, for all three configurations.
- E's repair turn uses the full schema with rows, plus the failed answer and the error. That reaches
  about 4,900 tokens, which left only ~100 tokens of margin, and a breach would stop the whole run.
- The guard only refuses prompts; it changes no prompt.
- Memory at batch 4 extrapolates to ~6 GiB.

## Not in this plan

Training on BIRD, runs without evidence, the newer dev split, Mini-Dev, leaderboard submission, and
voice on BIRD.
