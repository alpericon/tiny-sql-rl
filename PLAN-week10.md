# Week 10 plan: schema pruning (inference-only)

**Status: APPROVED 2026-11-20 (approach A). Done 2026-11-20: official EX 63.4 -> 65.9 (42/17, p=0.0015), decision rule met, demo prunes by default.**

## Why

Week 9 (`results/error_analysis_wrong_result.md`) labelled 100 of the released configuration's 301
wrong-result failures. **45% are schema linking or joins:** the model picks the wrong table or
column, or the wrong join path. Foreign keys are mostly already in the prompt: in 27 of the 29
schema/join errors whose gold query joins tables, every joined pair has a declared key. So the remaining
lever named in the week-9 plan is to show fewer tables.

**Headroom, measured on the same 100 rows before writing this plan:**
- In 22 of the 45 schema/join errors, and **33 of all 100**, the prediction reads a table that the gold
  query does not use. Pruning can remove that wrong option.
- In 26 of the 45, the prediction misses a table the gold needs. Pruning does not help these, and it can
  hurt if it drops a needed table.
- In 12 of the 45, the table set is right and a column is wrong. Pruning tables does not touch these.
- **Spider dev databases are small.** 16 of the 20 have 2–4 tables, and only 256 of the 1,034 questions
  are on databases with 5 or more. Expected gains are therefore modest. A null result is plausible and
  would be reported as such.

## Question

**Does showing only the relevant tables improve EX in the released configuration, without hurting
voice?** Latency is measured too, since shorter prompts should partly undo week 7's slowdown.

## Approaches considered

| | Approach | For | Against |
|---|---|---|---|
| **A (recommended)** | **Lexical pruning**: keep tables whose name, column names or stored values match words in the question, then add the bridge tables on foreign-key paths between kept tables. Fall back to the full schema when fewer than 2 tables match on a database with more than 2 tables. | Deterministic and fast, no extra model call. Reuses the value index from grounding. Can be tuned on **train** and frozen before dev is touched. | Misses paraphrases ("ticket expense" vs `Total_spent`). Recall decides everything. |
| B | Two-stage with the same model: first ask v2-900 which tables are needed, then generate SQL on that schema | Understands paraphrase | A prompt the model was never trained on; doubles the calls (worse on the CPU demo); its errors are the same schema-linking errors we are trying to fix |
| C | Embedding retrieval of tables | Paraphrase-robust | A new model dependency in the demo; overkill for 2–11 tables |

**Table level only.** Columns are never removed from a kept table: column pruning would risk the 12
"right table, wrong column" cases going the other way. Example rows (3 per kept table) stay as in
week 7.

**Repair turn:** the retry after an execution error uses the **full** schema. If pruning dropped a needed
table, the error (`no such table` / `no such column`) then gets one chance with everything visible.

## Method (fixed in advance)

1. **Tune on train only.** Measure **gold-table recall** on train questions: a question counts as kept
   if every table the gold query reads is in the pruned set. Use it to fix the matching rules (word
   normalisation, plural handling, the fallback threshold), then freeze them in code.
2. **Recall on dev, once, before any generation.** Report question-level recall, the average share of
   tables kept, and median prompt tokens.
   - **Stop rule:** if dev recall is below 95%, stop and decide with you before running the model.
     Every lost question is a likely loss.
3. **Evaluate** the frozen pruner in the released configuration (v2-900, nf4, greedy, rows, repair with
   full schema, grounding). The only change is the first-attempt prompt.

## Measures and tests

| Measure | Data | Test |
|---|---|---|
| **Primary: official EX**, pruned vs full schema | full dev 1,034 | paired McNemar vs week-7 records (`v2-900-repair-ground-rows3`, reused, not re-run) |
| Ours EX, first-attempt EX | full dev | paired McNemar |
| Effect on the 5+ table databases (256 questions) | full dev subset | wins / losses, descriptive |
| Error mix | full dev | wrong result / `no such column` / other exec / format |
| Voice EX per condition (guard rail) | 100 SpokenSpider × 5 | paired McNemar per condition |
| Cost | CPU demo path, 30 questions | median prompt tokens; cold and warm text-to-SQL latency (`scripts/bench_demo.py`) |

**Decision rule (fixed now):** pruning becomes the demo default **only if** the official-EX gain is
significant (p<0.05) **and** no voice condition is significantly worse. Otherwise the result is
reported and nothing changes. Latency is reported but does not decide, on its own, to change the
default.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| P1 | Pruner, test-first | `src/tinysql/prune.py` (`select_tables`, `gold_tables`), `serialize_schema(..., tables=...)`, `tests/test_prune.py` | Tests on a toy database: name, column and value matches keep a table; FK bridge tables are added; the fallback gives the full schema; `tables=None` reproduces today's schema byte for byte. |
| P2 | Train recall, tune, freeze | `scripts/prune_recall.py` → `results/prune_recall_train.json` | Recall and tables kept on train, per rule change, in a log. Rules frozen in a commit before P3. |
| P3 | Dev recall (once) | `results/prune_recall_dev.json` | Recall, tables kept, prompt tokens. **Stop rule** applied. |
| P4 | Plumbing, test-first | `evaluate(..., prune=True)` (repair turn with full schema), `--prune` in `eval_baseline.py`, `eval_voice.py`, `bench_demo.py`; demo `TINYSQL_PRUNE` (off by default) | Tests: pruned first-attempt prompt, full-schema repair prompt; `prune=False` gives identical records to week 7 on dev-200. |
| — | **Review gate** (P1–P4) | | Reviewer approves before the measurement. |
| P5 | Measure | `results/baseline_*_prune.*`, `results/voice_*_prune.*`, official EX, `results/demo_latency_*_prune*.json` | The measures table above. |
| P6 | Docs + decision | `results/week10_results.md`, README (+ `check_readme.py`) | Decision rule applied as written. |
| — | **Final review gate** | | Reviewer approves; README numbers verified. |

**Cost:** ~1.5 h code (P1, P4), ~30 min tuning (CPU). GPU: ~20 min full dev, ~10 min voice. CPU: ~10
min official EX, ~10 min latency.

## Risks

- **Low recall** loses questions that work today. Handled by the 95% stop rule and the full-schema
  repair turn.
- **Small headroom** (see Why). The likely size is a few points at most.
- **Prompt change again.** Week 7 showed that greedy answers move a lot when the prompt changes (112
  flips). Expect churn; the net count is the measurement.
- **Tuning leakage.** Rules are tuned on train only. Dev recall is computed once, after freezing.
- **Single run**, no seed variance at inference, as in weeks 6–7.

## Not in this plan

Column pruning, the two-stage model approach (B), self-consistency, BIRD, and the HF model card.
