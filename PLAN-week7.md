# Week 7 plan: example rows in the prompt (schema grounding, inference-only)

**Status: APPROVED 2026-11-19. Done 2026-11-19: official EX 60.9 -> 63.4 (p=0.021), decision rule met, demo uses 3 example rows.**

This is README "Future work" item 4, first and cheapest step: no training.

## Why

With repair and grounding, v2-900 still fails 409 of 1,034 Spider dev questions:
- 320 wrong results;
- 74 `no such column`;
- 11 other execution errors;
- 4 format failures.

Most of these come from not knowing what a column holds:
- picking the wrong table or column (`poker_player.Name`, when names are in `people`);
- the wrong value format (`'cat'` vs `'Cat'`, date formats, codes vs names).

The model currently sees only the `CREATE TABLE` statements. SPEC §5.1 already allows "optionally 3
example rows per table", and `serialize_schema(..., sample_rows=3)` exists and is tested. It has
simply never been evaluated.

**The catch:** the model was trained only on schema-only prompts, so example rows are
off-distribution. They may help or hurt; that is the question.

## Method (fixed in advance, no tuning)

- **Prompt:** the same as today, plus `-- Example rows from <table>:` with the first **3** rows of each
  table (`serialize_schema(sample_rows=3)`), for every database.
  - k=3 comes from SPEC. Other k values are **not** tried, to avoid tuning on dev.
- **Configuration:** the released one, v2-900 adapter, nf4, greedy, with repair and grounding. Only the
  prompt differs.
- **Prompt length:** prompts are never truncated. If any example-row prompt exceeds the current
  3,072-token guard, the guard is raised for both arms; the model's context is 40k. The longest
  prompt is reported.

## Measures and tests

| Measure | Data | Test |
|---|---|---|
| **Primary: Spider dev EX**, with rows vs without | full dev 1,034, ours + official | paired McNemar (one comparison) |
| Error mix: `no such column`, other exec errors, wrong result, format | full dev | counts, with rows vs without |
| Voice EX per condition (secondary) | 100 SpokenSpider × 5 | paired McNemar per condition, not pooled |
| Cost | demo path, CPU | median prompt tokens and text-to-SQL latency on the 30 benchmark questions (`scripts/bench_demo.py`) |

**Decision rule (fixed now):**
- Example rows become the default in the demo and the README "recommended usage" **only if** the
  dev gain is significant (p<0.05, official metric) **and** no voice condition gets significantly worse.
- Otherwise the result is reported and nothing changes.
- A significant dev *loss* is reported as such: the off-distribution prompt hurt.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| S1 | Plumbing, test-first | `scripts/eval_baseline.py --sample-rows`, `scripts/eval_voice.py --sample-rows`, `evaluate_repair_turn` / demo pass-through; a test that `evaluate(..., sample_rows=3)` puts example rows in the prompt and `sample_rows=0` reproduces today's prompt | Tests pass. With `--sample-rows 0` the outputs are identical to the existing week-6 records (checked on dev-200). |
| S2 | Measure | `results/baseline_*_repair_ground_rows3.*`, `results/voice_*_rows3.*`, official EX, `results/demo_latency_*_rows3.json` | The measures table above. |
| S3 | Docs + review gate | `results/week7_results.md`, README (+ `check_readme.py`) | Reviewer approves. The decision rule is applied as written. |

**Cost:** ~30 min code. GPU: ~20 min full dev + ~10 min voice. CPU: ~10 min official EX + ~5 min latency.

## Risks

- **Off-distribution prompts** may make things worse. That is a valid result, and it would point to
  training with example rows (a separate plan) rather than inference-only use.
- **Longer prompts:** more latency on the CPU demo, and the risk of exceeding the token guard. Both are
  measured and reported.
- **Sensitive or long cell values** (e.g. long text, BLOBs): `serialize_schema` already escapes newlines.
  Very long values are not capped. This is checked during the longest-prompt report, and a cap is added
  only if something breaks (it is then reported).
