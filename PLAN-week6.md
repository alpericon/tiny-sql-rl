# Week 6 plan: grounding SQL string literals in the database (ASR value normalisation)

**Status: APPROVED 2026-11-19.**

This is README "Future work" item 2. It needs no training: the released model (v2-900) stays as is.

## Why

Clean ASR costs ~3 points against typed text even with repair (58% vs 55% on the 100 SpokenSpider
queries). Of the 6 questions typed text gets right and clean ASR gets wrong, **4 are the same
failure**:
- transcripts lose capitalisation and quotes, so the model writes `WHERE Theme != 'aliens'` while
  the database holds `'Aliens'`;
- SQLite compares strings case-sensitively, so the filter silently matches nothing or everything.

The other 2 are real ASR errors ("ID" heard as "it"), which this plan does not address.

## Method: literal grounding (deterministic, fixed in advance)

After the model writes SQL and **before** it is executed:
1. **Value index:** for each single-quoted string literal in the SQL, look it up in an index of the
   database's text values. The index is built once per database from every TEXT column, with
   distinct values only; tables over 50,000 rows are skipped (a cost cap).
2. **Replace only when unambiguous.** A literal is replaced only when it does **not** occur exactly in
   the database, and its normalised form matches **exactly one** distinct database value. Normalised
   form: case-folded, with whitespace collapsed. The replacement is that exact value.
3. **Never touched:**
   - literals that already match exactly;
   - literals containing `%` or `_` (LIKE patterns);
   - literals with zero or several normalised matches;
   - anything that is not a single-quoted string (numbers, identifiers, double-quoted names).
4. **Order with the Y9 repair step:** ground → execute → if it fails, repair once → ground the
   repaired SQL → execute.

Scope is case and whitespace only. Number formatting ("1,000") and spelling errors are out of scope;
they belong to a later item if the data shows they matter.

## Measures and tests (fixed before running)

| Measure | Data | Test |
|---|---|---|
| **Primary: voice EX, clean ASR**, repair + grounding vs repair only | 100 SpokenSpider clean transcripts | paired McNemar |
| Voice EX at 20 / 10 / 0 dB and on typed text | same 100 queries | paired McNemar per condition (secondary; not pooled) |
| **Guard rail: Spider dev EX** (typed questions, where grounding should rarely fire) | full dev, ours + official | paired McNemar vs repair only; a significant loss fails the method |
| How often grounding fires, and what it changes | all of the above | counts, and a hand read of every changed literal on voice and a sample on dev |

- **Expected size:** small. At most ~4 of 100 on clean ASR, so the primary test **cannot reach
  significance** at n=100 even if every case is fixed (4 / 0 gives p=0.125).
- **It is reported as counts with the per-case hand read.** The plan says so now, so the result is
  not over- or under-sold later.
- **No checkpoint selection, no tuning:** one deterministic rule, one model.
- **SQLCoder:** its stored SQL is re-scored with grounding too, as a second row. The comparison is
  then about the model, not the post-processing.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| G1 | Grounding core, test-first | `src/tinysql/ground.py` (+`tests/test_ground.py`) | Tests cover: case and whitespace fix with a unique match; no change on exact match, ambiguous match, no match, LIKE patterns, numbers, double-quoted identifiers, escaped quotes (`'O''Brien'`); the 50,000-row cap; idempotence. |
| G2 | Wire into evaluation and demo | `evaluate(..., ground=False)`, `eval_baseline.py / eval_voice.py --ground`, `tinysql.demo.answer_question(ground=True)` | Records keep `ungrounded_sql` and the list of replacements. `ground=False` reproduces current results (existing tests unchanged). |
| G3 | Measure | `results/voice_*_repair_ground.*`, `results/baseline_*_repair_ground.*`, official EX | The measures table above, plus SQLCoder re-scored with grounding. Runs on v2-900 + repair, the released configuration. |
| G4 | Docs + review gate | `results/week6_results.md`, README, model card (local; HF upload only after your go-ahead) | Reviewer approves; README numbers verified by `scripts/check_readme.py`. The demo uses grounding only if the guard rail holds. |

**Cost:** ~1 h of code. GPU: ~15 min full dev + ~8 min voice. CPU: ~10 min official EX. No overnight run.

## Risks

- **A wrong replacement could change a correct answer.**
  - Example: the user really meant lowercase `'aliens'` and both `'aliens'` and `'Aliens'` exist.
    That case is ambiguous, so it is not replaced.
  - When only one case variant exists, querying for another variant can never match in SQLite
    anyway, so the replacement cannot make a correct answer wrong.
  - The guard rail on full dev checks this empirically.
- **Large databases** (e.g. wta_1): tables over 50,000 rows are skipped; the count is logged.
- **Literals in other positions** (e.g. `SELECT 'x'`, CASE expressions) are grounded too. This
  rarely matters, and the hand read will show it.
