# Week 9: what are the remaining "wrong result" errors? (PLAN-week9)

**Setup.** This covers the released configuration: v2-900, 3 example rows, repair and grounding. On
Spider dev, 301 questions get an answer that runs but is wrong on both our metric and the official one.
From these, 100 were drawn with seed 3407, stratified by hardness: 39 medium, 29 extra, 23 hard, 9
easy (`scripts/error_sample.py` → `results/error_analysis_sample.jsonl`).

Each question got one primary label: the first thing that would have to change for the predicted SQL
to be right. A secondary label was optional. The taxonomy was fixed in the plan. All 100 labels and
notes are in `results/error_analysis_labels.jsonl`.

## Agreement check (E3)

The user labelled the 20 marked rows blind (`results/error_analysis_spotcheck_labels.jsonl`). **The
primary labels agree on 17 of 20**, above the plan's threshold of 15.

| # | mine | user's |
|---|---|---|
| 27 | condition (counts arrivals only) | join |
| 28 | join (airline name vs uid) | schema-link |
| 33 | aggregation (groups by the wrong id) | schema-link |

All three disagreements lie on the boundaries between schema-link, join, aggregation and condition.
With schema-link and join merged, agreement is 18 of 20 (it rescues #28). Fine distinctions between
these labels are therefore less reliable than the merged shares.

## Shares (primary label, n = 100, Wilson 95% intervals)

| Label | Count | 95% CI |
|---|---:|---|
| `schema-link`: wrong table or column | **29** | 21–39 |
| `join`: wrong or missing join path | **16** | 10–24 |
| `gold-issue`: wrong gold or ambiguous question | **15** | 9–23 |
| `condition`: missing, extra or wrong condition | 12 | 7–20 |
| `aggregation` | 9 | 5–16 |
| `nesting-setop` | 8 | 4–15 |
| `output-shape`: right rows, wrong columns | 8 | 4–15 |
| `literal` | 2 | 1–7 |
| `order-limit` | 1 | 0–5 |
| **schema-link + join** | **45** | 36–55 |

**By hardness** (primary label):
- medium (39): schema-link 14, join 7, gold-issue 6, condition 5, aggregation 5, output-shape 2;
- hard (23): schema-link 9, join 6, nesting-setop 3, output-shape 2, order-limit 1, gold-issue 1, condition 1;
- extra (29): gold-issue 6, condition 5, nesting-setop 5, schema-link 4, join 3, output-shape 3, literal 2, aggregation 1;
- easy (9): aggregation 3, gold-issue 2, schema-link 2, condition 1, output-shape 1.

## What the errors look like

- **Schema linking and joins: 45 of 100, the largest group.** The model knows what the question asks
  but picks the wrong place in the schema. Examples:
  - it counts `car_names` (versions) when the question is about models (`model_list`);
  - it compares airline names with airline ids (`airlines.Airline` vs `flights.Airline`);
  - it groups visitors instead of museums.
  Most of these are medium and hard questions on databases with several similar tables (car_1,
  cre_Doc_Template_Mgt, network_1).
- **Foreign keys are mostly already in the prompt.** All 17 sampled databases declare foreign keys in
  their `CREATE TABLE` statements. Of the 45 schema-link and join errors, 29 have a gold query with
  joins. In 27 of those, every table pair the gold joins is linked by a declared key. The 2 exceptions
  are #28 and #29: flight_2's `flights.Airline` → `airlines.uid` is not declared.
  "Foreign-key hints", listed in the plan as a candidate, would therefore add little on this sample.
- **Invented conditions** (12, plus 5 as a secondary label). The model adds filters the question
  never mentions: `Sex = 'F'`, `grade = 9`, `Continent = 'america'`, `abandoned_yn = '1'`.
- **Literal errors are nearly gone** (2). Grounding (week 6) and example rows (week 7) cover this class.
- **Output shape** (8). An extra count or id column next to the right answer. On these rows both
  metrics score it wrong.

## Gold issues

15 primary, plus 6 more as a secondary label (21 rows mention a gold problem).
- **In 12 of the 15, the prediction is correct or a valid reading of the question.** Examples:
  - city names stored with a trailing space (`'Anthony '`), so the gold returns no rows;
  - a gold join with no `ON` condition;
  - a gold literal in the wrong case, which matches nothing;
  - `LEFT JOIN` vs inner join for "each document";
  - `SUM(cost)` vs `COUNT(*)` for "spent the most".
- In the other 3 (#21, #84, #85) the question is ambiguous or garbled, and the prediction is not clearly
  right either.
- **Only 7 of the 12 are clearly correct** (#23, 25, 43, 55, 56, 96, 97). The other 5 are defensible
  but arguable readings (#40 and #87 use `LEFT JOIN` for "each", #61 column order, #78 countries with no
  language rows, #81 ties).
- **Rough extrapolation:** 15% of the 301 is about 45 questions (95% CI ~27–69). Counting 7 to 12 of
  every 15 as right answers gives **roughly 20–40 questions, or about 2–4 points of dev EX**, scored
  wrong though the answer is right or defensible. This is an estimate from 15 rows. It does not change any reported number, but it
  means the measurable ceiling on Spider dev is lower than 100%.

## What this points to (the plan's table, applied)

- **The largest fixable share is schema-link + join (45%).** The plan names two candidates for this:
  - "foreign-key hints": little room, since the needed keys are declared in 27 of 29 cases;
  - "schema pruning": showing only the relevant tables and columns. This remains a candidate.
- **Gold issues are 15%**, under the plan's 25% threshold, so the plan's table does not point to a
  second benchmark on this evidence.
- No single other class dominates. Condition, aggregation, nesting and output-shape together are 37%.

The next experiment is chosen in a separate plan. Two candidates follow from these numbers:
1. **Schema pruning**, aimed at the 45%. A first step is to measure whether the gold tables are kept
   (recall) before any accuracy run.
2. **Self-consistency with execution voting**: sample several answers and pick the most common result.
   It needs no training and touches all classes, but it costs latency.

## Caveats

- **One main labeller (Claude), checked on 20 rows by the user.** Boundaries between schema-link, join,
  aggregation and condition are fuzzy (all 3 disagreements).
- **n = 100.** The intervals are wide; the order of classes below the top three is not reliable.
- **Only failures were labelled.** The shares say what the model gets wrong, not how often it gets the
  same kind of thing right.
