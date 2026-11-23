# Week 12 results: one-shot evaluation on BIRD dev (PLAN-week12)

**Setup.**
- **Data:** BIRD dev, 1,534 questions on 11 databases. It was downloaded from the official link and
  its sha256 recorded (`results/bird_split_check.json`).
- **Prompt:** the Spider prompt, with the question followed by BIRD's expert evidence on its own line
  (`Evidence: …`). 148 questions have no evidence and get the bare question.
- **Configurations:** each ran once, with nothing tuned on BIRD:
  - A: Qwen3-1.7B zero-shot;
  - B: v2-900;
  - E: v2-900 + repair + grounding + 3 example rows + schema pruning, i.e. the demo configuration.
- **Decoding:** nf4, greedy, 256 new tokens, batch 4 for all three, prompt guard 5,500 tokens. The
  longest prompt used was 4,730 (`results/bird_*.json`).
- **Metric: BIRD EX** as BIRD's `evaluation.py` defines it: the set of predicted rows equals the set
  of gold rows. Row order and duplicates are ignored, but column order counts. It is implemented in
  `tinysql.bird.bird_ex`, not run through the official script; the differences are listed in its
  docstring.
- **Labels:** pre-registered comparisons, Bonferroni over three (p < 0.0167). "Confirmed" means
  significant with the same sign as on Spider.

## Results

| | Configuration | BIRD EX | simple (925) | moderate (464) | challenging (145) |
|---|---|---|---|---|---|
| A | zero-shot | 20.5% | 27.2% | 10.1% | 11.0% |
| B | v2-900 (GRPO) | 23.6% | 31.7% | 11.0% | 12.4% |
| E | + repair, grounding, rows, pruning (**demo**) | **30.2%** | 39.6% | 16.2% | 15.2% |

| # | Comparison | All 1,534: w / l, p | Without formula_1 (1,360): w / l, p | Label |
|---|---|---|---|---|
| T1 | GRPO (B vs A) | +3.1, 68 / 21, p<0.0001 | +3.2, 63 / 19, p<0.0001 | **confirmed** |
| T2 | inference stack (E vs B) | +6.6, 168 / 67, p<0.0001 | +5.9, 143 / 64, p<0.0001 | **confirmed** |
| T3 | whole configuration (E vs A) | +9.7, 204 / 56, p<0.0001 | +9.1, 177 / 54, p<0.0001 | **confirmed** |

Source: `results/bird_{A-base,B-v2-900,E-v2-900-full}.{json,records.jsonl}`.

## Reading

- **The gains transfer to BIRD.**
  - GRPO, trained only on Spider, adds 3.1 points.
  - The inference stack adds another 6.6 points.
  - The demo configuration scores 30.2% against 20.5% zero-shot, a 47% relative gain.
- **The formula_1 overlap does not drive this.**
  - formula_1 has the same schema as a Spider training database.
  - Without its 174 questions the effects are almost unchanged (+3.2 / +5.9 / +9.1).
  - formula_1 is below average for every configuration (A 9.8%, B 11.5%, E 24.1%).
- **What GRPO fixes on BIRD is mostly format.** Answers without a usable ```sql block drop from 98
  to 17. `no such column` errors do not change (412 → 427).
- **What the stack fixes is mostly schema errors.**
  - `no such column` drops from 427 to 280. Example rows and pruning matter more on BIRD's wide,
    cryptically named schemas than on Spider.
  - Repair rescued 50 of 521 retries; grounding fired on 32 queries.
  - In return, wrong-but-running answers rise (586 → 694): queries that used to fail now run but
    still answer wrongly.
- **The absolute level is low**, as expected for a 1.7B model on BIRD. About 70% of answers are still
  wrong, mostly with SQL that runs.
  - The weakest databases are california_schools (10%) and thrombosis_prediction (12%). They have
    long, cryptic column names and questions that need domain formulas.
- **Not a leaderboard number.** The setting is one-shot greedy decoding with our prompt and our
  implementation of BIRD EX, on the original dev split. No comparison with published systems is
  claimed.

## Notes

- **Gold queries:** 2 (question_id 518 and 701) do not finish within 30 s. They count as wrong for
  every configuration, so their effect on the comparisons is the same everywhere.
- **Run times:**
  - A 54 min, B 79 min, E 110 min (`results/w12_bird_runs.log`).
  - Batch 4 was needed because the 4.8k-token prompts do not fit at batch 8 in 8 GiB.
- **Pruner on BIRD:**
  - Recall 95.5%, below Spider's 98.5–98.9%, so pruning drops a needed table more often here.
  - E still wins clearly, but the pruning step was not isolated in this plan.
- **Single run, single seed.**
