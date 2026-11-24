# Week 13 results: our model vs Qwen2.5-Coder-1.5B-Instruct (PLAN-week13)

**Setup.**
- **New runs:** Qwen2.5-Coder-1.5B-Instruct, once each, on two configurations:
  - CA: zero-shot, schema-only;
  - CE: with our full inference stack (repair, grounding, 3 example rows, pruning).
- **Our side is not rerun.** The week 11 (Spider test) and week 12 (BIRD dev) records are reused:
  - A: Qwen3-1.7B zero-shot;
  - B: v2-900, the released GRPO model;
  - E: v2-900 with the same stack (the demo configuration).
- **Prompt and decoding:** the same for both models. nf4, greedy, no repetition penalty, 256 new tokens.
  - Spider test: batch 8, prompt guard 3,072.
  - BIRD dev: batch 4, prompt guard 5,500.
- **Metrics:**
  - Spider test (2,147 questions): official EX.
  - BIRD dev (1,534 questions): BIRD EX.
- **Tests:** four pre-registered comparisons, exact McNemar, two-sided, Bonferroni over four
  (p < 0.0125).

## Results

| | Spider test (official EX) | BIRD dev (BIRD EX) |
|---|---|---|
| Qwen3-1.7B zero-shot (A) | 56.3% | 20.5% |
| Qwen2.5-Coder-1.5B zero-shot (CA) | 55.1% | 21.9% |
| **our GRPO model (B)** | **61.7%** | **23.6%** |
| Coder + our inference stack (CE) | 57.5% | 25.7% |
| **our model + the stack (E, demo)** | **66.3%** | **30.2%** |

| # | Data | Comparison | Difference | w / l | p | Label |
|---|---|---|---|---|---|---|
| K1 | Spider test | B vs CA | +6.5 | 372 / 232 | <0.0001 | **ours ahead** |
| K2 | Spider test | E vs CE | +8.8 | 386 / 196 | <0.0001 | **ours ahead** |
| K3 | BIRD dev | B vs CA | +1.7 | 199 / 173 | 0.19 | no detectable difference |
| K4 | BIRD dev | E vs CE | +4.4 | 223 / 155 | 0.0006 | **ours ahead** |

BIRD without `formula_1` (1,360 questions; secondary, as in week 12):
- K3: +3.9, 191 / 138, p=0.004.
- K4: +4.6, 201 / 138, p=0.0007.

Sources:
- `results/official_ex_test.json` and `results/official_test/`;
- `results/bird_compare_week13.json`;
- `results/test_Qwen__Qwen2.5-Coder-1.5B-Instruct*.json` and `results/bird_C{A,E}-coder*.json`.

## Reading

**In the recommended configuration, our model is ahead on both benchmarks.**
- +8.8 points on Spider test and +4.4 on BIRD dev over Coder with the same stack.
- The bare GRPO model is clearly ahead on Spider test (+6.5). On BIRD it is +1.7, which is not
  significant on all questions.

**The two base models are about equal.**
- Coder vs Qwen3, both zero-shot: −1.1 points on Spider test and +1.4 on BIRD.
- Neither difference is significant (descriptive: p=0.36 and p=0.30).
- So the lead comes from what this project added (GRPO and the stack), not from a better starting
  model.

**Much of the lead is format: Coder explains instead of answering.**

| | Spider test, format failures | BIRD dev, format failures | BIRD, answers cut at 256 tokens |
|---|---|---|---|
| Qwen3 zero-shot (A) | 35 | 98 | 26 (1.7%)¹ |
| our model (B) | 6 | 17 | 14 (0.9%)¹ |
| Coder zero-shot (CA) | 222 (10.3%) | 233 (15.2%) | 249 (16.2%) |
| Coder + stack (CE) | 323 (15.0%) | 297 (19.4%) | 28.2% of all generations² |

¹ Week 12 did not record 256-token hits. These are estimated by re-tokenising the stored answers. On
Coder's CA run the estimate agrees with the recorded flag on 1,529 of 1,534 questions (244 vs 249).
² Includes repair turns, which interleave with first attempts, so no per-question flag is kept.

- Coder often writes a step-by-step explanation before the SQL and runs out of the 256 tokens.
- With the full stack it is worse: format failures rise from CA to CE on both benchmarks. Longer
  prompts and repair turns that also hit 256 tokens are possible causes; no Coder run isolates a step.
- **On questions where both models produced a SQL block:**
  - Spider test: ours 63.5% vs 61.5% (bare) and 68.6% vs 67.7% (stack);
  - BIRD dev: 25.3% vs 26.0% (bare) and 32.6% vs 32.2% (stack).
  - These subsets are selected by each model's own behaviour. On Spider test, Coder's format failures
    are mostly on hard and extra questions (3.6% easy, 18.8% hard); on BIRD they are about flat by
    difficulty. So these are indicative, not tested.
  - Read that way, on Spider part of the lead is better SQL. On BIRD the lead is almost all format.
- **This is a property of the setting, not only of the model.** Our model was trained on exactly this
  prompt and answer format, and Coder was not. A longer token budget or a prompt tuned for Coder might
  narrow the gap. The plan fixed both in advance, so neither was tried.

**The inference stack also helps Coder.** This is descriptive, the stack applied to a model it was not
developed on:
- Spider test: +2.3, 213 / 163, p=0.011;
- BIRD dev: +3.8, 150 / 91, p=0.0002.
- It helps Coder less than it helps our model (+4.7 Spider, +6.6 BIRD). Part of the reason is the
  format failures above.

**formula_1 favours Coder.**
- On this BIRD database, the one that shares its schema with Spider train, Coder zero-shot scores
  27.0%. Our GRPO model scores 11.5% and our demo configuration 24.1%.
- Without formula_1, K3 has p=0.004. This is the secondary analysis, not a pre-registered test.
- Why Coder is strong there is not known.

**Possible contamination in Coder's favour (plan risk).** Coder may have seen Spider (including dev)
or BIRD dev in pre-training. That would favour Coder. We cannot check it.

## Notes

- **Descriptive numbers** (format failures, 256-token hits, EX where both have a block, formula_1):
  `scripts/week13_extras.py` and `scripts/week13_hits.py`.
- **Run times (RTX 3070 Ti Laptop):**
  - Spider test: CA 71 min, CE 86 min.
  - BIRD dev: CA 47 min, CE 131 min.
  - Log: `results/w13_runs.log`.
- **Prompt length:** the longest Coder prompt was 4,774 tokens (BIRD CE), within the 5,500 guard.
- **Repair rescues for Coder:** 27 on Spider test and 67 on BIRD.
- **Single run, single seed** for our model. The Coder runs are deterministic (greedy).
- **Not a statement about Coder in general.** It was run in our setting: our prompt, 256 tokens,
  4-bit. The Qwen2.5-Coder report uses its own prompts and settings.
