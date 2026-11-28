---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: transformers
tags:
- text-to-sql
- spider
- bird
- grpo
- reinforcement-learning
- gguf
datasets:
- xlangai/spider
language:
- en
pipeline_tag: text-generation
---

# tiny-sql-rl: Qwen3-1.7B + GRPO for text-to-SQL

Qwen3-1.7B, fine-tuned with GRPO using an **execution reward** on Spider: a generated query is
rewarded when its result matches the gold query's result. It was trained as a 4-bit QLoRA on one
8 GB laptop GPU (RTX 3070 Ti Laptop).

This repository contains:
- `adapter/`: the LoRA adapter (checkpoint v2-900);
- the merged bf16 model (root);
- `gguf/tiny-sql-rl-Q8_0.gguf`: a llama.cpp export.

Code, training logs and every number below: https://github.com/alpericon/tiny-sql-rl

## Results

All numbers are execution accuracy (EX) with greedy decoding, thinking off, 256 new tokens and nf4.
"Official" is Spider's official evaluation script. "Ours" is the project's
column-permutation-tolerant multiset EX. All comparisons are paired (exact McNemar on the same questions).

### Held-out Spider test (2,147 questions, 40 unseen databases)

Every inference step below was chosen on Spider dev. The final configuration was then run **once** on
the Spider test set (released 2024). None of its databases appear in train or dev, and nothing was
tuned on it.

| Configuration | Official EX, test | Official EX, dev |
|---|---|---|
| Qwen3-1.7B zero-shot | 56.3% | 54.1% |
| **this model** (GRPO) | 61.7% | 59.0% |
| + one retry on SQL errors + literal grounding | 64.2% | 60.9% |
| + 3 example rows per table in the prompt | 65.7% | 63.4% |
| **+ schema pruning (recommended)** | **66.3%** | 65.9% |

- **Confirmed on test:**
  - GRPO over zero-shot: +5.4 points, 160 wins / 44 losses, p<0.0001.
  - The whole recommended configuration over zero-shot: +10.1 points, 308 / 92, p<0.0001.
- **Same direction on test but not significant:**
  - example rows: +1.5, p=0.073;
  - schema pruning: +0.6, p=0.28.
  - Both were adopted because they were significant on dev, and both gain less on test.

### Beyond Spider: BIRD dev (1,534 questions, 11 databases)

BIRD has large, messy databases with cryptic column names, and gives each question an expert
"evidence" hint, which is added to the prompt on its own line. Each configuration ran **once**, with
nothing tuned on BIRD. The metric is BIRD EX (the set of result rows equals the gold set), using the
project's own implementation of it, not BIRD's script.

| Configuration | BIRD EX | simple | moderate | challenging |
|---|---|---|---|---|
| Qwen3-1.7B zero-shot | 20.5% | 27.2% | 10.1% | 11.0% |
| **this model** (GRPO) | 23.6% | 31.7% | 11.0% | 12.4% |
| **recommended configuration** | **30.2%** | 39.6% | 16.2% | 15.2% |

- **All three comparisons are confirmed** (p<0.0001 each, Bonferroni over three):
  - GRPO over zero-shot: +3.1 points, 68 / 21;
  - the inference steps over this model: +6.6, 168 / 67;
  - the whole configuration over zero-shot: +9.7, 204 / 56.
- **The Spider overlap does not drive this.** BIRD's `formula_1` database has the same schema as a
  Spider training database. Without its 174 questions the gains are +3.2 / +5.9 / +9.1, all still
  significant.
- **What changes:**
  - GRPO removes most format failures (98 → 17); that is about a third of its wins.
  - The inference steps mainly remove `no such column` errors (427 → 280). They were not measured
    one by one on BIRD.
- **The absolute level is low**, as expected for a 1.7B model: about 70% of answers are still wrong.
  This is not a leaderboard number.

### Against a same-size SQL model: Qwen2.5-Coder-1.5B-Instruct

Coder was run once on both benchmarks in the same setting (same prompt, 256 tokens, nf4), bare and
with the same inference steps.

| | Spider test (official) | BIRD dev |
|---|---|---|
| Qwen2.5-Coder-1.5B zero-shot | 55.1% | 21.9% |
| **this model** | 61.7% | 23.6% |
| Coder + the recommended inference steps | 57.5% | 25.7% |
| **this model, recommended configuration** | **66.3%** | **30.2%** |

- **This model is ahead** on Spider test, bare (+6.5) and with the inference steps (+8.8), and on BIRD
  with the inference steps (+4.4). All have p≤0.0006, Bonferroni over four.
- **Bare on BIRD:** +1.7, not significant (p=0.19).
- **The two base models are about equal**, so the lead comes from GRPO and the inference steps.
- **Much of the lead is format.** Coder often explains step by step and runs out of 256 tokens without
  a SQL block (10–19% of answers). This model was trained on this exact prompt; Coder was not.
- Coder may have seen Spider or BIRD dev in pre-training, which would favour it. This cannot be checked.

### Three training seeds

The training recipe was repeated with seeds 1 and 2, identical otherwise, and each run was selected by
the same dev-200 rule and scored once, bare.

| Seed | Spider test (official) | BIRD dev |
|---|---|---|
| Qwen3-1.7B zero-shot | 56.3% | 20.5% |
| **3407 (this model)** | 61.7% | 23.6% |
| 1 | 61.9% | 23.3% |
| 2 | 61.7% | 23.9% |

- **The GRPO gain replicated with both seeds on both benchmarks** (p<0.0001 each, Bonferroni over four).
  Mean gain over zero-shot: +5.5 points on Spider test (range 5.4–5.6) and +3.1 on BIRD dev (2.8–3.3).
- This model's seed is the lowest of the three on Spider test (tied) and the middle one on BIRD. The
  release was not re-picked by test score.
- The recommended configuration's numbers were measured with this seed only.

### Training ablations (Spider dev, 1,034 questions)

| Model | Ours | Official |
|---|---|---|
| Qwen3-1.7B zero-shot | 53.7% | 54.1% |
| + GRPO, unfiltered data | 57.1% | 58.1% |
| **+ GRPO, pass-rate-filtered data (this model)** | **58.5%** | **59.0%** |

- **Filtered vs unfiltered data:** 39 / 30, p=0.34, so the difference is not significant.
- **Two further training runs gave null results**, so this model stays the release:
  - training on repair turns (dev 62.8 vs 60.8 official with retry, confounded with extra steps);
  - training with example rows in the prompt (63.9% vs 63.4%, p=0.58).
- **This release, merged:** bf16 59.7% (ours); re-quantised to nf4 57.5%.
- **Q8_0 GGUF:** 63.5% on a 200-question subset vs 64.5% for bf16 (no detectable difference).
- **Q4_K_M is not provided.** It breaks the output format (−12.5 points on the same subset).

### Spoken questions

100 SpokenSpider queries with ASR transcripts, compared with SQLCoder-7B as used in a voice-to-SQL
pipeline. Both systems use the same literal grounding.

| Input | this model, recommended configuration | SQLCoder-7B (+ grounding) | wins / losses, p |
|---|---|---|---|
| clean | 68% | 55% | 21 / 8, 0.024 |
| 20 dB SNR | 67% | 53% | 22 / 8, 0.016 |
| 10 dB SNR | 65% | 51% | 22 / 8, 0.016 |
| 0 dB SNR | 46% | 45% | 11 / 10, 1.0 |

The data is only 100 questions on 3 databases, and the four conditions are separate tests. With a
Bonferroni correction (α = 0.0125), none is significant on its own. The 1.7B model is ahead from clean
speech to 10 dB, and level at 0 dB.

### Latency

- **GPU** (RTX 3070 Ti Laptop, nf4, batch 1, merged model, schema-only prompt): 1.42 s median per
  query, vs 2.08 s for SQLCoder-7B in the same session. Peak VRAM is 1.4 GiB vs 4.0 GiB.
- **CPU** (Q8_0 GGUF, 2 threads, recommended configuration): 6.8 s median for a first question on a
  database, and 2.6 s for a follow-up on the same database.

## Training

- **Data:** Spider train, minus the 100 SpokenSpider queries (held out for the voice evaluation).
  Questions whose gold result is empty are dropped; they reward any empty query.
  - A pass-rate filter then keeps only questions the base model solves 1–7 times out of 8 samples:
    1,688 of 6,638.
- **GRPO:** TRL + Unsloth, 4-bit QLoRA (r=16, alpha=16, all linear projections).
  - 8 generations per prompt, 4 prompts per step, 900 steps (best dev-200 checkpoint of 1,000; rule fixed in advance).
  - lr 5e-6 (cosine), β (KL) 0.04, temperature 1.0, seed 3407.
- **Reward:** 1.0 if the result matches gold, 0.1 if the SQL executes but the result is wrong, else 0.
  A binary-reward ablation gave the same accuracy (19 / 17 on full dev).

## Usage

**Prompt:**
- The model was trained with a schema-only prompt. Use the chat template with
  `enable_thinking=False`, and give the schema as `CREATE TABLE` statements followed by the question.
  The exact prompt is `tinysql.prompt.build_messages` in the repository.
- The model answers with a ```sql block.
- Decode greedily, with no repetition penalty. llama-cpp-python's default `repeat_penalty=1.1`
  noticeably hurts it.

**Recommended configuration.** Each step is inference-only and implemented in the repository.
1. **Example rows:** after each `CREATE TABLE`, add the table's first 3 rows as
   `-- Example rows from <table>:` comment lines (`serialize_schema(db, sample_rows=3)`). It mainly
   reduces wrong-column errors. On CPU it roughly doubles the cold prompt cost.
2. **Schema pruning:** show only the tables the question mentions, by name, column word or stored
   value, plus the foreign-key bridge tables between them (`tinysql.prune.select_tables`). When fewer
   than 2 tables match, show the full schema. The pruner was tuned on Spider train and keeps every
   needed table for 98.9% of dev and 98.5% of test questions.
3. **Repair:** if the SQL fails to run, or there is no block, give one follow-up turn on the **full**
   schema. Add the model's answer as the assistant turn, then the user turn *"That SQL failed:
   `<error>`. Write a corrected SQLite query that answers the question, in a ```sql block."*.
4. **Literal grounding:** before executing, replace a string literal with the database's own value
   when exactly one value matches it, ignoring case and whitespace. This mainly matters for spoken
   input, where ASR loses capitalisation.

## Limitations

- **Seeds.** The training recipe was run with three seeds (see above). The ablations and the
  recommended configuration are single runs; their McNemar tests cover question sampling, not
  training variance.
- **English only, SQLite only; trained on Spider-style schemas.**
  - About a third of Spider test questions are still answered wrongly, mostly with valid SQL that
    returns the wrong result.
  - An error analysis of 100 such failures on dev found that 45% pick the wrong table, column or
    join path.
  - The pruner relies on question words matching schema names. On BIRD's cryptic schemas it keeps
    every needed table for 95.5% of questions, against 98.5–98.9% on Spider.
  - On BIRD, about 70% of answers are still wrong.
- **Spider's gold queries have errors.** On dev, about 15% of the remaining wrong-result failures
  came from a wrong gold query or an ambiguous question.
- **Spoken input:** ASR word errors ("ID" heard as "it") and number formats ("1,000") are not handled.
- **The official metric's default removes DISTINCT**, which makes it slightly more lenient than ours.
  Both are reported.

## License and attribution

- Base model: Qwen3-1.7B (Apache-2.0).
- Training/evaluation data: Spider (Yu et al., 2018; CC BY-SA 4.0).
- Evaluation only: BIRD dev (Li et al., 2023; CC BY-SA 4.0).
- SpokenSpider transcripts and SQLCoder outputs come from the author's earlier voice-to-SQL project.
- This adapter and the merged weights are released under Apache-2.0.
