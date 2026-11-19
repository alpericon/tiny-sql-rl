---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: transformers
tags:
- text-to-sql
- spider
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

**Spider dev, 1,034 questions** (greedy decoding, thinking off, 256 new tokens). "Official" is
Spider's official evaluation script (exec EX on the standard databases). "Ours" is the project's
column-permutation-tolerant multiset EX.

| Model | Ours | Official |
|---|---|---|
| Qwen3-1.7B zero-shot (nf4) | 53.7% | 54.1% |
| + GRPO, unfiltered data (nf4) | 57.1% | 58.1% |
| **+ GRPO, pass-rate-filtered data (this model, nf4 + adapter)** | **58.5%** | **59.0%** |
| this model + one retry on SQL errors (see Usage) | 60.3% | 60.8% |

- **vs zero-shot:** 84 wins / 33 losses on the same questions (official metric, exact McNemar p<0.0001).
- **vs the unfiltered run:** 39 / 30, p=0.34. That gap is not significant.
- **This release:**
  - merged bf16: 59.7% (ours);
  - merged and re-quantised to nf4: 57.5%;
  - Q8_0 GGUF: 63.5% on a 200-question subset vs 64.5% for bf16 (they disagree on 6 of 200; no
    detectable difference).
- **Q4_K_M is not provided.** It breaks the model's output format: −12.5 points on the same subset.

**Spoken questions** (100 SpokenSpider queries with ASR transcripts, compared with SQLCoder-7B as
used in a voice-to-SQL pipeline):

| Input | this model | SQLCoder-7B |
|---|---|---|
| clean | 52% | 52% |
| 20 dB SNR | 55% | 50% |
| 10 dB SNR | 52% | 48% |
| 0 dB SNR | 38% | 44% |

None of these differences is significant (n=100, 3 databases).

**Latency** (RTX 3070 Ti Laptop, nf4, batch 1): 1.42 s median per query, vs 2.08 s for SQLCoder-7B
measured in the same session. Peak VRAM is 1.4 GiB vs 4.0 GiB.

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

The prompt format matters. Use the chat template with `enable_thinking=False`, and give the schema
as `CREATE TABLE` statements followed by the question. The exact prompt is `tinysql.prompt.build_messages`
in the repository. The model answers with a ```sql block. Decode greedily, with no repetition penalty:
llama-cpp-python's default `repeat_penalty=1.1` noticeably hurts it.

**Repair:** if the SQL fails to run, or there is no block, one follow-up turn helps. Add the model's
answer as the assistant turn, then the user turn *"That SQL failed: `<error>`. Write a corrected
SQLite query that answers the question, in a ```sql block."*. This rescued 19 of 133 failures on
Spider dev. In 53 of the 133 retries the model repeated the same SQL.

**Spoken input:** ASR transcripts lose capitalisation, so the model may write `'aliens'` where the
database holds `'Aliens'`. Before executing, replace a string literal with the database's value when
exactly one value matches it, ignoring case and whitespace. On 100 spoken questions this fixed all 4
clean-ASR cases of this kind (55% → 59%) and broke none. On typed Spider dev it was neutral.

## Limitations

- **Single seed per run.** The McNemar tests cover question sampling, not training variance.
- **English only, SQLite only.** It is evaluated only on Spider-style databases. About 4 in 10 Spider
  dev questions are still answered wrongly, mostly with valid SQL that returns the wrong result.
- **Spoken input loses literal formatting** (case, quotes, "1,000"), which costs ~5 points even on
  clean ASR.
- **The official metric's default removes DISTINCT**, which makes it slightly more lenient than ours.
  Both are reported.

## License and attribution

- Base model: Qwen3-1.7B (Apache-2.0).
- Training/evaluation data: Spider (Yu et al., 2018; CC BY-SA 4.0).
- SpokenSpider transcripts and SQLCoder outputs come from the author's earlier voice-to-SQL project.
- This adapter and the merged weights are released under Apache-2.0.
