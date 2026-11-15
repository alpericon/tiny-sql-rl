# SPEC: Small Text-to-SQL Model Trained with RL (Execution Reward) + Voice Demo

Status: **APPROVED 2026-11-15** · Date: 2026-11-15 · Budget: 3–4 weeks · Hardware: 1× RTX 3070 Ti (8 GB, Ampere)

## 1. Why this project

In `voice-to-sql`, SQLCoder-7B (4-bit) accounted for **~96% of end-to-end latency (~16 s/query)**.
This project tests whether a **≤1.7B model trained with GRPO on an execution-based reward** can replace it:
much faster, with execution accuracy close enough to be usable.

The story for the portfolio: *"I measured the bottleneck in my previous project, then trained a model to remove it."*

## 2. Research questions and success criteria

| # | Question | Measured by |
|---|---|---|
| RQ1 | How much does GRPO improve a small model over its own zero-shot baseline? | Spider dev execution accuracy (EX), base vs. GRPO |
| RQ2 | How close does it get to SQLCoder-7B? | EX on the same eval set as voice-to-sql |
| RQ3 | What is the latency gain in the full voice pipeline? | Text-to-SQL latency and total latency on the 400 SpokenSpider rows |
| RQ4 | Is it still robust to ASR noise? | EX per SNR condition (clean / 20 / 10 / 0 dB) |

**Success** = an honest results table answering RQ1–RQ4, plus a working demo.
There is no fixed accuracy target. A negative result (e.g. "GRPO helps less than SFT") is still a valid outcome if it is measured properly.

## 3. Constraints (shaping every decision)

- **8 GB, Ampere → no FP8.** Use **4-bit QLoRA**. Model size: **0.5B–1.7B**.
- **vLLM / Unsloth fast generation need Linux** → train inside **WSL2 (Ubuntu)**. Windows is only for the demo.
- **Fallback if it does not fit or is too slow:** TRL `GRPOTrainer` without vLLM, fewer generations per prompt (G=4), shorter completions, or the smaller model.
- The library versions current at implementation time are pinned in `requirements.txt` and verified in Week 1, not assumed.

## 4. Data

| Use | Data | Notes |
|---|---|---|
| RL training | Spider **train** (~7k questions) | **Exclude `train_00000`–`train_00099`**: these are the 100 SpokenSpider questions used in voice-to-sql. Including them would leak the voice evaluation set. |
| Main eval | Spider **dev** (1,034 questions) | Never trained on. Greedy decoding. |
| Voice eval | SpokenSpider (100 questions × 4 SNR = 400 rows) | Reuse `data/metadata.csv` and the transcripts from voice-to-sql, so ASR is held fixed and only text-to-SQL changes. |
| Stretch | BIRD dev | Only if Weeks 1–3 finish early. |

## 5. Method

### 5.1 Prompt
Serialized schema (`CREATE TABLE` statements, plus optionally 3 example rows per table) + question → SQL inside a fenced block.
**Check:** no prompt may be truncated. Truncation hides the schema and drives the reward toward 0 (a known pitfall). Log the maximum prompt length.

### 5.2 Reward (execution-based, hack-resistant)
| Outcome | Reward |
|---|---|
| Result set matches the gold query's result (order-insensitive unless `ORDER BY`) | **1.0** |
| Valid SQL that executes, but the result is wrong | 0.1 |
| Syntax or execution error, or timeout | 0.0 |
| Output is not in the required format | 0.0 |

**Sandbox:** read-only SQLite connection (`mode=ro` URI), one statement only (`SELECT`/`WITH`), 5 s timeout, row cap. Rewards on model-written SQL are easy to game without these checks.

### 5.3 Training stages
1. **Baseline:** zero-shot EX for 2–3 candidate small models (e.g. Qwen-family 0.5B / 1.5B–1.7B instruct or coder variants, whichever are current). Pick the one with the best EX-per-latency that fits GRPO in 8 GB.
2. **(Conditional) SFT warm-up:** only if the baseline fails the output format more than ~20% of the time. A short QLoRA SFT on Spider train, excluding the same 100 questions.
3. **GRPO:** QLoRA, G=4–8 generations per prompt, KL to the reference model, about 1–2k steps. Fixed seed. Log reward, EX on a 200-question dev subset, and completion length.
4. **One ablation:** the partial-credit reward (0.1) vs. a pure binary reward.

### 5.4 Evaluation protocol
- Spider **execution accuracy** using the official evaluation script (test-suite EX where available).
- Same prompt and greedy decoding for all models. Single seed, stated as a limitation.
- Latency measured on the 3070 Ti, same quantization for every model, median and p95.

## 6. Plan (4 weeks)

| Week | Deliverable | Verification |
|---|---|---|
| 1 | WSL2 environment, data loader with leakage filter, sandboxed executor, eval harness, zero-shot baselines | Unit tests: executor rejects writes and multiple statements, and times out; filter removes exactly 100 IDs; baseline EX table |
| 2 | GRPO training runs (+ SFT warm-up if needed) | Reward curve rises; dev-subset EX beats baseline; VRAM headroom logged |
| 3 | Full Spider dev eval, ablation, voice-pipeline swap (RQ3–RQ4) | Results table filled in; same 400 rows as voice-to-sql |
| 4 | HF model card, demo, README with results and figures, short write-up | Demo works end to end; README numbers trace back to result files |

## 7. Demo

- **Hugging Face Space** (free CPU tier): a GGUF-quantized model via llama.cpp + Whisper-small. Text or audio input → SQL → results table → spoken answer, against a few Spider databases.
- If CPU latency is poor, the Space becomes text-only and the repo includes a recorded video of the local GPU voice demo.

## 8. Deliverables

- GitHub repo `alpericon/<name>`: code, configs, tests, results, README.
- Hugging Face model (LoRA adapter + merged/GGUF) with a model card.
- Hugging Face Space demo.
- Profile README entry.

## 9. Out of scope

Multi-turn conversations, PPO, training-data generation with frontier models, BIRD (unless stretch), multiple seeds.

## 10. Risks

| Risk | Mitigation |
|---|---|
| GRPO does not fit in 8 GB | Smaller model, G=4, shorter `max_completion_length`, gradient checkpointing; measure VRAM in a 50-step pre-flight before every new configuration |
| Reward stays ~0 (model never produces correct SQL) | SFT warm-up; partial-credit reward; check for prompt truncation |
| Reward hacking | Sandbox in 5.2; manually inspect 20 high-reward samples per run |
| WSL2 / CUDA setup consumes Week 1 | Time-box setup to 2 days, then fall back to TRL without vLLM on native Windows |
| `uv sync`-style tools revert torch to the CPU wheel | Verify `torch.cuda.is_available()` at the start of every training script |

## 11. Decisions (resolved at approval)

1. Repo name: **`tiny-sql-rl`** (the first proposed option; can be renamed before the first push).
2. Default reward uses **0.1 partial credit**; binary reward is the ablation.
3. **No SFT at the start**; add the warm-up only if the baseline format-failure rate exceeds ~20%.
