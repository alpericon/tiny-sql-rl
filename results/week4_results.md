# Week 4 results (PLAN-week4)

Official Spider EX (Y1) is in `results/week3_results.md`, next to the week-3 numbers it re-scores.

## Y2: merged v2-900

`scripts/merge_adapter.py` merges the v2-900 LoRA adapter into the **bf16** Qwen3-1.7B weights. The
adapter was trained on a 4-bit base, so the merge is not bit-identical to the evaluated adapter model,
and the merged model is evaluated again.

**Full dev, same command as week 3** (greedy, batch 8, 256 tokens, thinking off; our EX).

| Model | Precision | EX | Format fail | Exec error | Wrong result | vs adapter v2-900 (w / l, p) |
|---|---|---|---|---|---|---|
| v2-900, base + adapter (week 3) | nf4 | 58.5% | 10 | 123 | 296 | |
| v2-900 merged | bf16 | 59.7% | 2 | 128 | 287 | 85 / 73, 0.38 |
| v2-900 merged | nf4 (re-quantised) | 57.5% | 10 | 125 | 304 | 28 / 38, 0.27 |

Sources: `results/baseline_v2-900-merged-bf16.json`, `results/baseline_v2-900-merged-nf4.json`.

- **Merging loses nothing measurable.** Neither merged version differs significantly from the adapter
  model.
- **bf16 vs nf4 (both merged): 59.7 vs 57.5, 70 / 92, p=0.099.**
  - The direction suggests 4-bit costs ~2pp, but the test is not significant.
  - The base model was never evaluated in bf16, so this cannot be attributed to the adapter.
  - bf16 has fewer format failures (2 vs 10).
- **The flips are large even when the means agree:** merged bf16 vs adapter is 158 discordant
  questions out of 1,034, for a +1.2pp difference. Greedy decoding is sensitive to small numeric
  changes, so single-run differences under ~2pp should not be read as real.

**Latency (RTX 3070 Ti Laptop, nf4, batch 1, 100 clean transcripts; `scripts/bench_latency.py`).**
The three systems were run back to back under the same conditions, while the official-EX job held
2 CPU cores. All three are therefore ~10–15% slower than the week-3 numbers, so compare within this table only.

| System | median | p95 | ms / token | End-to-end est. median |
|---|---|---|---|---|
| v2-900 merged (nf4) | **1.42 s** | 3.63 s | 45 | 2.09 s |
| Qwen3-1.7B base (nf4, no adapter) | 1.49 s | 3.39 s | 46 | 2.16 s |
| SQLCoder-7B (nf4, voice-to-sql settings) | 2.08 s | 4.31 s | 50 | 2.65 s |

Sources: `results/latency_*_y2.json`.

- **The merged model has no adapter overhead.** Per token it is as fast as the base model (45 vs 46 ms).
- **It is ~32% faster than SQLCoder-7B** at the median (1.42 vs 2.08 s). This confirms the week-3
  hypothesis (~25% faster, from the no-adapter lower bound), now measured on the release model.
  - Most of the gap is fewer output tokens (31 vs 41.5 median), not faster decoding. Per token the
    1.7B and 7B 4-bit models remain close (45 vs 50 ms).
- Memory: 1.4 GiB vs 4.0 GiB peak.

## Y3: GGUF export (llama.cpp b12400)

**Export.** `scripts/export_gguf.sh` converts the merged bf16 model to GGUF (`convert_hf_to_gguf.py`)
and quantises it with `llama-quantize`. Both tools come from llama.cpp tag b12400, built by
`scripts/setup_llamacpp.sh`.

| File | Size | sha256 (first 16) |
|---|---|---|
| v2-900-merged-BF16.gguf | 3.45 GB | aac9f593a76a1d45 |
| v2-900-merged-Q8_0.gguf | 1.83 GB | 2d273bedd10c5819 |
| v2-900-merged-Q4_K_M.gguf | 1.11 GB | e16a0732a5a58a91 |

**Evaluation on dev-200, CPU** (`scripts/eval_gguf.py`):
- Runs through `llama-server` (8 threads, i9-12900H in WSL2) with greedy decoding, batch 1.
- Uses the same HF chat template as every other evaluation. The prompt goes to `/completion` as text.
- Pairs are on the same 200 questions. The merged-model rows come from their full-dev runs.

| Model | dev-200 EX | Format fail | vs merged bf16 (w / l, p) | CPU median / p95 per query | ms / output token |
|---|---|---|---|---|---|
| merged bf16 (GPU, HF) | 64.5% | — | | | |
| GGUF **Q8_0** | **63.5%** | 2 | 2 / 4, 0.69 | 4.5 s / 10.1 s | 130 |
| GGUF Q4_K_M | 52.0% | 23 | 6 / 31, <0.001 | 2.8 s / 6.3 s | 87 |

Sources: `results/dev200_gguf-v2-900-merged-Q8_0.json`, `results/dev200_gguf-v2-900-merged-Q4_K_M.json`.

- **Q8_0 is faithful.** It disagrees with the bf16 merged model on only 6 of 200 questions.
- **Q4_K_M breaks the model: −12.5pp vs bf16 (p<0.001).** 14 of its 31 losses are format failures:
  it answers with bare SQL instead of a ```sql block, which the evaluation (and the demo) require.
  - The cause is visible in the first output token. Q8_0 puts p≈1.0 on ```` ``` ````, while Q4_K_M
    splits it 0.59 / 0.41 with `SELECT` in llama-server.
  - In llama-cpp-python 0.3.16, whose bundled llama.cpp is older, Q4_K_M even prefers `SELECT`
    (0.83). The release therefore ships **Q8_0** as its GGUF. Q4_K_M is not published.
- CPU latency includes prompt processing: prompts are 300–900 tokens with the full schema. The
  demo gate (Y5) is measured separately on the demo's own code path.

## Y5: demo pipeline and the CPU gate

**Demo** (`demo/app.py`, Gradio):
- Pick a database (6 small Spider dev DBs in `demo/databases/`), then type a question or record/upload audio.
- Pipeline: faster-whisper `small` (int8, CPU) → the Q8_0 GGUF through llama-cpp-python →
  read-only execution (row cap 200, 10 s timeout) → result table and a one-sentence answer.
- The core, `tinysql.demo.answer_question`, is model-agnostic and tested (`tests/test_demo.py`).
- **Three llama-cpp-python pitfalls were fixed so the demo decodes like the evaluation:**
  1. `repeat_penalty` defaults to 1.1, and `top_p`/`min_p` filters are on.
  2. Prompts are passed as token ids from the HF tokenizer, so the chat-format special tokens are
     the evaluated ones.
  3. Q4_K_M is not used (see Y3).
- **Spoken answer:** not included. The answer is shown as text (the plan allowed text-only if TTS
  did not fit the CPU budget; TTS was not attempted).
- The example questions are ones the model answers correctly, checked by hand. The UI says it is
  wrong on roughly 4 in 10 questions.

**CPU gate** (`scripts/bench_demo.py`, demo code path, Q8_0, i9-12900H in WSL2):
- Text-to-SQL is timed on the first 30 clean SpokenSpider transcripts.
- Speech-to-text is timed on four 5-second clips, 3 runs each. The SpokenSpider audio itself was
  not kept by voice-to-sql.

| Threads | Prefix cache | Text-to-SQL + query median (p95) | Speech-to-text median | End-to-end median |
|---|---|---|---|---|
| 2 | reused (same DB as previous question) | 1.8 s (4.0) | 1.5 s | **3.3 s** |
| 2 | **cold** (reset before every question) | 4.8 s (6.1) | 1.5 s | **6.3 s** |
| 8 | reused | 1.2 s (3.5) | 1.4 s | 2.5 s |

Sources: `results/demo_latency_*.json`.

- **Gate: passed locally.** At 2 threads with a cold cache (the worst case), the median is 6.3 s,
  under the ~15 s limit. So the Space ships with **voice enabled**.
- 2 threads of an i9-12900H are faster than 2 free-tier vCPUs, so the gate is re-checked on the
  live Space (Y6). If it fails there, the fallback is text-only.
- llama-cpp-python reuses the schema prefix between consecutive questions on the same database.
  This is why follow-up questions are ~2.7× faster than the first one.
