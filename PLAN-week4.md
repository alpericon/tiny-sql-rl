# Week 4 implementation plan: official EX, release model, demo, README (from SPEC.md §5.4, §6–8)

**Status: APPROVED 2026-11-18.**

Goal of the week: ship it. That means the official Spider number, a published model that matches the
reported numbers, a working demo and a README whose every number traces to a result file.

## Where week 3 left off

- The best model is **v2-900**: 58.5% EX on full dev (our comparator) vs 53.7% base.
  - vs the unfiltered main run: +1.5pp, not significant.
  - vs SQLCoder-7B on voice: ties from clean to 10 dB, behind at 0 dB.
- **Latency:** the unmerged LoRA doubles ms/token, so the published model must be merged.
- **Open from SPEC §5.4:** EX is not yet measured with Spider's official evaluation script.

## Decisions (2026-11-18, user)

- **Demo:** a voice Space with a text fallback. Build the full voice pipeline and measure it on CPU first.
  If one query takes more than ~15 s median on the free tier, the Space goes text-only and the repo gets a
  recorded video of the local GPU voice demo.
- **Official EX:** re-score the stored predictions with the official script and report it next to our
  comparator. No regeneration is needed.
- **Release:** only v2-900 is published (adapter, merged model, GGUF). The other runs stay as numbers in
  the README.

Same rules as before:
- CPU code is test-first.
- GPU work runs in WSL with `~/venvs/tinysql`, and long jobs run in a terminal tab.
- Model comparisons are paired, with exact McNemar.
- Anything published (HF model, Space, profile README, making the GitHub repo public) is done only after
  the user's explicit go-ahead at that step.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| Y1 | **Official Spider EX** for base, main-1000, v2-900 and v2-binary-800 on full dev | `scripts/official_eval.py`, `src/tinysql/official.py` (+ test), `third_party/test-suite-sql-eval` (pinned commit, not vendored if a pip/git install works), `results/official_ex.json` | Test-first: records → predictions/gold files (one query per line; a format failure becomes a placeholder that scores 0, never a dropped line; order and count checked against dev). Official exec EX is run with the standard Spider DBs. Test-suite EX is added if its database download is feasible; otherwise this is stated. The table in `week3_results.md` gets an "official EX" column. Any disagreement with our comparator is explained by counting and reading ~10 example rows. |
| Y2 | **Merge v2-900 and check it** | `scripts/merge_adapter.py`, `results/baseline_*_v2-900-merged*.json`, `results/latency_*merged*.json` | Merge into bf16. Full-dev EX of the merged model at bf16 and at nf4 (re-quantised), each paired vs the adapter version: the difference must be inside noise, or it is reported. Latency is re-measured with `bench_latency.py` (merged, nf4). The X7 hypothesis (~25% faster than SQLCoder) is confirmed or refuted. |
| Y3 | **GGUF export and check** | `scripts/export_gguf.sh` (llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`, pinned llama.cpp commit), `src/tinysql/llamacpp.py` (generate adapter for `evaluate()`), `results/dev200_*_gguf-q4_k_m.json` | Quantisation Q4_K_M, plus Q8_0 if Q4 loses accuracy. Uses the same chat template (`enable_thinking=False`) and greedy decoding. dev-200 EX on CPU is paired vs nf4 merged. CPU latency is measured (median, p95, tokens/s) with the CPU model and thread count stated. This number decides the demo gate in Y5. |
| — | **Review gate** (Y1–Y3) | | Reviewer approves |
| Y4 | **Hugging Face model repo** (v2-900) | `release/model_card.md`, `scripts/upload_model.py` | Contains the LoRA adapter, merged safetensors, GGUF file(s) and the model card. The card covers: training recipe (pass-rate filter, GRPO settings, reward), the eval table (our + official EX, voice table, latency), limitations (single seed, own comparator vs official, 3 voice DBs, SQLCoder caveats), license (Qwen3 Apache-2.0, Spider CC BY-SA 4.0) and intended use. Every number traces to a result file. **Upload only after the user's go-ahead (public/private asked then).** |
| Y5 | **Demo app** | `src/tinysql/demo.py` (pure pipeline + tests), `demo/app.py` (Gradio), `demo/requirements.txt`, `demo/databases/` (3–5 small Spider DBs) | Test-first core: question text → SQL (via an injected generate fn) → read-only execution with row cap and timeout (existing executor) → table → one-sentence answer. Errors are shown, never crash the app. Audio input: Whisper-small (faster-whisper, int8, CPU). Spoken answer: a small offline TTS if it fits the CPU budget, otherwise text answer only (stated). Measured locally on CPU first: if median end-to-end > ~15 s, the Space is **text-only** and the voice path stays local-only. |
| Y6 | **Space deploy** (or text-only fallback) + **voice demo video** | Space repo (separate), `docs/demo.md` | Space runs on the free CPU tier: GGUF via llama-cpp-python, at least 5 example questions, a DB picker. Verified by using the live Space. If voice is off on the Space, a short screen recording of the local GPU voice demo is linked from the README. **Deploy only after the user's go-ahead.** |
| Y7 | **README, figures, write-up** | `README.md`, `scripts/make_figures.py`, `docs/figures/*.png`, `docs/writeup.md` | The README covers: what/why, results table (RQ1–RQ4), figures (reward + zero-std curves main vs v2, dev-200 EX per checkpoint, EX per SNR, latency bars), how to reproduce (env, commands), limitations, links (model, Space). `make_figures.py` reads only `results/`. A ~1-page write-up. A check script compares README numbers against result files. |
| — | **Final review gate** | | Reviewer approves; README numbers verified |
| Y8 | **Publish** (each step asked separately) | | GitHub repo made public (history already uses the noreply email; the local `backup/pre-email-rewrite` branch is deleted first), profile README entry, links checked. |

Order:
- Y1 is CPU-only and runs first.
- Y2 then Y3 need the GPU and CPU in turn (~1.5 h total, mostly evaluations).
- Y5 can be built in parallel with Y2/Y3 against the HF model, then switched to GGUF.
- Y4 and Y6 wait for the review gate. Y7 is last.

## Risks

- **GGUF Q4_K_M loses accuracy** vs nf4. Mitigation: report it, ship Q8_0 as well (≈1.8 GB), and let
  the Space use whichever fits CPU latency.
- **The official script disagrees with ours** (it is not column-permutation tolerant and handles
  ORDER BY differently). Report both and do not replace the week-3 numbers; explain the gap with examples.
- **Test-suite databases are large** (several GB). If the download is impractical, report official exec
  EX on the standard DBs only and say so.
- **The free-tier CPU is too slow for voice** (2 vCPU): Whisper-small plus a 1.7B Q4 model plus TTS may
  exceed 15 s. This is handled by the Y5 gate (text-only Space + video), as SPEC §7 already allows.
- **llama-cpp-python build issues on Spaces.** Use a prebuilt CPU wheel; the fallback is a Docker Space.
- **Licensing.** Spider DBs are CC BY-SA 4.0 (attribution in the demo and README). The Qwen3 license
  and attribution go on the model card. SQLCoder numbers are quoted, not redistributed.

## Not in this week

Value normalisation for ASR literals, vLLM serving, multiple seeds and BIRD stay out of scope. They go in
the README under "future work".
