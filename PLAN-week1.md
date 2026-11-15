# Week 1 implementation plan (from SPEC.md §6)

Goal of the week: everything needed to **score any model on Spider** safely and reproducibly, plus zero-shot baselines.
Everything in T1–T6 is pure Python (stdlib + pytest) and runs on native Windows. GPU work (T7–T8) runs in WSL2.

Each task is test-first: write the failing test → implement → `uv run pytest` green → commit.

| # | Task | Files | Done when |
|---|---|---|---|
| T1 | Repo skeleton | `pyproject.toml`, `src/tinysql/__init__.py`, `tests/`, `.gitignore`, `README.md` | `uv run pytest` runs (0 tests) |
| T2 | Sandboxed read-only executor | `src/tinysql/executor.py`, `tests/test_executor.py` | Tests pass: SELECT works; INSERT/UPDATE/DROP/ATTACH/PRAGMA writes rejected; multiple statements rejected; DB file unchanged after a write attempt; infinite query times out; row cap enforced |
| T3 | Result comparison | `src/tinysql/compare.py`, `tests/test_compare.py` | Order-insensitive multiset match by default; ordered match when gold SQL has `ORDER BY`; duplicates counted; columns may be permuted (changed week 2 after review) |
| T4 | Reward function | `src/tinysql/reward.py`, `tests/test_reward.py` | Extracts SQL from a fenced block; returns 1.0 / 0.1 / 0.0 per SPEC §5.2; `binary=True` mode for the ablation |
| T5 | Spider loader + leakage filter | `src/tinysql/data.py`, `data/spokenspider_exclude.json`, `tests/test_data.py` | Exclusion list built from voice-to-sql `metadata.csv` (100 unique `(db_id, question)`); filter matches by content, not index; test asserts exactly 100 removed and none of them remain |
| T6 | Schema serializer + prompt builder | `src/tinysql/prompt.py`, `tests/test_prompt.py` | `CREATE TABLE` text from a SQLite file; prompt length reported in tokens; a test fixture DB covers foreign keys |
| T7 | WSL2 environment | `scripts/setup_wsl.sh`, `requirements-train.txt` | `torch.cuda.is_available()` is True in WSL; pinned versions recorded |
| T8 | Eval harness + zero-shot baselines | `src/tinysql/evaluate.py`, `scripts/eval_baseline.py`, `results/baseline_*.json` | Spider dev EX for 2–3 candidates; max prompt length logged (no truncation); format-failure rate logged |

Review gate after T6 (the CPU-only core) and again after T8.

Notes
- The code lives on the Windows Desktop; inside WSL it is reached via `/mnt/c/Users/alper/Desktop/sql-rl`. Model weights and HF caches stay in the WSL home directory (faster I/O).
- This is a **laptop** 3070 Ti: log GPU temperature and clocks during long runs, since thermal throttling will affect the latency numbers.
