#!/bin/bash
# Run the demo locally in WSL against the Q8_0 GGUF built by scripts/export_gguf.sh.
cd "$(dirname "$0")/.." || exit 1
export TINYSQL_GGUF="${TINYSQL_GGUF:-$HOME/gguf/v2-900-merged-Q8_0.gguf}"
export TINYSQL_TOKENIZER="${TINYSQL_TOKENIZER:-checkpoints/v2-900-merged}"
export TINYSQL_THREADS="${TINYSQL_THREADS:-4}"
exec ~/venvs/demo/bin/python demo/app.py
