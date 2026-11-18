#!/bin/bash
# Create the demo venv (CPU) and run the demo against the published model on the Hugging Face Hub.
#   bash scripts/setup_demo.sh          # install (first time; llama-cpp-python compiles, ~5-10 min)
#   bash scripts/setup_demo.sh run      # start http://127.0.0.1:7860
# Needs: uv (https://docs.astral.sh/uv/), a C/C++ compiler. cmake is installed into the venv by pip.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${TINYSQL_DEMO_VENV:-$HOME/venvs/demo}"

if [ "${1:-}" != "run" ]; then
  uv venv -q --python 3.12 "$VENV"
  CMAKE_ARGS="-DGGML_NATIVE=OFF" uv pip install --python "$VENV/bin/python" \
    -r <(grep -v '^#' "$REPO/demo/requirements.txt") -e "$REPO"
  echo "installed; start with: bash scripts/setup_demo.sh run"
  exit 0
fi

# Defaults: the released Q8_0 GGUF and tokenizer from the Hub (downloaded once, ~1.8 GB).
export TINYSQL_GGUF="${TINYSQL_GGUF:-Alpericon/tiny-sql-rl:gguf/tiny-sql-rl-Q8_0.gguf}"
export TINYSQL_TOKENIZER="${TINYSQL_TOKENIZER:-Alpericon/tiny-sql-rl}"
cd "$REPO"
exec "$VENV/bin/python" demo/app.py
