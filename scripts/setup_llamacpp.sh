#!/bin/bash
# Build llama.cpp (CPU) in WSL for GGUF export and CPU evaluation (PLAN-week4 Y3).
# cmake comes from a small uv venv because the system has no cmake and no sudo is assumed.
set -euo pipefail
LLAMA="$HOME/llama.cpp"
LLAMA_COMMIT="${LLAMA_COMMIT:-b12400}"   # pinned tag; recorded in results with every GGUF number
TOOLS="$HOME/venvs/buildtools"

[ -d "$TOOLS" ] || ~/.local/bin/uv venv -q --python 3.12 "$TOOLS"
~/.local/bin/uv pip install -q --python "$TOOLS/bin/python" cmake ninja

[ -d "$LLAMA" ] || git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA"
git -C "$LLAMA" fetch -q --tags
git -C "$LLAMA" checkout -q "$LLAMA_COMMIT"
PATH="$TOOLS/bin:$PATH" cmake -S "$LLAMA" -B "$LLAMA/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null
PATH="$TOOLS/bin:$PATH" cmake --build "$LLAMA/build" --target llama-quantize llama-server llama-cli -j 8 > /dev/null
git -C "$LLAMA" describe --tags --always
ls "$LLAMA/build/bin"
