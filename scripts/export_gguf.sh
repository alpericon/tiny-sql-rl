#!/bin/bash
# Export the merged model to GGUF (bf16) and quantize to Q4_K_M and Q8_0 (PLAN-week4 Y3).
# Needs ~/llama.cpp built by scripts/setup_llamacpp.sh. The converter runs with the training venv's
# torch/transformers and llama.cpp's own gguf-py (not pip-installed).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MERGED="${1:-$REPO/checkpoints/v2-900-merged}"
NAME="${2:-v2-900-merged}"
OUT="$HOME/gguf"
LLAMA="$HOME/llama.cpp"
mkdir -p "$OUT"

if [ ! -f "$OUT/$NAME-BF16.gguf" ]; then
  PYTHONPATH="$LLAMA/gguf-py" ~/venvs/tinysql/bin/python "$LLAMA/convert_hf_to_gguf.py" "$MERGED" \
    --outtype bf16 --outfile "$OUT/$NAME-BF16.gguf" 2>&1 | tail -2
fi
for q in Q8_0 Q4_K_M; do
  [ -f "$OUT/$NAME-$q.gguf" ] || "$LLAMA/build/bin/llama-quantize" "$OUT/$NAME-BF16.gguf" "$OUT/$NAME-$q.gguf" "$q" 2>&1 | tail -1
done
ls -la "$OUT"/"$NAME"-*.gguf
sha256sum "$OUT"/"$NAME"-*.gguf
