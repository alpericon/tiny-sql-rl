#!/usr/bin/env bash
# Create the GPU training environment inside WSL2 (SPEC §3, PLAN T7).
# The venv lives on the WSL filesystem (fast I/O); the code stays on /mnt/c.
# Usage (inside WSL, from the repo root): bash scripts/setup_wsl.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/venvs/tinysql"

uv venv --python 3.12 "$VENV"
# Unsloth pins compatible torch / transformers / trl / peft / bitsandbytes versions.
uv pip install --python "$VENV/bin/python" unsloth
uv pip install --python "$VENV/bin/python" vllm || echo "WARN: vllm install failed; GRPO will fall back to HF generation"
# vLLM >= 0.28 moved bitsandbytes (4-bit) loading out of tree; Unsloth fast_inference needs it.
uv pip install --python "$VENV/bin/python" vllm-bnb-plugin || echo "WARN: vllm-bnb-plugin install failed"
uv pip install --python "$VENV/bin/python" -e "$REPO" pytest

"$VENV/bin/python" - <<'EOF'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
for m in ["transformers", "trl", "peft", "bitsandbytes", "unsloth", "vllm"]:
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "NOT AVAILABLE:", type(e).__name__, str(e)[:120])
EOF

"$VENV/bin/python" -m pip freeze 2>/dev/null > "$REPO/requirements-train.lock.txt" || uv pip freeze --python "$VENV/bin/python" > "$REPO/requirements-train.lock.txt"
echo "pinned versions written to requirements-train.lock.txt"
