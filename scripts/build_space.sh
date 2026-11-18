#!/bin/bash
# Assemble the Hugging Face Space folder (PLAN-week4 Y6) from the repo: demo app + the tinysql modules
# it imports + the demo databases. Output: build/space/ (gitignored), uploaded by scripts/upload_space.py.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/build/space"
rm -rf "$OUT" && mkdir -p "$OUT/tinysql" "$OUT/databases"
cp "$REPO/space/README.md" "$OUT/README.md"
cp "$REPO/demo/runtime.py" "$OUT/runtime.py"
# demo_app.py is demo/app.py unchanged; app.py only sets the Space's model defaults, then runs it.
cp "$REPO/demo/app.py" "$OUT/demo_app.py"
cat > "$OUT/app.py" <<'PY'
"""Hugging Face Space entry point: set the model defaults, then run demo_app.py (= demo/app.py)."""
import os
import runpy
from pathlib import Path

os.environ.setdefault("TINYSQL_GGUF", "Alpericon/tiny-sql-rl:gguf/tiny-sql-rl-Q8_0.gguf")
os.environ.setdefault("TINYSQL_TOKENIZER", "Alpericon/tiny-sql-rl")
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
runpy.run_path(str(Path(__file__).with_name("demo_app.py")), run_name="__main__")
PY
for m in __init__ demo executor prompt reward compare; do cp "$REPO/src/tinysql/$m.py" "$OUT/tinysql/"; done
cp "$REPO"/demo/databases/*.sqlite "$OUT/databases/"
grep -v '^#' "$REPO/demo/requirements.txt" | grep -v '^$' > "$OUT/requirements.txt"
printf 'build-essential\ncmake\n' > "$OUT/packages.txt"
find "$OUT" -type f | sed "s|$OUT/||" | sort
