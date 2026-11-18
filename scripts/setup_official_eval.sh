#!/bin/bash
# Set up Spider's official evaluation script for scripts/official_eval.py (PLAN-week4 Y1), in WSL.
#  - third_party/test-suite-sql-eval pinned to commit e97acc5 (not vendored into this repo)
#  - a small CPU venv with its dependencies (sqlparse, nltk + punkt)
#  - the 20 Spider dev databases copied to ~/spider_dev_db (sqlite over /mnt/c is ~10x slower)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TS="$REPO/third_party/test-suite-sql-eval"
[ -d "$TS" ] || git clone -q https://github.com/taoyds/test-suite-sql-eval "$TS"
git -C "$TS" checkout -q e97acc546ecbee8fa27fa8dbf025ef61493a876c

VENV="$HOME/venvs/officialeval"
[ -d "$VENV" ] || ~/.local/bin/uv venv -q --python 3.12 "$VENV"
~/.local/bin/uv pip install -q --python "$VENV/bin/python" sqlparse==0.5.3 nltk==3.9.1 tqdm pytest -e "$REPO"
"$VENV/bin/python" -c 'import nltk; nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)'

mkdir -p ~/spider_dev_db
for d in $("$VENV/bin/python" -c 'import json; print(" ".join(sorted({e["db_id"] for e in json.load(open("'"$REPO"'/data/spider/dev.json"))})))'); do
  [ -d ~/spider_dev_db/"$d" ] || cp -r "$REPO/data/spider/database/$d" ~/spider_dev_db/
done
echo "ok: $(ls ~/spider_dev_db | wc -l) dev databases"
