#!/usr/bin/env bash
# Download Spider 1.0 (questions + SQLite databases) into data/spider/.
# Source: the official Yale distribution on Google Drive (linked from yale-lily.github.io/spider).
# Usage (from the repo root, inside WSL): bash scripts/download_spider.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

GDRIVE_ID="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$HOME/.cache/tinysql"
ZIP="$CACHE/spider_data.zip"

mkdir -p "$CACHE"
if [ ! -s "$ZIP" ]; then
  uvx gdown "$GDRIVE_ID" -O "$ZIP"
fi
# Checksum published by the HF re-host HAL-9001/spider-databases for the same Drive file.
EXPECTED_SHA256="00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
echo "$EXPECTED_SHA256  $ZIP" | sha256sum -c -
echo "size:   $(du -h "$ZIP" | cut -f1)"

rm -rf "$CACHE/unzipped" && mkdir -p "$CACHE/unzipped"
python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$CACHE/unzipped"
SRC="$(dirname "$(find "$CACHE/unzipped" -name train_spider.json | head -1)")"

mkdir -p "$REPO/data/spider"
cp "$SRC"/train_spider.json "$SRC"/dev.json "$SRC"/tables.json "$REPO/data/spider/"
rm -rf "$REPO/data/spider/database"
cp -r "$SRC/database" "$REPO/data/spider/database"
# Held-out test split (released 02/05/2024, PLAN-week11): only the 40 test databases are copied.
cp "$SRC"/test.json "$SRC"/test_tables.json "$REPO/data/spider/"
rm -rf "$REPO/data/spider/test_database" && mkdir -p "$REPO/data/spider/test_database"
for db in $(python3 -c "import json;print(' '.join(sorted({e['db_id'] for e in json.load(open('$SRC/test.json'))})))"); do
  cp -r "$SRC/test_database/$db" "$REPO/data/spider/test_database/$db"
done
echo "train: $(python3 -c "import json;print(len(json.load(open('$REPO/data/spider/train_spider.json'))))")"
echo "dev:   $(python3 -c "import json;print(len(json.load(open('$REPO/data/spider/dev.json'))))")"
echo "dbs:   $(ls "$REPO/data/spider/database" | wc -l)"
echo "test:  $(python3 -c "import json;print(len(json.load(open('$REPO/data/spider/test.json'))))") questions, $(ls "$REPO/data/spider/test_database" | wc -l) dbs"
