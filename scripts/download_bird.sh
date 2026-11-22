#!/usr/bin/env bash
# Download BIRD dev (questions + SQLite databases) into ~/bird/dev (PLAN-week12 B1).
# Source: the "Dev Set" link on the official page https://bird-bench.github.io (CC BY-SA 4.0).
# Kept on the WSL filesystem, not in the repo: SQLite over /mnt/c is slow and the databases are large.
# Usage (inside WSL): bash scripts/download_bird.sh
set -euo pipefail

URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
EXPECTED_BYTES=346207293       # Content-Length when the plan was written (2026-11-22)
EXPECTED_ETAG='"04B4AF221C9186361F09B16ABFD917EC"'
BIRD="$HOME/bird"
ZIP="$BIRD/download/dev.zip"

mkdir -p "$BIRD/download"
ETAG="$(curl -sI --max-time 60 "$URL" | tr -d '\r' | awk -F': ' 'tolower($1)=="etag"{print $2}')"
echo "etag:   $ETAG"
[ "$ETAG" = "$EXPECTED_ETAG" ] || { echo "ETag changed since the plan; stop and check the release notes"; exit 1; }
if [ ! -s "$ZIP" ]; then
  curl -L --fail --max-time 3600 -o "$ZIP" "$URL"
fi
BYTES="$(stat -c %s "$ZIP")"
echo "bytes:  $BYTES"
[ "$BYTES" = "$EXPECTED_BYTES" ] || { echo "unexpected size"; exit 1; }
echo "sha256: $(sha256sum "$ZIP" | cut -d' ' -f1)"

# Extract with Python's zipfile (it drops absolute paths and '..' components); nothing is executed.
rm -rf "$BIRD/unzipped" && mkdir -p "$BIRD/unzipped"
python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$BIRD/unzipped"
DEV="$(dirname "$(find "$BIRD/unzipped" -name dev.json -not -path '*__MACOSX*' | head -1)")"
INNER="$(find "$DEV" -maxdepth 1 -name 'dev_databases.zip' | head -1)"
if [ -n "$INNER" ]; then
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$INNER" "$DEV"
fi
rm -rf "$BIRD/dev" && mv "$DEV" "$BIRD/dev"
find "$BIRD/dev" -name '__MACOSX' -prune -exec rm -rf {} +
echo "dev questions: $(python3 -c "import json;print(len(json.load(open('$BIRD/dev/dev.json'))))")"
echo "databases:     $(find "$BIRD/dev" -name '*.sqlite' | wc -l)"
