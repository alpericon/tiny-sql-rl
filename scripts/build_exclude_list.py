"""Build data/spokenspider_exclude.json from voice-to-sql's metadata.csv.

Usage: uv run python scripts/build_exclude_list.py <path/to/voice-to-sql/data/metadata.csv>
"""
import csv
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(__file__).resolve().parents[1] / "data" / "spokenspider_exclude.json"

seen = {}
with src.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        seen.setdefault(row["query_id"], {
            "query_id": row["query_id"],
            "db_id": row["db_id"],
            "question": row["question"],
            "sql": row["sql"],
        })

records = sorted(seen.values(), key=lambda r: r["query_id"])
out.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {len(records)} records to {out}")
