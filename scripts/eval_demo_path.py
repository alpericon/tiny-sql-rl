"""dev-200 EX of the demo's own generation path (demo/runtime.py: Q8_0 GGUF via llama-cpp-python).

The GGUF accuracy in results/week4_results.md (Y3) was measured through llama-server; the demo uses
the llama-cpp-python binding, whose bundled llama.cpp is older. This checks that the demo path
decodes the same model: same evaluate() harness, batch 1, dev-200.

    TINYSQL_GGUF=~/gguf/v2-900-merged-Q8_0.gguf TINYSQL_TOKENIZER=checkpoints/v2-900-merged \
      ~/venvs/demo/bin/python scripts/eval_demo_path.py --threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.environ["TINYSQL_THREADS"] = str(args.threads)

    from runtime import load_generator

    from tinysql.data import load_spider_split
    from tinysql.evaluate import evaluate
    from tinysql.subset import load_subset

    dev = load_subset(load_spider_split(ROOT / "data" / "spider" / "dev.json"),
                      json.loads((ROOT / "data" / "dev200_ids.json").read_text(encoding="utf-8")))
    records, summary = evaluate(dev, ROOT / "data" / "spider" / "database", load_generator(), batch_size=1)
    summary.update(runtime="llama-cpp-python (demo/runtime.py)", gguf=Path(os.environ["TINYSQL_GGUF"]).name,
                   threads=args.threads)
    tag = f"dev200_demo-path-{Path(os.environ['TINYSQL_GGUF']).stem}"
    (ROOT / "results" / f"{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (ROOT / "results" / f"{tag}.records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("execution_accuracy", "format_failure_rate", "execution_error_rate")}))


if __name__ == "__main__":
    main()
