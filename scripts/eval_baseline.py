"""Zero-shot Spider dev evaluation of a Hugging Face chat model (PLAN T8).

Runs inside WSL with the GPU venv:
    ~/venvs/tinysql/bin/python scripts/eval_baseline.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct

Writes results/baseline_<model>.json (summary) and results/baseline_<model>.records.jsonl.
Use --limit for a quick smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tinysql.data import load_spider_split
from tinysql.evaluate import evaluate
from tinysql.hf import HFGenerator
from tinysql.subset import load_subset

ROOT = Path(__file__).resolve().parents[1]
SPIDER = ROOT / "data" / "spider"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--subset", type=Path, default=None, help="ids file, e.g. data/dev200_ids.json")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir (GRPO checkpoint)")
    ap.add_argument("--tag", default=None, help="output name; default derives from model/adapter/subset")
    ap.add_argument("--repair", action="store_true", help="one retry on exec error / missing SQL block (PLAN-week4 Y9)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available: is the CPU torch wheel installed?"

    generate = HFGenerator(args.model, args.adapter, args.no_4bit, args.max_new_tokens, args.max_prompt_tokens)
    prompt_tokens, hit_limit = generate.prompt_tokens, generate.hit_limit

    dev = load_spider_split(SPIDER / "dev.json")
    if args.subset is not None:
        dev = load_subset(dev, json.loads(args.subset.read_text(encoding="utf-8")))
    dev = dev[: args.limit]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    records, summary = evaluate(dev, SPIDER / "database", generate, batch_size=args.batch_size, repair=args.repair)
    if not args.repair:  # with repair, retries interleave with first attempts; only the overall rate is kept
        assert len(hit_limit) == len(records)
        for r, cut in zip(records, hit_limit):
            r["hit_max_new_tokens"] = cut
    summary.update(
        decoding="greedy, repetition_penalty=1.0",
        repair=args.repair,
        hit_max_new_tokens_rate=sum(hit_limit) / len(hit_limit),
        model=args.model,
        adapter=str(args.adapter) if args.adapter else None,
        subset=str(args.subset) if args.subset else None,
        quantization="none" if args.no_4bit else "nf4-4bit",
        wall_seconds=round(time.perf_counter() - t0, 1),
        max_prompt_tokens=max(prompt_tokens),
        peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )

    tag = args.tag or (
        ("dev200" if args.subset else "baseline")
        + "_"
        + args.model.replace("/", "__")
        + (f"_{args.adapter.parent.name}-{args.adapter.name}" if args.adapter else "")
        + ("_repair" if args.repair else "")
        + (f"_limit{args.limit}" if args.limit else "")
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (out_dir / f"{tag}.records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
