"""BIRD dev evaluation (PLAN-week12 B3/B4), scored with BIRD's set-based EX (tinysql.bird.bird_ex).

    BIRD_DIR=~/bird/dev ~/venvs/tinysql/bin/python scripts/eval_bird.py --tag A-base
    ... --adapter checkpoints/v2/checkpoint-900 --tag B-v2-900
    ... --adapter checkpoints/v2/checkpoint-900 --repair --ground --sample-rows 3 --prune --tag E-v2-900-full

Writes results/bird_<tag>.json and results/bird_<tag>.records.jsonl (records carry question_id and
difficulty). With --report-only, writes results/bird_prompt_report.json (prompt lengths for the
schema-only and the full configuration, and pruner recall on BIRD gold) and generates nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from tinysql.bird import bird_ex, load_bird

ROOT = Path(__file__).resolve().parents[1]
BIRD = Path(os.environ.get("BIRD_DIR", Path.home() / "bird" / "dev"))
GUARD = 3072


def report(examples, db_dir):
    from transformers import AutoTokenizer

    from tinysql.prompt import build_messages, serialize_schema
    from tinysql.prune import gold_tables, profile, select_tables

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    n_tok = lambda m: len(tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                              add_special_tokens=False)["input_ids"])
    lens, kept, fewer, fallback, cache = defaultdict(list), 0, 0, 0, {}
    for e in examples:
        db = db_dir / e.db_id / f"{e.db_id}.sqlite"
        sel = select_tables(e.question, db)
        for name, rows, tables in [("A_schema_only", 0, None), ("E_rows3_pruned", 3, sel)]:
            key = (e.db_id, rows, None if tables is None else tuple(tables))
            if key not in cache:
                cache[key] = serialize_schema(db, rows, tables=tables)
            lens[name].append(n_tok(build_messages(cache[key], e.question)))
        if sel is None:
            fallback += 1
            kept += 1
        else:
            kept += gold_tables(e.query) <= {t.lower() for t in sel}
            fewer += len(sel) < len(profile(str(db)).names)
    n = len(examples)
    out = {k: {"median": statistics.median(v), "p95": sorted(v)[int(0.95 * (n - 1))], "max": max(v),
               f"over_{GUARD}": sum(x > GUARD for x in v), f"share_over_{GUARD}": round(sum(x > GUARD for x in v) / n, 4)}
           for k, v in lens.items()}
    out["pruner"] = {"recall": round(kept / n, 4), "fallback_rate": round(fallback / n, 4), "questions_with_fewer_tables": fewer}
    (ROOT / "results" / "bird_prompt_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=GUARD)
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--ground", action="store_true")
    ap.add_argument("--sample-rows", type=int, default=0)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    examples, meta = load_bird(BIRD / "dev.json")
    db_dir = BIRD / "dev_databases"
    if args.report_only:
        report(examples, db_dir)
        return
    if args.tag is None:
        raise SystemExit("--tag is required for a run")

    import torch

    from tinysql.evaluate import evaluate
    from tinysql.hf import HFGenerator

    examples, meta = examples[: args.limit], meta[: args.limit]
    generate = HFGenerator(args.model, args.adapter, False, args.max_new_tokens, args.max_prompt_tokens)
    t0 = time.perf_counter()
    records, summary = evaluate(examples, db_dir, generate, batch_size=args.batch_size, repair=args.repair,
                                ground=args.ground, sample_rows=args.sample_rows, prune=args.prune, scorer=bird_ex)
    for r, m in zip(records, meta):
        r.update(m)
    by = defaultdict(list)
    for r in records:
        by[r["difficulty"]].append(r["match"])
    summary.update(
        benchmark="BIRD dev", metric="BIRD EX (set of result rows)", model=args.model,
        adapter=str(args.adapter) if args.adapter else None, repair=args.repair, ground=args.ground,
        sample_rows=args.sample_rows, prune=args.prune, decoding="greedy, nf4, thinking off",
        max_new_tokens=args.max_new_tokens, limit=args.limit,
        by_difficulty={k: {"n": len(v), "ex": sum(v) / len(v)} for k, v in sorted(by.items())},
        max_prompt_tokens=max(generate.prompt_tokens), wall_seconds=round(time.perf_counter() - t0, 1),
        peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
    )
    tag = args.tag + (f"_limit{args.limit}" if args.limit else "")
    (ROOT / "results" / f"bird_{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (ROOT / "results" / f"bird_{tag}.records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
