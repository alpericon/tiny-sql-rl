"""Voice EX: our model vs voice-to-sql's SQLCoder on the same ASR transcripts (PLAN-week3 X6).

For each condition (original question text, then ASR transcripts at clean / 20 / 10 / 0 dB SNR)
our model generates SQL with the Spider dev decoding (greedy, 4-bit, batch 8). SQLCoder's stored
SQL from voice-to-sql is re-scored with the same comparator (tinysql.evaluate.score_sql), so both
systems are measured by one EX. Paired exact McNemar per condition.

    ~/venvs/tinysql/bin/python scripts/eval_voice.py --adapter checkpoints/v2/checkpoint-900

data/spokenspider_transcripts.csv is built once from voice-to-sql's data/metadata.csv:
    python scripts/eval_voice.py --build-csv <voice-to-sql>/data/metadata.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "spider" / "database"
VOICE_CSV = ROOT / "data" / "spokenspider_transcripts.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--repair", action="store_true", help="one retry on exec error / missing SQL block (Y9)")
    ap.add_argument("--ground", action="store_true", help="ground string literals in DB values, for ours and SQLCoder (PLAN-week6)")
    ap.add_argument("--sample-rows", type=int, default=0, help="example rows per table in the prompt (PLAN-week7)")
    ap.add_argument("--build-csv", type=Path, default=None, help="voice-to-sql metadata.csv -> " + str(VOICE_CSV))
    args = ap.parse_args()

    from tinysql.evaluate import score_sql
    from tinysql.ground import cached_index, ground_sql
    from tinysql.stats import mcnemar_exact
    from tinysql.voice import CONDITIONS, load_voice, slim_metadata, voice_examples, voice_rows

    if args.build_csv is not None:
        slim_metadata(args.build_csv, VOICE_CSV)
        print(f"wrote {VOICE_CSV}")
        return

    import torch

    from tinysql.evaluate import evaluate
    from tinysql.hf import HFGenerator

    assert torch.cuda.is_available(), "CUDA not available: is the CPU torch wheel installed?"
    rows = load_voice(VOICE_CSV)
    generate = HFGenerator(args.model, args.adapter, max_new_tokens=args.max_new_tokens)

    all_records, summary = [], {"model": args.model, "adapter": str(args.adapter) if args.adapter else None,
                                "decoding": "greedy, nf4 4-bit, thinking off", "repair": args.repair, "ground": args.ground, "sample_rows": args.sample_rows, "conditions": {}}
    t0 = time.perf_counter()
    for cond in CONDITIONS:
        meta = voice_rows(rows, cond)
        records, s = evaluate(voice_examples(rows, cond), DATABASE, generate, batch_size=args.batch_size,
                                      repair=args.repair, ground=args.ground,
                                      sample_rows=args.sample_rows)
        for r, m in zip(records, meta):
            db_file = DATABASE / m["db_id"] / f"{m['db_id']}.sqlite"
            coder_sql = m["sqlcoder_sql"]
            if args.ground:  # same post-processing for SQLCoder's stored SQL, so the comparison is model vs model
                coder_sql, _ = ground_sql(coder_sql, cached_index(str(db_file)))
            sc = score_sql(db_file, m["gold"], coder_sql)
            r.update(condition=cond, query_id=m["query_id"], wer=float(m["wer"]),
                     sqlcoder_sql=coder_sql, sqlcoder_exec_ok=sc["exec_ok"], sqlcoder_match=sc["match"])
        n = len(records)
        wins = sum(r["match"] and not r["sqlcoder_match"] for r in records)
        losses = sum(r["sqlcoder_match"] and not r["match"] for r in records)
        summary["conditions"][cond] = {
            "n": n,
            "ours_ex": s["execution_accuracy"],
            "ours_format_fail": s["format_failure_rate"],
            "ours_exec_error": s["execution_error_rate"],
            "sqlcoder_ex": sum(r["sqlcoder_match"] for r in records) / n,
            "sqlcoder_exec_error": sum(not r["sqlcoder_exec_ok"] for r in records) / n,
            "ours_wins": wins, "ours_losses": losses, "mcnemar_p": mcnemar_exact(wins, losses),
            "mean_wer": sum(r["wer"] for r in records) / n,
        }
        print(cond, json.dumps(summary["conditions"][cond]), flush=True)
        all_records += records
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    summary["hit_max_new_tokens_rate"] = sum(generate.hit_limit) / len(generate.hit_limit)

    tag = "voice_" + args.model.replace("/", "__") + (
        f"_{args.adapter.parent.name}-{args.adapter.name}" if args.adapter else "") + ("_repair" if args.repair else "") + ("_ground" if args.ground else "") + (f"_rows{args.sample_rows}" if args.sample_rows else "")
    (ROOT / "results" / f"{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (ROOT / "results" / f"{tag}.records.jsonl").open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
