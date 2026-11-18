"""CPU latency of the demo pipeline, for the voice/text-only gate (PLAN-week4 Y5).

Times the demo's own code path (demo/runtime.py + tinysql.demo.answer_question) on CPU:
  * text-to-SQL + query: the first N clean SpokenSpider transcripts on their databases,
  * speech-to-text: Whisper-small int8 on the wav files given (each run 3 times),
and reports median / p95 for both and their sum as the end-to-end estimate. Run it with
--threads 2 to approximate the free Space tier (2 vCPU); the CPU model is recorded.

    TINYSQL_GGUF=~/gguf/v2-900-merged-Q4_K_M.gguf TINYSQL_TOKENIZER=checkpoints/v2-900-merged \
      ~/venvs/demo/bin/python scripts/bench_demo.py --threads 2 --wav a.wav b.wav
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
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--wav", nargs="*", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cold", action="store_true", help="forget the cached schema prefix before every query")
    args = ap.parse_args()
    os.environ["TINYSQL_THREADS"] = str(args.threads)

    from runtime import load_generator, load_transcriber, timed

    from tinysql.demo import answer_question
    from tinysql.stats import latency_summary
    from tinysql.voice import load_voice, voice_rows

    rows = voice_rows(load_voice(ROOT / "data" / "spokenspider_transcripts.csv"), "clean")[: args.n]
    gen = load_generator()
    db = lambda r: ROOT / "data" / "spider" / "database" / r["db_id"] / f"{r['db_id']}.sqlite"  # noqa: E731
    answer_question(rows[0]["transcript"], db(rows[0]), gen)  # warm-up, untimed
    sql_times, ok = [], 0
    for r in rows:
        if args.cold:
            gen.reset()
        res, dt = timed(answer_question, r["transcript"], db(r), gen)
        sql_times.append(dt)
        ok += res.error is None
    summary = {
        "gguf": Path(os.environ["TINYSQL_GGUF"]).name,
        "threads": args.threads,
        "prefix_cache": "reset before every query (cold)" if args.cold else "reused across consecutive same-database queries",
        "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
        "text_to_sql_and_query": latency_summary(sql_times),
        "queries_without_error": ok,
    }
    if args.wav:
        tr = load_transcriber()
        tr(args.wav[0])  # warm-up
        stt = [timed(tr, w)[1] for w in args.wav for _ in range(3)]
        summary["speech_to_text"] = latency_summary(stt)
        summary["end_to_end_median_s"] = summary["speech_to_text"]["median_s"] + summary["text_to_sql_and_query"]["median_s"]
    tag = args.tag or f"{Path(os.environ['TINYSQL_GGUF']).stem}_t{args.threads}" + ("_cold" if args.cold else "")
    (ROOT / "results" / f"demo_latency_{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
