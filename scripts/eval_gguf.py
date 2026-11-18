"""dev-200 EX and CPU latency of a GGUF export through llama-server (PLAN-week4 Y3).

Starts ~/llama.cpp/build/bin/llama-server on the GGUF file (CPU only), renders prompts with the
merged model's HF chat template (enable_thinking=False, as in every evaluation), decodes greedily
one query at a time, and scores with the same evaluate() harness. Per-query wall time is recorded,
so the run is also the CPU latency measurement.

    ~/venvs/tinysql/bin/python scripts/eval_gguf.py --gguf ~/gguf/v2-900-merged-Q4_K_M.gguf --threads 8
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from tinysql.data import load_spider_split
from tinysql.evaluate import evaluate
from tinysql.llamacpp import LlamaServerGenerator
from tinysql.stats import latency_summary
from tinysql.subset import load_subset

ROOT = Path(__file__).resolve().parents[1]
SPIDER = ROOT / "data" / "spider"
SERVER = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints" / "v2-900-merged"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "dev200_ids.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    render = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)  # noqa: E731

    server = subprocess.Popen(
        [str(SERVER), "-m", str(args.gguf), "-t", str(args.threads), "-c", "4096", "--port", str(args.port),
         "-np", "1", "--no-webui"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit("llama-server did not become healthy")

        gen = LlamaServerGenerator(render, url=f"http://127.0.0.1:{args.port}/completion")
        times: list[float] = []

        def timed(batch):
            t0 = time.perf_counter()
            out = gen(batch)
            times.append(time.perf_counter() - t0)
            return out

        dev = load_spider_split(SPIDER / "dev.json")
        dev = load_subset(dev, json.loads(args.subset.read_text(encoding="utf-8")))[: args.limit]
        gen([[{"role": "user", "content": "warm-up"}]])  # untimed
        gen.hit_limit.clear(); gen.new_tokens.clear()
        records, summary = evaluate(dev, SPIDER / "database", timed, batch_size=1)
    finally:
        server.terminate()
        server.wait(timeout=30)

    for r, cut in zip(records, gen.hit_limit):
        r["hit_max_new_tokens"] = cut
    llama_rev = subprocess.run(["git", "-C", str(SERVER.parents[2]), "describe", "--tags", "--always"],
                               capture_output=True, text=True).stdout.strip()
    summary.update(
        decoding="greedy (temperature 0, top_k 1), llama-server /completion, HF chat template",
        hit_max_new_tokens_rate=sum(gen.hit_limit) / len(gen.hit_limit),
        model=str(args.gguf.name), subset=str(args.subset.name), limit=args.limit,
        latency=latency_summary(times),
        new_tokens=latency_summary([float(n) for n in gen.new_tokens]),
        ms_per_token=1000 * sum(times) / max(sum(gen.new_tokens), 1),
        cpu=platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
        threads=args.threads, llama_cpp=llama_rev,
    )
    tag = f"dev200_gguf-{args.gguf.stem}" + (f"_limit{args.limit}" if args.limit else "")
    (ROOT / "results" / f"{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (ROOT / "results" / f"{tag}.records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("execution_accuracy", "latency", "ms_per_token", "llama_cpp")}, indent=1))


if __name__ == "__main__":
    main()
