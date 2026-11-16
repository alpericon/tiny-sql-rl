"""Base-model pass rate per training question, for the GRPO difficulty filter (PLAN-week3 X2).

Samples G completions per question with the training decoding (temperature 1.0, no top-p/top-k,
256 new tokens, thinking off) through vLLM (Unsloth fast_inference, no LoRA), scores each with the
training reward, and appends {db_id, question, gold, n_correct, n} lines. Re-running skips
questions already written, so an interrupted run resumes.

    ~/venvs/tinysql/bin/python scripts/pass_rate.py
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before transformers)

import argparse
import json
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from vllm import SamplingParams

from tinysql.data import load_exclusions, load_spider_split
from tinysql.grpo_data import build_grpo_records
from tinysql.reward import FULL, sql_reward

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=256, help="questions per vLLM call")
    ap.add_argument("--vllm-mem", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N questions")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train_pass_rate.jsonl")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA not available: is the CPU torch wheel installed?"

    # Same records the v2 run trains on before filtering: empty-gold questions are already dropped.
    records = build_grpo_records(
        load_spider_split(ROOT / "data" / "train_filtered.json"),
        ROOT / "data" / "spider" / "database",
        load_exclusions(ROOT / "data" / "spokenspider_exclude.json"),
        drop_empty_gold=True,
    )[: args.limit]
    done = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            done = {(r["db_id"], r["question"], r["gold"]) for r in map(json.loads, f)}
    todo = [r for r in records if (r["db_id"], r["question"], r["gold"]) not in done]
    print(f"{len(records)} questions, {len(done)} already done, {len(todo)} to sample", flush=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2304 + args.max_completion_length,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=args.vllm_mem,
    )
    params = SamplingParams(
        n=args.num_generations, temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=args.max_completion_length, seed=args.seed,
    )
    t0 = time.perf_counter()
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start : start + args.chunk]
        texts = [
            tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for r in chunk
        ]
        outs = model.fast_generate(texts, sampling_params=params)
        with args.out.open("a", encoding="utf-8") as f:
            for r, out in zip(chunk, outs):
                rewards = [sql_reward(o.text, r["gold"], r["db_path"]) for o in out.outputs]
                f.write(json.dumps({
                    "db_id": r["db_id"], "question": r["question"], "gold": r["gold"],
                    "n_correct": sum(x == FULL for x in rewards), "n": len(rewards),
                }, ensure_ascii=False) + "\n")
        done_n = start + len(chunk)
        rate = (time.perf_counter() - t0) / done_n
        print(f"{done_n}/{len(todo)}  {rate:.2f} s/question  eta {rate * (len(todo) - done_n) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
