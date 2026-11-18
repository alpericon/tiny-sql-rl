"""Sample v2-900 on the training questions and build repair-turn GRPO records (PLAN-week5 Z2).

Same machinery as scripts/pass_rate.py (Unsloth fast_inference / vLLM, training decoding:
temperature 1.0, no top-p/top-k, 256 tokens, thinking off, 8 samples), plus the v2-900 LoRA adapter.

Outputs:
  data/train_repair_candidates.jsonl   every sample: db_id, question, gold, completion, reward
  data/train_repair.jsonl              repair records (tinysql.grpo_data.build_repair_records)

Adapter check: the share of fully correct samples is printed next to the base model's share on the
same questions (data/train_pass_rate.jsonl). If the adapter were not applied, they would match.

    ~/venvs/tinysql/bin/python scripts/sample_failures.py [--limit 64]
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before transformers)

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from vllm import SamplingParams

from tinysql.data import load_exclusions, load_spider_split
from tinysql.grpo_data import build_grpo_records, build_repair_records
from tinysql.reward import FULL, extract_sql, sql_reward

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, default=ROOT / "checkpoints" / "v2" / "checkpoint-900")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--vllm-mem", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N questions")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA not available: is the CPU torch wheel installed?"

    records = build_grpo_records(
        load_spider_split(ROOT / "data" / "train_filtered.json"),
        ROOT / "data" / "spider" / "database",
        load_exclusions(ROOT / "data" / "spokenspider_exclude.json"),
        drop_empty_gold=True,
    )[: args.limit]

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2304 + args.max_completion_length, load_in_4bit=True,
        fast_inference=True, max_lora_rank=16, gpu_memory_utilization=args.vllm_mem,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    lora = model.load_lora(str(args.adapter))
    params = SamplingParams(n=args.num_generations, temperature=1.0, top_p=1.0, top_k=-1,
                            max_tokens=args.max_completion_length, seed=args.seed)

    samples, t0 = [], time.perf_counter()
    for start in range(0, len(records), args.chunk):
        chunk = records[start : start + args.chunk]
        texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for r in chunk]
        outs = model.fast_generate(texts, sampling_params=params, lora_request=lora)
        for r, out in zip(chunk, outs):
            for o in out.outputs:
                samples.append({**{k: r[k] for k in ("db_id", "question", "gold", "db_path", "prompt")},
                                "completion": o.text, "reward": sql_reward(o.text, r["gold"], r["db_path"])})
        done = start + len(chunk)
        rate = (time.perf_counter() - t0) / done
        print(f"{done}/{len(records)}  {rate:.2f} s/question  eta {rate * (len(records) - done) / 60:.0f} min", flush=True)

    suffix = f"_limit{args.limit}" if args.limit else ""
    with open(ROOT / "data" / f"train_repair_candidates{suffix}.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({k: s[k] for k in ("db_id", "question", "gold", "completion", "reward")},
                               ensure_ascii=False) + "\n")
    repair = build_repair_records(samples, max_per_question=args.max_per_question)
    with open(ROOT / "data" / f"train_repair{suffix}.jsonl", "w", encoding="utf-8") as f:
        for r in repair:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Adapter check against the base model's pass@8 on the same questions.
    base = {(r["db_id"], r["question"], r["gold"]): r for r in map(json.loads, open(ROOT / "data" / "train_pass_rate.jsonl"))}
    keys = {(r["db_id"], r["question"], r["gold"]) for r in records}
    base_full = sum(base[k]["n_correct"] for k in keys) / sum(base[k]["n"] for k in keys)
    v2_full = sum(s["reward"] == FULL for s in samples) / len(samples)
    kinds = Counter("no block" if extract_sql(s["completion"]) is None else ("correct" if s["reward"] == FULL else
                    ("wrong result" if s["reward"] > 0 else "exec error")) for s in samples)
    print(json.dumps({
        "questions": len(records), "samples": len(samples), "sample_outcomes": dict(kinds),
        "repair_records": len(repair), "questions_with_repair_record": len({(r["db_id"], r["question"]) for r in repair}),
        "full_reward_rate_v2_900": round(v2_full, 4), "full_reward_rate_base_same_questions": round(base_full, 4),
    }, indent=1))


if __name__ == "__main__":
    main()
