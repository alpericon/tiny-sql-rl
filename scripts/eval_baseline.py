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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tinysql.data import load_spider_split
from tinysql.evaluate import evaluate
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
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available: is the CPU torch wheel installed?"

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.bfloat16, device_map="cuda:0"
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    prompt_tokens: list[int] = []
    hit_limit: list[bool] = []

    def render(messages):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:  # Qwen3-style models: disable the thinking block for a fair, fast baseline
            return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tok.apply_chat_template(messages, **kwargs)

    @torch.inference_mode()
    def generate(batch):
        texts = [render(m) for m in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        prompt_tokens.extend(lengths)
        too_long = [n for n in lengths if n > args.max_prompt_tokens]
        if too_long:
            raise RuntimeError(f"prompt of {max(too_long)} tokens exceeds --max-prompt-tokens; refusing to truncate")
        # Override each model's generation_config sampling defaults (e.g. Qwen2.5-Coder ships
        # repetition_penalty=1.1, which penalises re-using schema names) so decoding is identical.
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1]:]
        cfg_eos = model.generation_config.eos_token_id
        cfg_eos = cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos]  # int for some models
        stop_ids = {t for t in (tok.eos_token_id, tok.pad_token_id, *cfg_eos) if t is not None}
        for row in new.tolist():
            hit_limit.append(len(row) == args.max_new_tokens and not any(t in stop_ids for t in row))
        return tok.batch_decode(new, skip_special_tokens=True)

    dev = load_spider_split(SPIDER / "dev.json")
    if args.subset is not None:
        dev = load_subset(dev, json.loads(args.subset.read_text(encoding="utf-8")))
    dev = dev[: args.limit]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    records, summary = evaluate(dev, SPIDER / "database", generate, batch_size=args.batch_size)
    assert len(hit_limit) == len(records)
    for r, cut in zip(records, hit_limit):
        r["hit_max_new_tokens"] = cut
    summary.update(
        decoding="greedy, repetition_penalty=1.0",
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
