"""Model loading for the demo: GGUF via llama-cpp-python, Whisper via faster-whisper (CPU).

Prompts are rendered with the model's HF chat template (enable_thinking=False), as in every
evaluation, and decoding is greedy. Paths come from the environment so the same code runs locally
and on a Hugging Face Space:
    TINYSQL_GGUF       local .gguf path, or "<repo_id>:<filename>" to download from the Hub
    TINYSQL_TOKENIZER  local dir or Hub repo id holding the chat template (default: Qwen/Qwen3-1.7B)
    TINYSQL_THREADS    CPU threads for llama.cpp (default: all)
    TINYSQL_WHISPER    faster-whisper size (default: small)
"""

from __future__ import annotations

import os
import time
from functools import lru_cache


def _resolve_gguf(spec: str) -> str:
    if os.path.exists(spec):
        return spec
    repo_id, filename = spec.split(":", 1)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename)


@lru_cache(maxsize=1)
def load_generator():
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ.get("TINYSQL_TOKENIZER", "Qwen/Qwen3-1.7B"))
    threads = int(os.environ.get("TINYSQL_THREADS", os.cpu_count() or 2))
    llm = Llama(model_path=_resolve_gguf(os.environ["TINYSQL_GGUF"]), n_ctx=4096, n_threads=threads, verbose=False)

    def generate(batch):
        outs = []
        for messages in batch:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            # Pass token ids, not text: llama-cpp-python would tokenize <|im_start|> etc. as plain text,
            # so the model would not see the chat format it was trained and evaluated on.
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            # llama-cpp-python defaults to repeat_penalty=1.1 (and top_p/min_p filters); evaluation decodes
            # with no penalty, so neutralise them to keep the demo identical to the measured model.
            r = llm.create_completion(ids, max_tokens=256, temperature=0.0, top_k=1, top_p=1.0, min_p=0.0,
                                      repeat_penalty=1.0)
            outs.append(r["choices"][0]["text"])
        return outs

    # llama-cpp-python reuses the KV cache of a shared prompt prefix (same schema = same database);
    # reset() forgets it, for cold-start latency measurements.
    generate.reset = llm.reset
    return generate


@lru_cache(maxsize=1)
def load_transcriber():
    from faster_whisper import WhisperModel

    threads = int(os.environ.get("TINYSQL_THREADS", os.cpu_count() or 2))
    model = WhisperModel(os.environ.get("TINYSQL_WHISPER", "small"), device="cpu", compute_type="int8",
                         cpu_threads=threads)

    def transcribe(audio_path: str) -> str:
        segments, _ = model.transcribe(audio_path, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    return transcribe


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0
