"""Generation through a llama.cpp server, for GGUF evaluation on CPU (PLAN-week4 Y3).

The prompt text is rendered by the caller with the same HF chat template as every other evaluation
(``enable_thinking=False``) and sent to llama-server's raw ``/completion`` endpoint, so llama.cpp's
own template handling cannot change the input. Decoding is greedy (temperature 0, top_k 1).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable

Messages = list[dict[str, str]]


def http_post(url: str) -> Callable[[dict], dict]:
    def post(payload: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    return post


class LlamaServerGenerator:
    def __init__(self, render: Callable[[Messages], str], post: Callable[[dict], dict] | None = None,
                 url: str = "http://127.0.0.1:8080/completion", max_new_tokens: int = 256):
        self.render = render
        self.post = post or http_post(url)
        self.max_new_tokens = max_new_tokens
        self.hit_limit: list[bool] = []
        self.new_tokens: list[int] = []

    def __call__(self, batch: list[Messages]) -> list[str]:
        outs = []
        for messages in batch:
            r = self.post({
                "prompt": self.render(messages),
                "n_predict": self.max_new_tokens,
                "temperature": 0,
                "top_k": 1,
                "cache_prompt": False,
            })
            if "content" not in r:
                raise RuntimeError(f"llama-server error: {r.get('error', r)}")
            self.new_tokens.append(int(r.get("tokens_predicted", 0)))
            self.hit_limit.append(r.get("stop_type") == "limit")
            outs.append(r["content"])
        return outs
