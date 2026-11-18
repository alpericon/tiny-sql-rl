"""Greedy HF generation used by every evaluation script (moved out of scripts/eval_baseline.py
unchanged, so eval_voice.py decodes exactly like the Spider dev evaluation)."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HFGenerator:
    """``generate(batch_of_messages) -> completions``; records prompt lengths and 256-token hits."""

    def __init__(self, model: str, adapter: Path | None = None, no_4bit: bool = False,
                 max_new_tokens: int = 256, max_prompt_tokens: int = 3072):
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.prompt_tokens: list[int] = []
        self.hit_limit: list[bool] = []

        tok = AutoTokenizer.from_pretrained(model)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        quant = None if no_4bit else BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        m = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=quant, dtype=torch.bfloat16, device_map="cuda:0"
        )
        if adapter is not None:
            from peft import PeftModel

            m = PeftModel.from_pretrained(m, str(adapter))
        m.eval()
        self.tok, self.model = tok, m

    def render(self, messages):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:  # Qwen3-style models: disable the thinking block for a fair, fast baseline
            return self.tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tok.apply_chat_template(messages, **kwargs)

    @torch.inference_mode()
    def __call__(self, batch):
        tok, model = self.tok, self.model
        texts = [self.render(m) for m in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        self.prompt_tokens.extend(lengths)
        too_long = [n for n in lengths if n > self.max_prompt_tokens]
        if too_long:
            raise RuntimeError(f"prompt of {max(too_long)} tokens exceeds --max-prompt-tokens; refusing to truncate")
        # Override each model's generation_config sampling defaults (e.g. Qwen2.5-Coder ships
        # repetition_penalty=1.1, which penalises re-using schema names) so decoding is identical.
        out = model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
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
            self.hit_limit.append(len(row) == self.max_new_tokens and not any(t in stop_ids for t in row))
        return tok.batch_decode(new, skip_special_tokens=True)
