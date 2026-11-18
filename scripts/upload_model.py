"""Upload the v2-900 release to the Hugging Face Hub (PLAN-week4 Y4).

Layout of the model repo:
    README.md                      release/model_card.md
    config.json, model.safetensors, tokenizer*, chat_template.jinja, generation_config.json
                                   merged bf16 model (checkpoints/v2-900-merged)
    adapter/adapter_config.json, adapter/adapter_model.safetensors
                                   the LoRA adapter (checkpoints/v2/checkpoint-900; trained on the
                                   unsloth 4-bit Qwen3-1.7B base)
    gguf/tiny-sql-rl-Q8_0.gguf     llama.cpp b12400 export (Q4_K_M is not published, see week4_results.md)

    ~/venvs/demo/bin/python scripts/upload_model.py --repo <user>/tiny-sql-rl [--private]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "checkpoints" / "v2-900-merged"
ADAPTER = ROOT / "checkpoints" / "v2" / "checkpoint-900"
GGUF = Path.home() / "gguf" / "v2-900-merged-Q8_0.gguf"
MERGED_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja", "merge_info.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in MERGED_FILES:
            (stage / name).symlink_to(MERGED / name)
        (stage / "adapter").mkdir()
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            (stage / "adapter" / name).symlink_to(ADAPTER / name)
        (stage / "gguf").mkdir()
        (stage / "gguf" / "tiny-sql-rl-Q8_0.gguf").symlink_to(GGUF)
        shutil.copy(ROOT / "release" / "model_card.md", stage / "README.md")
        api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=str(stage))
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
