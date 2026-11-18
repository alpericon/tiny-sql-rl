"""Upload build/space/ (made by scripts/build_space.sh) as a Gradio Space (PLAN-week4 Y6).

    bash scripts/build_space.sh && ~/venvs/demo/bin/python scripts/upload_space.py --repo <user>/tiny-sql-rl-demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=str(ROOT / "build" / "space"),
                      commit_message="Deploy tiny-sql-rl demo")
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
