---
title: tiny-sql-rl
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
- Alpericon/tiny-sql-rl
short_description: Ask a Spider database a question, by text or voice
---

# tiny-sql-rl demo

Qwen3-1.7B trained with GRPO on an execution reward (59% official execution accuracy on Spider dev).
Pick a database, then type or say a question. The model writes SQL, which runs read-only on a
small Spider database.

- Model: [Alpericon/tiny-sql-rl](https://huggingface.co/Alpericon/tiny-sql-rl) (Q8_0 GGUF via llama.cpp).
- Speech-to-text: faster-whisper `small` (int8).
- Everything runs on the free CPU tier, so expect a few seconds per question.
- Databases: Spider (Yu et al., 2018), CC BY-SA 4.0.

This folder is built from the project repository by `scripts/build_space.sh`; the app code is
`demo/app.py` and `demo/runtime.py`, and the pipeline core is `tinysql/`.
