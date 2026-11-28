# Demo

Ask a Spider database a question by text or voice. The model writes SQL, the SQL runs read-only, and
the app shows the result table with a one-sentence answer. Everything runs on CPU.

**Video (20 s):** [video/tiny-sql-rl.mp4](video/tiny-sql-rl.mp4). It shows a real run of this demo on CPU
(world_1, "Which continent has the most countries?": the generated SQL, the result and the 11.2 s timing),
redrawn at 1080p, followed by the held-out Spider test results. It has music and no voice-over; how it
was made is in `docs/video/` (`plan.md`, `build.py`).

## Run it locally

Requirements: [uv](https://docs.astral.sh/uv/) and a C/C++ compiler (llama-cpp-python compiles
on install). Linux/WSL2 or macOS.

```bash
bash scripts/setup_demo.sh        # once: creates ~/venvs/demo (~5-10 min)
bash scripts/setup_demo.sh run    # http://127.0.0.1:7860
```

- The first question downloads the released model from
  [Alpericon/tiny-sql-rl](https://huggingface.co/Alpericon/tiny-sql-rl): the Q8_0 GGUF (~1.8 GB),
  plus Whisper-small on the first voice question.
- Set `TINYSQL_THREADS` to limit CPU threads. Set `TINYSQL_VOICE=0` for text only.
- Measured speed on CPU (i9-12900H, `results/week4_results.md` Y5): ~1–5 s per question, plus ~1.5 s
  to transcribe a short spoken question.

## Why there is no hosted Space

The plan was a Hugging Face Space on the free CPU tier (SPEC §7). When the Space was created
(2026-11-18), Hugging Face refused it: Gradio and Docker Spaces on free `cpu-basic` now require a PRO
subscription, and only static Spaces are free. SPEC §7's fallback applies: a local demo plus a
recorded video.

The Space tooling is kept and works if hosting becomes available:
- `space/README.md` is the Space card;
- `scripts/build_space.sh` assembles the app;
- `scripts/upload_space.py` deploys it.

## Recording a longer voice demo

The video above has no live voice input. To record one with the microphone:
1. Start the demo locally with `bash scripts/setup_demo.sh run`.
2. Ask two or three of the example questions by voice (microphone), and one by text.
3. Ask one question the model gets wrong, to show how errors are displayed.
4. Keep it under a minute.
5. Upload the video, for example to the GitHub repo's README or as a release asset, and link it at
   the top of this page and in the README.
