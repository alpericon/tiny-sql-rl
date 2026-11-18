# tiny-sql-rl

**Can a 1.7B model, trained with reinforcement learning on one 8 GB laptop GPU, replace a 7B text-to-SQL
model in a voice pipeline?**

In an earlier voice-to-SQL project, SQLCoder-7B looked like the bottleneck: about 14 s of a ~14 s
query. This project trains Qwen3-1.7B with GRPO on an **execution reward** (does the query return the
right rows?) and measures it against that baseline on accuracy, robustness to ASR noise and latency.

- **Model:** [Alpericon/tiny-sql-rl](https://huggingface.co/Alpericon/tiny-sql-rl) (LoRA adapter, merged bf16, Q8_0 GGUF)
- **Demo:** text or voice, runs locally on CPU: [docs/demo.md](docs/demo.md). A hosted Space was
  not possible: free Gradio Spaces now need a paid plan.
- **Full results with every caveat:** [results/week3_results.md](results/week3_results.md), [results/week4_results.md](results/week4_results.md)

## Results

All accuracy numbers are execution accuracy (EX): greedy decoding, 4-bit, Spider dev (1,034 questions)
unless stated. "Official" is Spider's official evaluation script; "ours" is this project's
column-order-tolerant comparator. Comparisons are paired (exact McNemar on the same questions).
Every training run uses **one seed**, so the tests cover question sampling, not training variance.

### RQ1: does GRPO help?

| Model | EX (ours) | EX (official) | vs zero-shot (wins / losses) |
|---|---|---|---|
| Qwen3-1.7B zero-shot | 53.7% | 54.1% | |
| GRPO, all training questions | 57.1% | 58.1% | 58 / 16, p<0.0001 |
| **GRPO, pass-rate-filtered questions (released)** | **58.5%** | **59.0%** | 84 / 33, p<0.0001 |
| GRPO, filtered, binary reward (ablation) | 58.2% | 58.8% | |
| **released model + one retry on SQL errors** | **60.3%** | **60.8%** | 19 more rescued, 0 lost |

- **Yes: +4.9 points** (official), significant.
- **The difficulty filter** (train only on questions the base model solves 1–7 of 8 times) cut the
  share of groups with no learning signal from ~65% to 25–41% and made training reward rise.
  But on dev it is **not significantly better** than training on everything (39 / 30, p=0.34).
- **The 0.1 partial credit** for executable-but-wrong SQL made **no difference** (19 / 17).
- **One retry with the error message** (the same repair step SQLCoder's pipeline had) rescues 19 of 133
  failing answers. In 53 of the retries the model just repeats its SQL, since it was never trained on
  multi-turn prompts.
- What still fails is mostly SQL logic. Without repair, 296 of 429 failures return the wrong rows
  and only 10 are format failures.

![Training curves](docs/figures/training_curves.png)

### RQ2 + RQ4: voice input vs SQLCoder-7B

The test set is 100 SpokenSpider questions (3 databases, never trained on), each read by TTS and
transcribed by ASR at four noise levels. SQLCoder's SQL is the voice-to-SQL project's stored output,
re-scored with the same comparator.

| Input | Qwen3-1.7B base | **GRPO model** | GRPO + repair | SQLCoder-7B (with its repair) |
|---|---|---|---|---|
| typed question | 46% | **57%** | 58% | — |
| clean ASR | 43% | 52% | 55% | 52% |
| 20 dB SNR | 43% | 55% | 57% | 50% |
| 10 dB SNR | 43% | 52% | 54% | 48% |
| 0 dB SNR | 27% | 38% | 39% | 44% |

- **The 1.7B model ties the 7B model from clean to 10 dB.** With a repair step on both sides it is level
  or ahead (55 vs 52 clean, 57 vs 50 at 20 dB). No per-condition difference is significant (n=100).
- **It degrades more at 0 dB**: −14 points vs −8 for SQLCoder. That gap is directional and was not tested.
- **Even clean ASR costs ~5 points vs typed text.** Transcripts lose literal case and formatting
  (`"Aliens"` → `aliens`, `1000` → `1,000`), so string filters stop matching.

![Voice EX by SNR](docs/figures/voice_by_snr.png)

### RQ3: latency

RTX 3070 Ti Laptop, nf4, batch 1, 100 clean transcripts:

![Latency](docs/figures/latency.png)

- **Merged, the 1.7B model is ~30% faster than SQLCoder-7B** at the median (1.42 vs 2.08 s), mostly
  because its answers are shorter. Per token the two 4-bit models are almost equally fast (45 vs 50 ms).
  - The unmerged LoRA adapter doubles the cost per token, so it must be merged for deployment.
- **The bigger finding: the "14 s bottleneck" was not the 7B model.** Re-run with the identical
  prompt and settings (it reproduces 93/100 of the stored SQL), SQLCoder takes 1.8–2.1 s. The
  earlier pipeline's runtime setup was the bottleneck.
  - With either model, the whole voice pipeline is about 2–3 s.
  - So the honest answer to RQ3 is: a small model saves ~0.6 s and ~2.6 GB of VRAM, not ~12 s.
- **On CPU** (the demo's Q8_0 GGUF), a question takes ~1.2–4.8 s plus ~1.5 s of speech-to-text.

## How it works

1. **Data.** Spider train, minus the 100 SpokenSpider questions (leakage filter). Questions whose gold
   result is empty are dropped: a reward-hacking check found the policy getting full reward for any query
   that returns nothing ([results/hack_check.md](results/hack_check.md)).
2. **Difficulty filter.** The base model samples 8 answers per training question. Only questions it
   solves 1–7 times are kept: 1,688 of 6,638.
3. **GRPO.** TRL + Unsloth + vLLM, 4-bit QLoRA r=16. Settings:
   - 8 samples per prompt, 1,000 steps. The released run peaked at 5.75 GiB reserved VRAM and took
     5.3 h of training steps overnight on the laptop (`results/train_log_v2.csv`).
   - Reward: 1.0 if the rows match gold, 0.1 if the SQL runs but is wrong, 0 otherwise.
   - Sandboxed read-only SQLite executor with a timeout.
4. **Selection and evaluation.** The checkpoint is chosen on a 200-question dev subset with a rule fixed
   in advance (best EX, ties to the later step). It is then evaluated on full dev, on the 834 questions
   outside the subset, with the official script, and on voice.
5. **Release.** The adapter is merged into bf16, then exported to GGUF.
   - Q8_0 is shipped. Q4_K_M breaks the model's output format (−12.5 points), so it is not.

## Reproduce

Training and GPU evaluation run in WSL2 (Ubuntu) with CUDA. The demo runs on any CPU.

```bash
bash scripts/setup_wsl.sh                      # training venv (~/venvs/tinysql)
bash scripts/download_spider.sh                # Spider into data/spider/
python scripts/build_exclude_list.py           # SpokenSpider leakage filter
python scripts/check_data.py
~/venvs/tinysql/bin/python scripts/pass_rate.py                                   # difficulty filter (~80 min)
~/venvs/tinysql/bin/python scripts/train_grpo.py --run-name v2 --pass-rate data/train_pass_rate.jsonl
~/venvs/tinysql/bin/python scripts/eval_baseline.py --model Qwen/Qwen3-1.7B --adapter checkpoints/v2/checkpoint-900
bash scripts/setup_official_eval.sh && ~/venvs/officialeval/bin/python scripts/official_eval.py
python scripts/make_figures.py                 # figures from results/ only
```

The planning history is in `SPEC.md` and `PLAN-week1..4.md`. Each week ends with a review gate,
and every number in `results/` comes from a committed script and its output file.

## Limitations

- **One seed per run.** Differences under ~2 points between trained models are not distinguishable
  from training noise. Greedy decoding flips ~15% of answers under small numeric changes, even when
  the averages agree.
- **Spider only, SQLite only, English only.** The voice test covers just 100 questions on 3 databases.
- **Two EX definitions.** The official default removes DISTINCT, which makes it slightly more lenient
  than ours. Both are reported. Our executor cannot read one wta_1 question with broken UTF-8.
- **The demo's speech latency** was measured on short synthetic clips, on a fast laptop CPU.

## License and attribution

- Code: Apache-2.0.
- Base model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0).
- Data: [Spider](https://yale-lily.github.io/spider) (Yu et al., 2018; CC BY-SA 4.0); the demo bundles 6 Spider databases.
- Official metric: [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval).
