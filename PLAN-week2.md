# Week 2 implementation plan: GRPO training (from SPEC.md §5.3, §6)

**Status: APPROVED 2026-11-15.**

Goal of the week: a QLoRA GRPO run on **Qwen3-1.7B** (SPEC §12) that **beats its own zero-shot EX on a fixed 200-question dev subset**, with VRAM headroom and reward-hacking checks logged.

Baseline to beat: 53.7% EX on full dev; **55.0% on dev-200** (`results/dev200_Qwen__Qwen3-1.7B.json`). The dev-200 baseline is re-run with the same command as the checkpoints: with 4-bit weights, batch composition alone changed 28/200 greedy completions and 3 matches, so differences under ~1.5 pp are noise.

Same rules as week 1:
- CPU tasks are test-first.
- GPU tasks run in WSL with `~/venvs/tinysql`.
- Every training script asserts `torch.cuda.is_available()` before doing anything else.

## Setup (proposed defaults; the W4 pre-flight measures them and may change them)

| Setting | Value | Why |
|---|---|---|
| Stack | Unsloth `FastLanguageModel` (4-bit, LoRA) + TRL `GRPOTrainer`; vLLM generation via Unsloth `fast_inference` | Fastest path on 8 GB. Fallback: HF generation without vLLM (SPEC §10). |
| LoRA | r=16, alpha=16, all attention + MLP projections | Common GRPO-on-small-model setting |
| Generations per prompt G | 8 (drop to 4 if VRAM or speed fails) | SPEC: 4–8 |
| Max completion length | 256 tokens | Baseline answers fit; longer answers were the format failures |
| KL beta | 0.04 to the reference model | SPEC: KL to reference |
| Learning rate | 5e-6, cosine, warmup 3% | Conservative for RL on LoRA |
| Steps | ~1,000 (more if time allows) | SPEC: 1–2k |
| Prompt | Same as baseline (`build_messages`, `enable_thinking=False`) | Train and eval prompts must match |
| Reward | `sql_reward` with 0.1 partial credit (binary reward is the week-3 ablation) | SPEC §5.2, §11 |
| Seed | 3407, fixed | Single seed, stated as a limitation |

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| W1 | GRPO dataset builder | `src/tinysql/grpo_data.py`, `tests/test_grpo_data.py` | Turns `data/train_filtered.json` (6,890) into records with `prompt` (chat messages), `db_path` and `gold`. Test: no SpokenSpider example present, every db_path exists, prompt matches `build_messages` exactly. |
| W2 | TRL reward adapter | `src/tinysql/grpo_reward.py`, `tests/test_grpo_reward.py` | `reward_fn(prompts, completions, db_path, gold, **kw) -> list[float]` handles both conversational and plain-string completions. Never raises on model output. Raises loudly on a failing gold. Tests cover all SPEC §5.2 outcomes and batch alignment. |
| — | **Review gate** (CPU core of week 2) | | Reviewer approves, or findings are fixed |
| W3 | Fixed dev-200 subset + adapter eval | `scripts/make_dev_subset.py`, `data/dev200_ids.json`, `scripts/eval_baseline.py --adapter` | Subset is seeded and committed. Baseline re-run with the same command as checkpoints (see above). Eval loads a LoRA adapter on the same 4-bit base and uses the same prompt and greedy decoding. |
| W4 | Training script + 50-step pre-flight | `scripts/train_grpo.py`, `configs/grpo_qwen3_1p7b.yaml` | 50 steps run without OOM. Logged: peak VRAM, seconds/step, GPU temperature, mean reward, completion length. Full-run time is extrapolated from these. |
| W5 | Main GRPO run | `checkpoints/` (git-ignored), `results/train_log_*.csv` | Reward curve rises. Checkpoint every 100 steps, and the run resumes from a checkpoint after an interruption. |
| W6 | Checkpoint selection on dev-200 | `results/dev200_*.json` | EX per checkpoint. **The best checkpoint beats the baseline on dev-200.** Only the best one goes to full dev in week 3. |
| W7 | Reward-hacking inspection | `results/hack_check.md` | 20 highest-reward training samples read by hand. Any exploit is written down and fixed before week 3. |
| — | **Review gate** after W7 | | Reviewer approves |

## Risks this week

- **Unsloth + vLLM do not run together in 8 GB.** Fall back to HF generation (slower: fewer steps, or G=4).
- **The reward does not rise.** Check:
  - truncation of the 256-token completions;
  - the share of groups where all 8 rewards are equal (zero advantage, so no learning signal);
  - the learning rate.
- **Thermal throttling on the laptop.** Log temperature and clocks. Long runs go in the background and can resume from checkpoints.
