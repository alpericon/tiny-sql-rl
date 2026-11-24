# Week 13 plan: where does it stand? Our model vs Qwen2.5-Coder-1.5B on Spider test and BIRD dev

**Status: APPROVED 2026-11-23, as written (Qwen2.5-Coder-7B not added). Done 2026-11-24: ours ahead in K1, K2, K4; K3 (bare, BIRD) no detectable difference. Much of the lead is answer format.**

## Why

All the results so far compare the project with itself: zero-shot Qwen3-1.7B, the GRPO model, and the
inference steps. The only outside comparison is SQLCoder-7B, on 100 spoken questions. "Is 66% on Spider
test or 30% on BIRD good for this size?" has no measured answer yet. The closest reference point is a
model of the same size that was **built for code and SQL**: Qwen2.5-Coder-1.5B-Instruct.

## Facts checked before writing

- **Qwen2.5-Coder-1.5B-Instruct** is already in the WSL Hugging Face cache (used in week 1), so
  **nothing is downloaded**.
- **Week 1, Spider dev, zero-shot, our metric:**
  - Qwen2.5-Coder-1.5B 54.2%, Qwen3-1.7B 53.7%;
  - but Coder had 8.8% format failures (Qwen3: 4.9%), and 9.4% of its answers hit the 256-token
    limit (Qwen3: 1.5%) (`results/baseline_Qwen__*.json`).
  - So the two base models start at about the same level. The comparison was never run on test or
    BIRD, or against the GRPO model.
- **The harness already supports it:**
  - `HFGenerator` overrides Coder's shipped `repetition_penalty=1.1`, so decoding is identical (greedy,
    no penalty).
  - `eval_baseline.py` and `eval_bird.py` take `--model`.
  - Qwen2.5's chat template has no `enable_thinking`. `HFGenerator.render` falls back to the plain
    template. Our prompt has its own system message, so Qwen2.5's default system message is not
    inserted.

## Question

**At the same size, is our GRPO model (and our recommended configuration) better than an off-the-shelf
SQL-capable model, on held-out Spider test and on BIRD dev?**

## Method (fixed in advance; each run once)

**New runs: Qwen2.5-Coder-1.5B-Instruct, nf4, greedy, 256 new tokens.**

| | Configuration | Spider test | BIRD dev |
|---|---|---|---|
| CA | Coder zero-shot, schema-only | batch 8, guard 3,072 (as week 11) | batch 4, guard 5,500 (as week 12) |
| CE | Coder + repair + grounding + 3 example rows + pruning (our recommended stack) | same | same |

- Our side is **not rerun**. The week 11 and week 12 records are reused: A (Qwen3 zero-shot),
  B (v2-900), E (v2-900 + the stack).
- The prompt, the repair turn, the pruner and the grounding are exactly the same for every model.
  Nothing is adapted to Coder.
- **Metrics:**
  - Spider test: official EX (test-suite-sql-eval), the week 11 decision metric;
  - BIRD dev: BIRD EX (`tinysql.bird.bird_ex`), the week 12 decision metric.

**Pre-registered comparisons** (paired exact McNemar, two-sided; Bonferroni over four, p < 0.0125):

| # | Data | Comparison | Question |
|---|---|---|---|
| K1 | Spider test | B vs CA | Our GRPO model vs Coder, both bare |
| K2 | Spider test | E vs CE | Both with the same inference stack |
| K3 | BIRD dev | B vs CA | as K1 |
| K4 | BIRD dev | E vs CE | as K2 |

**Labels:** "ours ahead" (significant, ours higher), "Coder ahead" (significant, Coder higher), or "no
detectable difference". No direction is predicted.

**Reported but not tested:**
- Coder vs Qwen3, both zero-shot (CA vs A): how the two base models compare.
- Whether the stack helps Coder too (CE vs CA). This tests the stack on a model it was not developed on.
- Format failures and 256-token hits per model.
  - Our model was trained on exactly this prompt and answer format; Coder was not.
  - So EX is also reported on the questions where both models produced a ```sql block. This shows how
    much of a gap is format and how much is SQL.
- Results by BIRD difficulty, and with and without `formula_1`, as in week 12.

**One-shot rules:** each run once. No change to the prompt, decoding, pruner or anything else because
of these results. A loss is reported as a loss.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| K0 | Smoke test (not scored) | — | 20 Spider dev questions with Coder through both scripts and the full stack: no crash, the output is parsed, the prompt guard holds. |
| K1 | Registries and the comparison script, test-first | `scripts/official_eval.py` (Coder entries in `TEST_MODELS`; the record prefix no longer assumes Qwen3), `scripts/compare_bird.py` (paired McNemar on two BIRD record files, overall and without `formula_1`; checks the question order), `tests/` | Tests: the McNemar counts on toy records; refusal when the question order differs; the Coder record names resolve. |
| — | **Review gate** (K0–K1) | | Reviewer approves before any scored run. |
| K2 | Four runs | `results/test_Qwen__Qwen2.5-Coder-1.5B-Instruct*`, `results/bird_CA-*`, `results/bird_CE-*` | CA and CE, once each, on Spider test and BIRD dev. Official EX for the two Spider test runs. |
| K3 | Write-up + README + model card + final review | `results/week13_results.md`, README, `release/model_card.md` | K1–K4 labelled; the descriptive numbers above. `check_readme.py` covers the new numbers. Upload and push only after you say so. |

**Cost:** ~1–2 h code. GPU ~5 h; no training, no download.
- Spider test: ~30 min (CA) + ~1 h (CE).
- BIRD dev: ~1.2 h (CA) + ~2 h (CE).
- Coder was ~1.5× slower than Qwen3 on dev because it writes longer answers.

## Risks

- **Possible contamination in Coder's favour.** Qwen2.5-Coder was trained on large public data,
  including text-to-SQL data. Spider (including dev) and BIRD dev may be in it; Spider test was
  released before Coder (2024). We cannot check this. It would favour Coder, and it is stated next to
  the results.
- **The prompt favours our model.** It is the format our model was trained on. This is reported through
  the format-failure counts and the both-have-a-block EX above, not fixed by tuning a prompt for Coder.
- **The stack was developed on our model** (on Spider dev). If it helps Coder less, that is a finding,
  not a bug.
- **Single run, single seed.** No training is involved, but our model is still one training run.

## Option to decide (not in the plan unless you add it)

**Qwen2.5-Coder-7B-Instruct as an upper reference**, descriptive only:
- a ~15 GB download (the WSL disk is on C:, which has ~40 GB free);
- roughly 3× the GPU time (~15 h for the same four runs);
- BIRD's 4.8k-token prompts at batch 4 probably do not fit in 8 GB with a 7B model, so a smaller batch
  would be needed.

My recommendation is to leave it out. The 1.5B comparison answers the "same size" question. Results
for 7B models on these benchmarks are published elsewhere, though not in our exact setting.

## Not in this plan

Larger models (unless the option above is added), API models, prompt tuning for Coder, training Coder
with our GRPO recipe, other benchmarks, and voice with Coder.
