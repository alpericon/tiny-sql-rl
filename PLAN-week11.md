# Week 11 plan: one-shot confirmation on the Spider test set

**Status: APPROVED 2026-11-22. Done 2026-11-22: GRPO and the whole stack confirmed on test (66.3% vs 56.3%); example rows and pruning directionally consistent but not significant.**

## Why

Every improvement after GRPO was kept or rejected by a test on the same 1,034 Spider dev questions:
- repair (week 4);
- grounding (week 6);
- example rows (week 7);
- schema pruning (week 10).

Each decision followed a rule written in advance, and the pruner was tuned on train. Even so, four
sequential decisions on one set can fit that set. **Nothing has been measured on data that played no
part in any decision.**

**The Spider 1.0 test set is now public.** The official page (news of 02/05/2024) says it has been
released, and Spider 1.0 submissions are closed.
- The archive we already downloaded and checksum-verified (`~/.cache/tinysql/spider_data.zip`, the same
  official Google Drive file) contains `test.json`, `test_tables.json` and `test_database/`.
- `scripts/download_spider.sh` copied only train and dev, so no new download is needed.
- Checked: 2,147 questions on 40 databases, with gold SQL. **No database overlaps with train or dev**,
  and no question repeats.

## Question

**Do the gains chosen on dev hold on held-out databases?**

## Method (fixed in advance; run once)

**Configurations** (all as released; nothing is tuned on test):

| | Configuration | What it isolates |
|---|---|---|
| A | Qwen3-1.7B zero-shot, schema-only | reference |
| B | v2-900, schema-only | GRPO (weeks 2–3) |
| C | v2-900 + repair + grounding | the step before example rows |
| D | C + 3 example rows | example rows (week 7) |
| E | D + schema pruning (frozen pruner) | pruning (week 10) = **the current demo** |

All runs use nf4, greedy decoding and 256 new tokens, as on dev.

**Pre-registered comparisons** (official EX, the decision metric throughout; paired exact McNemar):

| # | Comparison | Decision it re-tests |
|---|---|---|
| T1 | B vs A | GRPO |
| T2 | D vs C | example rows |
| T3 | E vs D | schema pruning |
| T4 | E vs A | the whole stack (headline) |

- **A decision counts as "confirmed on test"** if its comparison has p < 0.0125 (Bonferroni over
  the four tests) **and the effect has the same sign as on dev**.
- It counts as "directionally consistent" if the sign matches but p is above that, and "not
  reproduced" if the sign flips.
- Our own metric is reported next to official EX, as on dev.

**Rules for a one-shot test:**
- Each configuration runs once. A rerun is allowed only after a crash, with the cause written down.
- No code, prompt, pruner setting or threshold changes because of test results. If a bug shows up,
  it is fixed, reported, and the affected run is redone in full, with the reason stated.
- The results are reported whatever they are. **Nothing in the demo or the release changes because
  of week 11.** A negative result is written up as such.

**Prompt length:** before generation, the test prompts are measured for configurations A, D and E
(the longest). Prompts are never truncated. If any exceeds the current 3,072-token guard, the guard is
raised for all configurations, as in week 7. The pruner's fallback stays as frozen. The longest prompt
is reported.

## Tasks

| # | Task | Files | Done when |
|---|---|---|---|
| H1 | Test data in place | `scripts/download_spider.sh` also copies `test.json`, `test_tables.json` and `test_database/` from the verified archive; `data/check` style counts | 2,147 questions and 40 databases present; overlap check (none with train or dev) saved to `results/test_split_check.json`. |
| H2 | Split support, test-first | `--split test` in `eval_baseline.py` (and `prune_recall.py` for reporting only), `official_eval.py` with a test gold and a test database directory; tests for the split path | Tests pass. With `--split dev` the outputs are unchanged (checked on dev-200 for one configuration). |
| H3 | Prompt lengths + pruner recall on test | `results/prompt_lengths_test.json`, `results/prune_recall_test.json` | Reported only; there is no stop rule, since nothing is tuned. |
| — | **Review gate** (H1–H3) | | Reviewer approves before any test generation. |
| H4 | The five runs | `results/test_*` records, official EX for each | All five complete. |
| H5 | Write-up | `results/week11_results.md`, README (+ `check_readme.py`) | T1–T4 with the labels above, next to the dev numbers. |
| — | **Final review gate** | | Reviewer approves; README numbers verified. |

**Cost:** ~1 h code. GPU: ~4–5 h for the five runs on 2,147 questions (pruned runs were slower on dev).
CPU: ~1 h for official EX.

## Risks

- **Test databases differ** (40 new schemas). Larger or messier schemas could change pruner recall and
  prompt lengths; both are measured and reported before generation.
- **The official script's slow queries.** On dev a few predictions ran for hours, and they are capped
  at 120 s (scored wrong), as before. The capped rows are listed.
- **Small effects may not reach significance** even on 2,147 questions. Week 7's +2.5 points on dev
  would need roughly a 1.5× larger margin of wins over losses than on dev. "Directionally consistent" is
  then the honest label.
- **Test leakage is impossible by construction** for training (no shared databases, and the training
  code never reads `test.json`). The pruner was tuned on train only.

## Not in this plan

Voice on test (SpokenSpider has no test-set audio), SQLCoder on test (no stored outputs), BIRD, the
HF model card and the demo video.
