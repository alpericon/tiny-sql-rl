# Week 6 results: grounding SQL string literals in the database (PLAN-week6)

**Setup.** The model is v2-900, the released model, with the Y9 repair step and the same command as
before. The only change is `--ground`: before execution, a single-quoted literal that is not in the
database but matches exactly one database value after case-folding and whitespace collapsing is
replaced by that value (`src/tinysql/ground.py`). The rule was fixed in the plan before any
measurement. The model's text is identical in the runs with and without grounding (500/500 voice,
1,034/1,034 dev completions), so every difference comes from grounding.

## Primary: voice, clean ASR (100 SpokenSpider queries)

| Input | repair only | **repair + grounding** | grounded vs not (w / l, p) | grounding fired | SQLCoder-7B, re-scored with grounding |
|---|---|---|---|---|---|
| typed question | 58% | 58% | 0 / 0 | 0 | (55%) |
| **clean** | 55% | **59%** | **4 / 0, 0.125** | 9 | 55% |
| 20 dB | 57% | 60% | 3 / 0, 0.25 | 7 | 53% |
| 10 dB | 54% | 56% | 2 / 0, 0.50 | 5 | 51% |
| 0 dB | 39% | 40% | 1 / 0, 1.0 | 3 | 45% |

Sources: `results/voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground.json` and
`results/voice_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair.json`.

- **Grounding fixed exactly the 4 clean-ASR failures it was designed for and lost nothing.**
  - The fixes are `'aliens'` → `'Aliens'` (×2) and `'village'` → `'Village'` (×2).
  - As stated in the plan, 4 / 0 cannot reach significance at n=100 (p=0.125).
  - Across all ASR conditions it is 10 wins and 0 losses.
- **On clean speech the model now matches typed text** (59% vs 58%). That case/quote gap is gone.
- **All 24 replacements in the voice runs are case changes**, read by hand:
  - `fail/pass/yes` → `Fail/Pass/Yes`, `ALABAMA` → `Alabama`, `village` → `Village`, `aliens` → `Aliens`.
  - 10 of them changed the outcome. The rest were in queries that were wrong for another reason as well.
- **SQLCoder gains about as much from the same post-processing:** 52 → 55 clean, 3 / 0 per condition.
  With grounding on both sides, our 1.7B model vs SQLCoder-7B is:

  | Input | Ours vs SQLCoder (w / l, p) | EX |
  |---|---|---|
  | clean | 18 / 14, 0.60 | 59 vs 55 |
  | 20 dB | 19 / 12, 0.28 | 60 vs 53 |
  | 10 dB | 18 / 13, 0.47 | 56 vs 51 |
  | 0 dB | 7 / 12, 0.36 | 40 vs 45 |

  It is level or ahead through 10 dB and behind at 0 dB. None of these differences is significant.

## Guard rail: Spider dev (typed questions, 1,034)

| | repair only | repair + grounding | w / l, p |
|---|---|---|---|
| Ours | 60.3% | 60.4% | 5 / 4, 1.0 |
| Official | 60.8% | 60.9% | 5 / 4, 1.0 |

Sources: `results/baseline_Qwen__Qwen3-1.7B_v2-checkpoint-900_repair_ground.json` and
`results/official/v2-900-repair-ground.json`.

- **The guard rail holds**, with no significant loss. Grounding fired on 34 of 1,034 queries:
  - 17 correct before and after;
  - 8 wrong before and after;
  - 5 fixed (`'Cat'` → `'cat'` ×2, `'Jetblue Airways'` → `'JetBlue Airways'` ×2, `'Math'` → `'math'`);
  - 4 lost.
- **All 4 losses come from a Spider data quirk, not a grounding error.**
  - In flight_2, every one of the 100 city names is stored with a trailing space (`'Aberdeen '`).
  - The gold queries (`city = "Aberdeen"`, `city = "Anthony"`) therefore return **no rows**.
  - The ungrounded prediction also returned no rows and was scored correct.
  - Grounding finds the airports the question asks for (e.g. `('ANY', 'Anthony ')`), which no
    longer equals the empty gold.
  - Under the pre-registered metric these count as losses, and they are reported as such.
  - The plan's whitespace normalisation is what triggers them; it was not changed after the fact.

## Decision (per the plan)

- The guard rail held, so **the demo grounds by default**: `tinysql.demo.answer_question(ground=True)`.
  The UI lists every replaced literal.
- The released model weights are unchanged. Grounding is inference-time post-processing.

## Caveats

- n=100 voice queries on 3 databases. The effect is 4 fixed queries on clean ASR: a mechanism
  confirmed case by case, not a statistically established rate.
- Grounding cannot fix ASR word errors ("ID" heard as "it") or number formats ("1,000"). These remain.
