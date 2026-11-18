# I measured the bottleneck, trained a model to remove it, and found the bottleneck was somewhere else

In a voice-to-SQL project, SQLCoder-7B took about 14 of ~14 seconds per spoken question. The plan for
this project followed from that: train a model four times smaller, with reinforcement learning, on
one 8 GB laptop GPU, and swap it in.

**The training worked.**
- Setup: Qwen3-1.7B with GRPO and a reward that runs the generated SQL and checks the rows, as a 4-bit QLoRA.
- Each run took about 5.5 hours overnight.
- Result: Spider dev execution accuracy went from 54.1% to 59.0% with the official script
  (84 wins / 33 losses on the same questions).

Three things mattered more than tuning:

- **Most training groups taught nothing.** GRPO learns from differences within a group of 8 answers.
  In ~65% of groups all 8 got the same reward, because the question was always or never solved.
  Filtering to the questions the base model solves 1–7 times cut that to 25–41%, and training reward
  finally rose.
  - On dev, though, the filtered run was not significantly better than the unfiltered one (39 / 30,
    p=0.34). Faster learning on 1,688 questions did not become better generalization.
- **The reward was hackable.** Reading the highest-reward samples by hand showed full reward for
  queries that were wrong but returned nothing, whenever the gold answer was also empty. Those
  questions were dropped.
  - Partial credit for runnable-but-wrong SQL turned out not to matter (19 / 17 vs a binary reward).
- **Release details changed the numbers more than training did.**
  - The unmerged adapter doubled latency per token.
  - A 4-bit GGUF (Q4_K_M) cost 12.5 points, because it tipped the model's first token from ```` ``` ````
    to `SELECT`, breaking the output format.
  - llama-cpp-python's default `repeat_penalty=1.1` broke it again.
  - Q8_0, merged, with neutral sampling matches the evaluated model.

**On voice, the 1.7B model ties the 7B model** from clean audio down to 10 dB SNR. It falls behind
at 0 dB. Clean ASR alone costs ~5 points, because transcripts lose capitalisation and quotes in
literal values, so `WHERE theme = 'Aliens'` stops matching.

**Then the latency result.** I re-ran SQLCoder-7B with the earlier project's exact prompt and
settings; it reproduced 93 of its 100 stored queries. It took **1.8–2.1 s**, not 14. The bottleneck
had been how the old pipeline ran the model, not the model itself. With that fixed:
- the 1.7B model is ~30% faster (1.42 vs 2.08 s, mostly from shorter answers);
- it uses 2.6 GB less VRAM;
- the full voice pipeline takes 2–3 s with either model.

The honest summary is that RL made a small model as good as a 7B specialist on this voice task. The
large speed-up I set out to buy was mostly available for free, by measuring the old system properly.

*Single seed per run; Spider/SQLite/English only; 100 voice questions on 3 databases. Details and every
number's source: [results/week3_results.md](../results/week3_results.md), [results/week4_results.md](../results/week4_results.md).*
