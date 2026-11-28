# brag plan: tiny-sql-rl (polished, landscape 1920×1080, 30 fps, 20 s, no voice)

## Answers
- **What:** Qwen3-1.7B trained with GRPO on an execution reward to turn English questions into SQL, on one 8 GB laptop GPU.
- **For whom:** people who want small, local text-to-SQL (and the ML crowd curious whether RL works at 1.7B).
- **What sets it apart:** tiny model, laptop training, honest held-out evaluation, gain replicated over 3 seeds, runs on CPU.
- **Most impressive claim (all from results/):** Spider test official EX 56.3% → 61.7% from GRPO alone; 66.3% with the inference stack; the gain replicated with 3 training seeds (+5.5 mean).
- **Visual hook:** "1.7B parameters. One 8 GB laptop GPU." on the demo's own dark background, the orange accent drawing in.
- **Real UI:** the local demo (Gradio, dark), with a real run captured today: world_1, "Which continent has the most countries?" → 5-line SQL → Africa | 58, "text-to-SQL + query: 11.2 s" on CPU. No personal data involved (countries).
- **Tone:** polished: 4 scenes, long holds, soft fades, restrained motion.
- **Share caption:** see share-copy.txt.

## Visual identity (from the demo's computed styles)
Background #0f0f11, panels #27272a, borders #3f3f46, text #f4f4f5, accent (Ask button) #c2410c.
UI text Arial (what the demo renders); code Consolas (the demo's monospace fallback); titles Segoe UI.
The demo UI is redrawn at 1080p from the real page's layout and the captured outputs (the pane only gives 800×500 screenshots).

## Music
96 BPM (beat 0.625 s, bar 2.5 s), 8 bars = 20 s, A minor: Am – F – C – G. Soft pad, sub bass on bar starts,
quiet kick on beats 1 and 3 from bar 3, soft offbeat ticks. SFX in key: soft click on the Ask press, a
low whoosh dip under each cut, a soft bell on each number. Fade in 0.3 s, fade out last 1.2 s. All cuts on bar lines.

## Storyboard
| # | Time | Bars | Content | Motion |
|---|---|---|---|---|
| 1 Hook | 0.0–5.0 | 1–2 | "1.7B parameters." / "One 8 GB laptop GPU." then small line "Reinforcement learning that writes SQL." | lines rise in one by one, orange rule draws under them; hold |
| 2 Product | 5.0–12.5 | 3–5 | Caption "Ask in plain English." Demo panel: question types in, Ask presses, SQL lines appear, result Africa · 58, "text-to-SQL + query: 11.2 s (CPU)" | typing, button press, lines in one by one, slow push-in |
| 3 Proof | 12.5–17.5 | 6–7 | "Spider test, official EX" 56.3% → 61.7% (count-up), then "+5.5 points, replicated over 3 training seeds" and "66.3% with the full inference stack" | numbers count up, lines fade in staggered |
| 4 Outro | 17.5–20.0 | 8 | "tiny-sql-rl" + huggingface.co/Alpericon/tiny-sql-rl + github.com/alpericon/tiny-sql-rl | fade in, hold, music tail |

Transitions: dip through the background (old out, then new in), 0.35 s each side of the bar line.
