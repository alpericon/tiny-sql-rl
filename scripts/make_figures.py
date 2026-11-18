"""README figures, built only from files in results/ (PLAN-week4 Y7).

    python scripts/make_figures.py      # writes docs/figures/*.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs" / "figures"
RUNS = {"main": "main (unfiltered)", "v2": "v2 (pass-rate filter)", "v2-binary": "v2-binary (filter, binary reward)"}
COLORS = {"main": "#8c8c8c", "v2": "#1f77b4", "v2-binary": "#ff7f0e", "sqlcoder": "#2ca02c", "base": "#bbbbbb"}


def rolling(xs, k=50):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= k:
            s -= xs[i - k]
        out.append(s / min(i + 1, k))
    return out


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for run, label in RUNS.items():
        rows = [r for r in csv.DictReader(open(R / f"train_log_{run}.csv", encoding="utf-8")) if r["reward"]]
        steps = [int(r["step"]) for r in rows]
        axes[0].plot(steps, rolling([float(r["reward"]) for r in rows]), label=label, color=COLORS[run])
        axes[1].plot(steps, rolling([float(r["frac_reward_zero_std"]) for r in rows]), label=label, color=COLORS[run])
    axes[0].set(title="Training reward (50-step mean)", xlabel="step", ylabel="reward")
    axes[1].set(title="Groups with no learning signal (50-step mean)", xlabel="step", ylabel="fraction of groups", ylim=(0, 1))
    axes[0].text(0.01, 0.02, "partial-reward runs include 0.1 credits;\nbinary run's reward = fraction correct",
                 transform=axes[0].transAxes, fontsize=7, color="#555")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)


def dev200_checkpoints():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    base = json.load(open(R / "dev200_Qwen__Qwen3-1.7B.json"))["execution_accuracy"]
    ax.axhline(100 * base, color="#999", ls="--", label="base (zero-shot)")
    for run, label in RUNS.items():
        xs, ys = [], []
        for n in range(100, 1001, 100):
            p = R / f"dev200_Qwen__Qwen3-1.7B_{run}-checkpoint-{n}.json"
            if p.exists():
                xs.append(n)
                ys.append(100 * json.load(open(p))["execution_accuracy"])
        ax.plot(xs, ys, marker="o", label=label, color=COLORS[run])
    ax.set(title="dev-200 EX per checkpoint (selection set)", xlabel="step", ylabel="EX (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dev200_checkpoints.png", dpi=150)


def voice_by_snr():
    conds = ["text", "clean", "20db", "10db", "0db"]
    v2 = json.load(open(R / "voice_Qwen__Qwen3-1.7B_v2-checkpoint-900.json"))["conditions"]
    base = json.load(open(R / "voice_Qwen__Qwen3-1.7B.json"))["conditions"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(conds, [100 * v2[c]["ours_ex"] for c in conds], marker="o", label="v2-900 (1.7B)", color=COLORS["v2"])
    ax.plot(conds, [100 * base[c]["ours_ex"] for c in conds], marker="o", label="Qwen3-1.7B base", color=COLORS["base"])
    ax.plot(conds[1:], [100 * v2[c]["sqlcoder_ex"] for c in conds[1:]], marker="s", label="SQLCoder-7B (voice-to-sql)", color=COLORS["sqlcoder"])
    ax.set(title="Voice EX by condition (100 SpokenSpider queries)", xlabel="input", ylabel="EX (%)", ylim=(20, 65))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "voice_by_snr.png", dpi=150)


def latency():
    # Same-session triple (results/week4_results.md, Y2) plus the week-3 unmerged-adapter run.
    files = [("v2-900 + LoRA (unmerged, wk 3)", "latency_ours_Qwen__Qwen3-1.7B_v2-checkpoint-900.json", "#9ecae1"),
             ("v2-900 merged (wk 4)", "latency_ours_checkpoints__v2-900-merged_y2.json", COLORS["v2"]),
             ("Qwen3-1.7B base (wk 4)", "latency_ours_Qwen__Qwen3-1.7B_y2.json", COLORS["base"]),
             ("SQLCoder-7B (wk 4)", "latency_sqlcoder_y2.json", COLORS["sqlcoder"])]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (name, f, c) in enumerate(files):
        s = json.load(open(R / f))["summary"]["text_to_sql"]
        ax.barh(i, s["median_s"], color=c, alpha=0.9)
        ax.errorbar(s["median_s"], i, xerr=[[0], [s["p95_s"] - s["median_s"]]], color="k", capsize=3)
        ax.text(s["median_s"] + 0.05, i + 0.28, f"{s['median_s']:.2f} s", fontsize=8)
    ax.set_yticks(range(len(files)), [n for n, _, _ in files])
    ax.invert_yaxis()
    ax.set(title="Text-to-SQL latency, RTX 3070 Ti Laptop, nf4, batch 1 (median; whisker = p95)", xlabel="seconds")
    ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    training_curves()
    dev200_checkpoints()
    voice_by_snr()
    latency()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
