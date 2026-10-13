"""Make the figures used in report/report.pdf from the saved training metrics.

    python scripts/report_figures.py      # writes report/figures/{sft,dpo,lora}.png

Only needs matplotlib (no torch), so it can be run anywhere the repository is checked out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLORS = {"hh": "#1f77b4", "uf": "#d62728", "train": "#7f7f7f"}


def read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ema(values, alpha=0.9):
    out, avg = [], None
    for v in values:
        avg = v if avg is None else alpha * avg + (1 - alpha) * v
        out.append(avg)
    return out


def running_mean(values, k=10):
    """Mean of the last k values (only where a full window exists)."""
    return [sum(values[i - k + 1 : i + 1]) / k for i in range(k - 1, len(values))]


def plot_sft(results: Path, out: Path) -> None:
    rows = read_jsonl(results / "sft" / "metrics.jsonl")
    train = [r for r in rows if r["type"] == "train"]
    val = [r for r in rows if r["type"] == "val"]
    best = min(val, key=lambda r: r["val_loss"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    steps, losses = [r["step"] for r in train], [r["loss"] for r in train]
    ax.plot(steps, losses, color=COLORS["train"], alpha=0.35, lw=0.9, label="train loss (per logged step)")
    ax.plot(steps[9:], running_mean(losses, 10), color=COLORS["train"], lw=1.8,
            label="train loss (mean of last 10 logged steps)")
    ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", color=COLORS["hh"], lw=1.8, ms=4,
            label="validation NLL (1,000 examples)")
    ax.axvline(best["step"], color="green", ls="--", lw=1.2, label=f"selected checkpoint (step {best['step']})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("response-token NLL")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.set_ylim(1.15, 1.70)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_dpo(results: Path, out: Path) -> None:
    runs = {}
    for tag in ("hh", "uf"):
        rows = read_jsonl(results / f"dpo_{tag}" / "metrics.jsonl")
        summary = json.loads((results / f"dpo_{tag}" / "summary.json").read_text(encoding="utf-8"))
        runs[tag] = ([r for r in rows if r["type"] == "train"], [r for r in rows if r["type"] == "val"],
                     summary["best_step"])
    labels = {"hh": "HH-RLHF", "uf": "UltraFeedback"}
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.1))
    for tag, (train, val, best) in runs.items():
        c = COLORS[tag]
        axes[0].plot([r["step"] for r in train], ema([r["loss"] for r in train]), color=c, lw=1.5, label=labels[tag])
        vs = [r["step"] for r in val]
        axes[1].plot(vs, [r["val_loss"] for r in val], "o-", color=c, ms=3.5, lw=1.5, label=labels[tag])
        axes[2].plot([r["step"] for r in train], ema([r["reward_acc"] for r in train], 0.95), color=c, lw=0.9,
                     alpha=0.35)
        axes[2].plot(vs, [r["val_reward_acc"] for r in val], "o-", color=c, ms=3.5, lw=1.5, label=labels[tag])
        axes[3].plot(vs, [r["val_margin"] for r in val], "o-", color=c, ms=3.5, lw=1.5, label=labels[tag])
        for ax in axes:
            ax.axvline(best, color=c, ls="--", lw=1.0, alpha=0.8)
    axes[0].set_title("train DPO loss (EMA 0.9)")
    axes[1].set_title("validation DPO loss")
    axes[2].set_title("reward accuracy (strict $r_w > r_l$)")
    axes[3].set_title("validation margin $r_w - r_l$")
    axes[2].axhline(0.5, color="black", lw=0.8, ls=":")
    for ax, ylabel in zip(axes, ("DPO loss", "DPO loss", "reward accuracy", "mean margin")):
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_lora(results: Path, out: Path) -> None:
    hist = json.loads((results / "lora_persona" / "log_history.json").read_text(encoding="utf-8"))
    train = [r for r in hist if "loss" in r and "eval_loss" not in r]
    ev = [r for r in hist if "eval_loss" in r]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    pairs = [("loss", "eval_loss", "DPO loss"), ("rewards/accuracies", "eval_rewards/accuracies", "reward accuracy"),
             ("rewards/margins", "eval_rewards/margins", "reward margin")]
    final_step = max(r["step"] for r in hist)
    for ax, (tk, ek, title) in zip(axes, pairs):
        ax.plot([r["step"] for r in train], [r[tk] for r in train], color=COLORS["train"], lw=1.6, label="train")
        ax.plot([r["step"] for r in ev], [r[ek] for r in ev], "o-", color=COLORS["uf"], ms=4.5, lw=1.6,
                label="eval (20 pairs)")
        ax.axvline(final_step, color="green", ls="--", lw=1.0, label=f"kept adapter (step {final_step})")
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(title)
    axes[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(0.9, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out_dir", default="report/figures")
    args = ap.parse_args(argv)
    results, out_dir = Path(args.results), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sft(results, out_dir / "sft.png")
    plot_dpo(results, out_dir / "dpo.png")
    plot_lora(results, out_dir / "lora.png")
    print(f"wrote {out_dir}/sft.png, dpo.png, lora.png")


if __name__ == "__main__":
    main()
