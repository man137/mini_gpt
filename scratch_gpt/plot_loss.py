"""Plot the training/validation loss curves written by train.py.

Usage: python scratch_gpt/plot_loss.py [--log scratch_gpt/logs/train_log.csv]
Creates scratch_gpt/logs/loss_curve.png (and prints a summary table).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render to file without a display
import matplotlib.pyplot as plt

LOG = Path(__file__).resolve().parent / "logs" / "train_log.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=LOG)
    p.add_argument("--out", type=Path, default=LOG.parent / "loss_curve.png")
    args = p.parse_args()

    if not args.log.exists():
        raise SystemExit(f"no log at {args.log} - run `python scratch_gpt/train.py` first")

    steps, train, val = [], [], []
    with open(args.log, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(steps, train, label="train loss", linewidth=2)
    ax.plot(steps, val, label="val loss", linewidth=2)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title("From-scratch GPT - training curves")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    print(f"wrote {args.out} ({len(steps)} points)")
    print(f"first eval: train={train[0]:.4f} val={val[0]:.4f}")
    print(f"last  eval: train={train[-1]:.4f} val={val[-1]:.4f}")
    print(f"best  val : {min(val):.4f} at step {steps[val.index(min(val))]}")


if __name__ == "__main__":
    main()
