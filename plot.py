"""Make the figures used in the README.

    python plot.py runs/mnist_l4_w6000 --out docs/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lgn.ops import OP_NAMES

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def curves(run: Path, out: Path):
    h = json.loads((run / "history.json").read_text())
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(ep, [r["soft_test_acc"] for r in h], label="relaxed (training mode)", color="#8a94a6")
    ax.plot(ep, [r["hard_test_acc"] for r in h], label="discrete circuit", color="#d9930d", lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"{run.name}_curves.png", dpi=160)
    plt.close(fig)


def ops(run: Path, out: Path):
    m = json.loads((run / "metrics.json").read_text())
    names = [n for n in OP_NAMES]
    trained = [m["ops_trained"].get(n, 0) for n in names]
    simple = [m["ops_after_simplify"].get(n, 0) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = range(len(names))
    ax.bar([x - 0.2 for x in xs], trained, width=0.4, label=f"trained ({m['gates_trained']:,})", color="#c5cbd6")
    ax.bar([x + 0.2 for x in xs], simple, width=0.4, label=f"after simplification ({m['gates_after_simplify']:,})",
           color="#d9930d")
    ax.set_xticks(list(xs), [n.replace("_", " ") for n in names], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("gates")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{run.name}_ops.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        curves(run, args.out)
        ops(run, args.out)
        print(f"wrote figures for {run.name}")


if __name__ == "__main__":
    main()
