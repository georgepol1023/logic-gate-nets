"""A conventional float MLP on exactly the same binarized inputs, for comparison.

    python baseline_mlp.py --dataset mnist --hidden 256 256
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lgn.data import DEFAULT_THRESHOLDS, load, thermometer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    ds = load(args.dataset)
    th = DEFAULT_THRESHOLDS[args.dataset]
    x_tr = torch.from_numpy(thermometer(ds.x_train, th))
    x_te = torch.from_numpy(thermometer(ds.x_test, th))
    y_tr, y_te = torch.from_numpy(ds.y_train), torch.from_numpy(ds.y_test)

    dims = [x_tr.shape[1]] + args.hidden + [ds.n_classes]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        print(f"epoch {epoch:3d}  test acc {acc:.4f}")

    params = sum(p.numel() for p in model.parameters())
    macs = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    out = Path("runs") / f"mlp_{ds.name}"
    out.mkdir(parents=True, exist_ok=True)
    weights = {f"w{i}": l.weight.detach().numpy().T.copy() for i, l in enumerate(model) if isinstance(l, nn.Linear)}
    weights.update({f"b{i}": l.bias.detach().numpy().copy() for i, l in enumerate(model) if isinstance(l, nn.Linear)})
    np.savez(out / "weights.npz", **weights)
    metrics = {"args": vars(args), "test_acc": acc, "params": params,
               "multiply_accumulates": macs, "train_seconds": time.time() - t0}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\ntest acc {acc:.4f}, {params:,} float params, {macs:,} MACs per image -> {out}/")


if __name__ == "__main__":
    main()
