"""Train a logic gate network on digits or MNIST.

    python train.py --dataset digits
    python train.py --dataset mnist --layers 6 --width 8000 --tau 20 --epochs 30

Writes to runs/<name>/:
    model.pt        the trained network
    netlist.json    the simplified boolean circuit
    history.json    loss and accuracy per epoch (soft and hard)
    metrics.json    final numbers
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lgn import LogicNet, Netlist
from lgn.circuit import verify
from lgn.data import DEFAULT_THRESHOLDS, load, thermometer


def accuracy(scores: torch.Tensor, y: torch.Tensor) -> float:
    return (scores.argmax(1) == y).float().mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--width", type=int, default=2000)
    p.add_argument("--tau", type=float, default=10.0, help="group-sum temperature")
    p.add_argument("--thresholds", type=float, nargs="+", help="thermometer thresholds")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--init", default="gaussian", choices=["residual", "gaussian"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", help="run name (default: derived from settings)")
    p.add_argument("--threads", type=int, help="torch CPU threads")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    name = args.name or f"{args.dataset}_l{args.layers}_w{args.width}_s{args.seed}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    ds = load(args.dataset)
    thresholds = args.thresholds or DEFAULT_THRESHOLDS[args.dataset]
    x_tr = torch.from_numpy(thermometer(ds.x_train, thresholds))
    x_te = torch.from_numpy(thermometer(ds.x_test, thresholds))
    y_tr, y_te = torch.from_numpy(ds.y_train), torch.from_numpy(ds.y_test)
    print(f"{ds.name}: {len(x_tr)} train, {len(x_te)} test, {x_tr.shape[1]} input bits")

    model = LogicNet(x_tr.shape[1], [args.width] * args.layers, ds.n_classes,
                     tau=args.tau, init=args.init, seed=args.seed)
    print(f"{args.layers} layers x {args.width} gates = {model.n_gates:,} gates")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(x_tr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        row = {
            "epoch": epoch,
            "loss": total / len(x_tr),
            "soft_test_acc": accuracy(model.predict(x_te, hard=False), y_te),
            "hard_test_acc": accuracy(model.predict(x_te, hard=True), y_te),
            "hard_train_acc": accuracy(model.predict(x_tr, hard=True), y_tr),
            "seconds": time.time() - t_start,
        }
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['loss']:.4f}  test soft {row['soft_test_acc']:.4f}  "
              f"hard {row['hard_test_acc']:.4f}  ({row['seconds']:.0f}s)")

    # report the final epoch, not the best one: picking the best epoch by test
    # accuracy would quietly tune on the test set
    model.save(out / "model.pt")
    final = history[-1]
    meta = {"dataset": ds.name, "thresholds": thresholds, "n_classes": ds.n_classes,
            "image_shape": list(ds.image_shape), "max_value": ds.max_value}
    raw = Netlist.from_model(model, simplify=False, meta=meta)
    net = Netlist.from_model(model, simplify=True, meta=meta)

    xb_te = x_te.numpy().astype(bool)
    assert verify(net, model, xb_te), "simplified netlist disagrees with the model"
    net_acc = float((net.class_scores(xb_te, ds.n_classes).argmax(1) == ds.y_test).mean())
    net.save(out / "netlist.json")

    metrics = {
        "args": vars(args),
        "epochs": args.epochs,
        "soft_test_acc": final["soft_test_acc"],
        "hard_test_acc": final["hard_test_acc"],
        "netlist_test_acc": net_acc,
        "gates_trained": raw.n_gates,
        "gates_after_simplify": net.n_gates,
        "depth": net.depth,
        "ops_trained": raw.op_histogram(),
        "ops_after_simplify": net.op_histogram(),
        "train_seconds": history[-1]["seconds"],
    }
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\nfinal: soft test acc {final['soft_test_acc']:.4f}, hard {final['hard_test_acc']:.4f}")
    print(f"circuit: {raw.n_gates:,} gates trained -> {net.n_gates:,} after simplification, depth {net.depth}")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
