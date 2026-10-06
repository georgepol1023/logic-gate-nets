"""Learn an n-bit adder from its truth table, then read the circuit back out.

    python adder.py --bits 2

A small enough task that the learned circuit fits on a screen, so you can
compare it to a hand-designed ripple-carry adder. Writes:
    runs/adder<n>/circuit.txt      gate-by-gate listing
    runs/adder<n>/builder.json     state for the Logic Circuit Builder
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lgn import LogicNet, Netlist
from lgn.circuit import verify
from lgn.export import describe, to_builder


def truth_table(bits: int):
    rows, targets = [], []
    for x in itertools.product([0, 1], repeat=2 * bits):
        a = int("".join(map(str, x[:bits])), 2)
        b = int("".join(map(str, x[bits:])), 2)
        s = a + b
        rows.append(x)
        targets.append([(s >> (bits - k)) & 1 for k in range(bits + 1)])  # MSB first
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def train_once(x, y, widths, seed, steps, lr):
    model = LogicNet(x.shape[1], widths, n_classes=None, init="gaussian", seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        model.train()
        out = model(x).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 99 and torch.equal(model.predict(x) > 0.5, y > 0.5):
            return model, step + 1
    return None, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=4, help="hidden layers before the output layer")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-seeds", type=int, default=50)
    args = p.parse_args()

    n = args.bits
    x, y = truth_table(n)
    widths = [args.width] * args.layers + [n + 1]
    in_labels = [f"A{n - 1 - i}" for i in range(n)] + [f"B{n - 1 - i}" for i in range(n)]
    out_labels = [f"S{n - k}" for k in range(n + 1)]
    print(f"{n}-bit adder: {len(x)} truth-table rows, {sum(widths)} gates to train")

    solved, tried = [], 0
    for seed in range(args.max_seeds):
        tried += 1
        model, steps = train_once(x, y, widths, seed, args.steps, args.lr)
        if model is None:
            print(f"seed {seed:2d}: not exact after {steps} steps")
            continue
        net = Netlist.from_model(model)
        assert verify(net, model, x.numpy().astype(bool))
        print(f"seed {seed:2d}: exact after {steps} steps -> {net.n_gates} gates, depth {net.depth}")
        solved.append((net.n_gates, net.depth, seed, net))
        if len(solved) >= 10:
            break

    if not solved:
        raise SystemExit("no seed found an exact circuit; try more width or steps")

    gates, depth, seed, net = min(solved, key=lambda s: (s[0], s[1]))
    out = Path("runs") / f"adder{n}"
    out.mkdir(parents=True, exist_ok=True)
    listing = describe(net, in_labels, out_labels)
    (out / "circuit.txt").write_text(listing + "\n")
    (out / "builder.json").write_text(json.dumps(to_builder(net, in_labels, out_labels), indent=1))
    summary = {"bits": n, "trained_gates": sum(widths), "seeds_tried": tried,
               "exact_solutions": len(solved), "smallest_circuit_gates": gates,
               "smallest_circuit_depth": depth,
               "gate_counts_of_solutions": sorted(s[0] for s in solved),
               "ops": net.op_histogram()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsmallest exact circuit (seed {seed}): {gates} gates, depth {depth}\n")
    print(listing)


if __name__ == "__main__":
    main()
