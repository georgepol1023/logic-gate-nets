"""Write a trained circuit into the browser demo.

    python export_demo.py runs/mnist_l4_w6000
"""

import argparse
import json
from pathlib import Path

from lgn import Netlist
from lgn.export import to_demo_js


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=Path("docs/model.js"))
    args = p.parse_args()

    net = Netlist.load(args.run / "netlist.json")
    metrics = json.loads((args.run / "metrics.json").read_text())
    extra = {"test_accuracy": metrics["netlist_test_acc"], "gates_trained": metrics["gates_trained"],
             "depth": net.depth}
    to_demo_js(net, args.out, extra)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {net.n_gates:,} gates)")


if __name__ == "__main__":
    main()
