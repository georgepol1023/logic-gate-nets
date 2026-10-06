"""Measure inference speed of a trained circuit, and check every path agrees.

    python benchmark.py runs/mnist_l4_w6000 --mlp runs/mlp_mnist

Paths compared (all single-threaded CPU, whole test set):
  pytorch     the LogicNet in eval mode, as float tensors
  numpy       the simplified netlist, bit-sliced 64 samples per uint64
  cpp         the netlist compiled to straight-line C++ (needs g++)
  mlp         the float MLP baseline as numpy matmuls, if --mlp is given
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from lgn import LogicNet, Netlist
from lgn.circuit import pack
from lgn.codegen import BENCH_MAIN, to_cpp
from lgn.data import load, thermometer


def timed(fn, min_seconds=1.0):
    """Best-of-several wall time for fn(), in seconds."""
    fn()  # warm up
    best, spent = float("inf"), 0.0
    while spent < min_seconds:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best, spent = min(best, dt), spent + dt
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--mlp", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)

    net = Netlist.load(args.run / "netlist.json")
    meta = net.meta
    ds = load(meta["dataset"])
    xb = thermometer(ds.x_test, meta["thresholds"]).astype(bool)
    n, C = len(xb), meta["n_classes"]
    results = {"samples": n}

    # numpy, bit-sliced
    scores = net.class_scores(xb, C)
    acc = float((scores.argmax(1) == ds.y_test).mean())
    packed = pack(xb)
    t = timed(lambda: net.run_packed(packed))
    results["numpy"] = {"ns_per_sample": t * 1e9 / n, "accuracy": acc}

    # pytorch, float tensors in eval mode
    model = LogicNet.load(args.run / "model.pt")
    x_t = torch.from_numpy(xb.astype(np.float32))
    t = timed(lambda: model.predict(x_t, hard=True, batch_size=n))
    results["pytorch"] = {"ns_per_sample": t * 1e9 / n}

    # generated C++
    if shutil.which("g++"):
        cdir = args.run / "cpp"
        cdir.mkdir(exist_ok=True)
        (cdir / "circuit.cpp").write_text(to_cpp(net))
        (cdir / "bench.cpp").write_text(f"#include <cstdint>\nconstexpr int LGN_INPUTS = {net.n_inputs};\n"
                                        f"constexpr int LGN_OUTPUTS = {len(net.outputs)};\n" + BENCH_MAIN)
        print("compiling generated C++...")
        t0 = time.time()
        subprocess.run(["g++", "-std=c++20", "-O2", "-march=native", "-o", str(cdir / "bench"),
                        str(cdir / "circuit.cpp"), str(cdir / "bench.cpp")], check=True)
        compile_s = time.time() - t0
        np.ascontiguousarray(packed.T).tofile(cdir / "inputs.bin")  # word-major
        res = subprocess.run([str(cdir / "bench"), str(cdir / "inputs.bin"), str(packed.shape[1]),
                              str(C), str(cdir / "counts.bin")], capture_output=True, text=True, check=True)
        c_res = json.loads(res.stdout)
        counts = np.fromfile(cdir / "counts.bin", dtype=np.int32).reshape(-1, C)[:n]
        assert np.array_equal(counts, scores), "C++ output disagrees with numpy"
        results["cpp"] = {**c_res, "compile_seconds": compile_s, "matches_numpy": True}

    # float MLP baseline
    if args.mlp:
        w = np.load(args.mlp / "weights.npz")
        keys = sorted({k[1:] for k in w.files}, key=int)
        xf = xb.astype(np.float32)

        def mlp():
            h = xf
            for i, k in enumerate(keys):
                h = h @ w[f"w{k}"] + w[f"b{k}"]
                if i < len(keys) - 1:
                    np.maximum(h, 0, out=h)
            return h

        t = timed(mlp)
        mlp_metrics = json.loads((args.mlp / "metrics.json").read_text())
        results["mlp"] = {"ns_per_sample": t * 1e9 / n, "accuracy": float((mlp().argmax(1) == ds.y_test).mean()),
                          "params": mlp_metrics["params"], "macs": mlp_metrics["multiply_accumulates"]}

    results["circuit"] = net.stats()
    (args.run / "benchmark.json").write_text(json.dumps(results, indent=1))

    print(f"\n{n} test images, {net.n_gates:,} gates, depth {net.depth}")
    for k in ("pytorch", "numpy", "cpp", "mlp"):
        if k in results:
            r = results[k]
            ns = r.get("ns_per_sample", r.get("ns_per_sample_circuit"))
            print(f"  {k:8s} {ns:12.1f} ns/image")
    if "cpp" in results:
        print(f"  cpp (with vote counting) {results['cpp']['ns_per_sample_with_votes']:.1f} ns/image")


if __name__ == "__main__":
    main()
