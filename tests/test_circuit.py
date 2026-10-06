import itertools

import numpy as np
import torch

from lgn import LogicNet, Netlist
from lgn.circuit import pack, unpack, verify
from lgn.codegen import to_cpp
from lgn.ops import multilinear_coefficients, truth


def test_relaxation_matches_truth_tables():
    coef = multilinear_coefficients()
    for op in range(16):
        for a, b in itertools.product([0, 1], repeat=2):
            c = coef[op]
            assert c[0] + c[1] * a + c[2] * b + c[3] * a * b == truth(op, a, b)


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.random((130, 7)) > 0.5
    assert np.array_equal(unpack(pack(x), 130), x)


def _random_trained_looking_model(seed, in_dim=12, widths=(40, 40, 20)):
    model = LogicNet(in_dim, list(widths), n_classes=None, init="gaussian", seed=seed)
    # sharpen the logits so every op appears, including constants and wires
    with torch.no_grad():
        for layer in model.layers:
            layer.weights.mul_(3)
    return model


def test_simplified_netlist_is_exact():
    rng = np.random.default_rng(1)
    for seed in range(20):
        model = _random_trained_looking_model(seed)
        x = rng.random((500, 12)) > 0.5
        raw = Netlist.from_model(model, simplify=False)
        net = Netlist.from_model(model)
        assert verify(raw, model, x)
        assert verify(net, model, x)
        assert net.n_gates <= raw.n_gates


def test_cpp_codegen_mentions_every_gate():
    net = Netlist.from_model(_random_trained_looking_model(0))
    src = to_cpp(net)
    assert src.count("    v[") == net.n_gates
