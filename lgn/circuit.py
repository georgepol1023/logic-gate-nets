"""Turn a trained LogicNet into a plain boolean netlist and run it fast.

A trained network contains many gates that do nothing useful: pass-throughs,
constants, duplicates and gates whose output never reaches the classifier.
`Netlist.from_model` removes them with a few classic logic-synthesis passes:

  * constant propagation (AND with 0 is 0, ...)
  * wire collapsing (the A and B ops are just wires)
  * negation absorption (AND(NOT x, y) becomes a single A_AND_NOT_B-style gate)
  * structural hashing (identical gates are merged)
  * dead-gate elimination (anything not reachable from an output is dropped)

The result computes exactly the same function as the network in eval mode,
which `verify` checks.
"""

import json
from collections import Counter

import numpy as np

from .ops import N_OPS, NOT_A, OP_NAMES, truth

CONST0, CONST1 = -1, -2


def _op_from_fn(fn) -> int:
    return sum(fn(a, b) << (3 - (2 * a + b)) for a in (0, 1) for b in (0, 1))


class _Builder:
    """Builds a simplified netlist one gate at a time."""

    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        self.gates: list[tuple[int, int, int]] = []
        self.hash: dict[tuple[int, int, int], int] = {}
        self.not_of: dict[int, int] = {}  # node -> x, for nodes that are NOT x

    def _node(self, op, a, b) -> int:
        key = (op, a, b)
        if key not in self.hash:
            self.hash[key] = self.n_inputs + len(self.gates)
            self.gates.append(key)
            if op == NOT_A:
                self.not_of[self.hash[key]] = a
        return self.hash[key]

    def negate(self, r: int) -> int:
        if r == CONST0:
            return CONST1
        if r == CONST1:
            return CONST0
        if r in self.not_of:
            return self.not_of[r]
        if r >= self.n_inputs:
            # NOT of a gate is the gate with its truth table inverted
            # (op ^ 15). If the original gate ends up unused, pruning drops it.
            op, a, b = self.gates[r - self.n_inputs]
            return self._node(op ^ 15, a, b)
        return self._node(NOT_A, r, r)

    def unary(self, r: int, f0: int, f1: int) -> int:
        """A function of one wire, given by its outputs on 0 and 1."""
        if f0 == f1:
            return CONST1 if f0 else CONST0
        return r if f1 else self.negate(r)

    def add(self, op: int, ra: int, rb: int) -> int:
        f = lambda a, b, op=op: truth(op, a, b)

        # absorb negations on the inputs into the op itself
        if ra in self.not_of:
            ra, f = self.not_of[ra], (lambda a, b, g=f: g(1 - a, b))
        if rb in self.not_of:
            rb, f = self.not_of[rb], (lambda a, b, g=f: g(a, 1 - b))

        # constant inputs
        if ra in (CONST0, CONST1):
            ca = int(ra == CONST1)
            return self.unary(rb, f(ca, 0), f(ca, 1)) if rb >= 0 else \
                (CONST1 if f(ca, int(rb == CONST1)) else CONST0)
        if rb in (CONST0, CONST1):
            cb = int(rb == CONST1)
            return self.unary(ra, f(0, cb), f(1, cb))

        # both inputs are the same wire
        if ra == rb:
            return self.unary(ra, f(0, 0), f(1, 1))

        uses_a = any(f(0, b) != f(1, b) for b in (0, 1))
        uses_b = any(f(a, 0) != f(a, 1) for a in (0, 1))
        if not uses_a and not uses_b:
            return CONST1 if f(0, 0) else CONST0
        if not uses_b:
            return self.unary(ra, f(0, 0), f(1, 0))
        if not uses_a:
            return self.unary(rb, f(0, 0), f(0, 1))

        # canonical input order so commuted duplicates hash together
        if ra > rb:
            ra, rb, f = rb, ra, (lambda a, b, g=f: g(b, a))
        return self._node(_op_from_fn(f), ra, rb)


class Netlist:
    """A feed-forward boolean circuit.

    Node ids: 0..n_inputs-1 are inputs, n_inputs + j is gate j. Gates are in
    topological order. Outputs may also point at an input or a constant
    (CONST0 = -1, CONST1 = -2). NOT gates are stored as op NOT_A with b == a.
    """

    def __init__(self, n_inputs, ops, a, b, outputs, meta=None):
        self.n_inputs = int(n_inputs)
        self.ops = np.asarray(ops, dtype=np.int8)
        self.a = np.asarray(a, dtype=np.int64)
        self.b = np.asarray(b, dtype=np.int64)
        self.outputs = np.asarray(outputs, dtype=np.int64)
        self.meta = meta or {}
        self._levels = None

    # ---- construction -------------------------------------------------

    @classmethod
    def from_model(cls, model, simplify: bool = True, meta=None) -> "Netlist":
        n_in = model.layers[0].in_dim
        if not simplify:
            ops, a, b, offset, prev = [], [], [], n_in, np.arange(n_in)
            for layer in model.layers:
                ops.append(layer.ops().numpy())
                a.append(prev[layer.idx_a.numpy()])
                b.append(prev[layer.idx_b.numpy()])
                prev = np.arange(offset, offset + layer.out_dim)
                offset += layer.out_dim
            return cls(n_in, np.concatenate(ops), np.concatenate(a), np.concatenate(b), prev, meta)

        builder = _Builder(n_in)
        refs = list(range(n_in))
        for layer in model.layers:
            ops, ia, ib = layer.ops().tolist(), layer.idx_a.tolist(), layer.idx_b.tolist()
            refs = [builder.add(op, refs[x], refs[y]) for op, x, y in zip(ops, ia, ib)]

        g = builder.gates
        net = cls(n_in, [x[0] for x in g], [x[1] for x in g], [x[2] for x in g], refs, meta)
        return net.prune()

    def prune(self) -> "Netlist":
        """Drop gates that no output depends on, renumbering the rest."""
        n_in, n_g = self.n_inputs, len(self.ops)
        live = np.zeros(n_in + n_g, dtype=bool)
        live[self.outputs[self.outputs >= 0]] = True
        for j in range(n_g - 1, -1, -1):
            if live[n_in + j]:
                live[self.a[j]] = live[self.b[j]] = True
        keep = np.nonzero(live[n_in:])[0]
        remap = np.full(n_in + n_g, -999, dtype=np.int64)
        remap[:n_in] = np.arange(n_in)
        remap[n_in + keep] = n_in + np.arange(len(keep))
        fix = lambda r: np.where(r >= 0, remap[np.maximum(r, 0)], r)
        return Netlist(n_in, self.ops[keep], remap[self.a[keep]], remap[self.b[keep]],
                       fix(self.outputs), self.meta)

    # ---- structure ----------------------------------------------------

    @property
    def n_gates(self) -> int:
        return len(self.ops)

    @property
    def levels(self) -> np.ndarray:
        """Logic depth of every gate (inputs are depth 0)."""
        if self._levels is None:
            depth = np.zeros(self.n_inputs + self.n_gates, dtype=np.int64)
            for j in range(self.n_gates):
                depth[self.n_inputs + j] = 1 + max(depth[self.a[j]], depth[self.b[j]])
            self._levels = depth[self.n_inputs:]
        return self._levels

    @property
    def depth(self) -> int:
        return int(self.levels.max()) if self.n_gates else 0

    def op_histogram(self) -> dict:
        c = Counter(self.ops.tolist())
        return {OP_NAMES[k]: c[k] for k in range(N_OPS) if c[k]}

    def stats(self) -> dict:
        return {"inputs": self.n_inputs, "gates": self.n_gates, "depth": self.depth,
                "outputs": len(self.outputs), "ops": self.op_histogram()}

    # ---- evaluation ---------------------------------------------------

    def run_packed(self, x_packed: np.ndarray) -> np.ndarray:
        """Evaluate on bit-packed inputs.

        x_packed: (n_inputs, W) uint64, bit i of word w is sample 64*w + i.
        Returns (n_outputs, W) uint64. Every bitwise op processes 64 samples.
        """
        n_in, n_g, W = self.n_inputs, self.n_gates, x_packed.shape[1]
        v = np.empty((n_in + n_g + 2, W), dtype=np.uint64)
        v[:n_in] = x_packed
        v[-2] = 0                      # CONST0 (index -2 via remap below)
        v[-1] = np.uint64(~np.uint64(0))  # CONST1
        for gidx, op in self._schedule():
            A, B = v[self.a[gidx]], v[self.b[gidx]]
            v[n_in + gidx] = _apply(op, A, B)
        out = np.where(self.outputs == CONST0, n_in + n_g,
                       np.where(self.outputs == CONST1, n_in + n_g + 1, self.outputs))
        return v[out]

    def _schedule(self):
        """Group gates by (depth, op) so each group is one vectorized numpy op."""
        if not hasattr(self, "_sched"):
            lv = self.levels
            self._sched = []
            for d in range(1, self.depth + 1):
                at_d = np.nonzero(lv == d)[0]
                for op in np.unique(self.ops[at_d]):
                    self._sched.append((at_d[self.ops[at_d] == op], int(op)))
        return self._sched

    def run(self, x_bits: np.ndarray) -> np.ndarray:
        """(N, n_inputs) bits -> (N, n_outputs) bool."""
        n = len(x_bits)
        return unpack(self.run_packed(pack(x_bits)), n)

    def class_scores(self, x_bits: np.ndarray, n_classes: int) -> np.ndarray:
        """Group-sum of the output bits: (N, n_classes) integer vote counts."""
        y = self.run(x_bits)
        return y.reshape(len(y), n_classes, -1).sum(-1)

    # ---- io -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {"n_inputs": self.n_inputs, "ops": self.ops.tolist(), "a": self.a.tolist(),
                "b": self.b.tolist(), "outputs": self.outputs.tolist(), "meta": self.meta}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path) -> "Netlist":
        with open(path) as f:
            d = json.load(f)
        return cls(d["n_inputs"], d["ops"], d["a"], d["b"], d["outputs"], d.get("meta"))


def _apply(op: int, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if op == 0:  return np.zeros_like(A)
    if op == 1:  return A & B
    if op == 2:  return A & ~B
    if op == 3:  return A.copy()
    if op == 4:  return ~A & B
    if op == 5:  return B.copy()
    if op == 6:  return A ^ B
    if op == 7:  return A | B
    if op == 8:  return ~(A | B)
    if op == 9:  return ~(A ^ B)
    if op == 10: return ~B
    if op == 11: return A | ~B
    if op == 12: return ~A
    if op == 13: return ~A | B
    if op == 14: return ~(A & B)
    return ~np.zeros_like(A)


def pack(x_bits: np.ndarray) -> np.ndarray:
    """(N, D) bits -> (D, ceil(N/64)) uint64."""
    n, d = x_bits.shape
    W = -(-n // 64)
    padded = np.zeros((d, W * 64), dtype=np.uint8)
    padded[:, :n] = x_bits.T
    return np.packbits(padded, axis=1, bitorder="little").view(np.uint64)


def unpack(words: np.ndarray, n: int) -> np.ndarray:
    """(D, W) uint64 -> (n, D) bool."""
    bits = np.unpackbits(np.ascontiguousarray(words).view(np.uint8), axis=1, bitorder="little")
    return bits[:, :n].T.astype(bool)


def verify(net: Netlist, model, x_bits: np.ndarray) -> bool:
    """Check the netlist reproduces the model's discretized outputs exactly."""
    import torch

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_bits.astype(np.float32))
        h = x
        for layer in model.layers:
            h = layer(h)
    return bool(np.array_equal(net.run(x_bits), h.numpy() > 0.5))
