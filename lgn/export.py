"""Export a netlist for the browser demo or for my Logic Circuit Builder."""

import json

from .circuit import Netlist
from .ops import OP_NAMES


def to_demo_js(net: Netlist, path, extra: dict | None = None):
    """Write the netlist as `window.LGN_MODEL = {...}` so the demo works from file://."""
    d = net.to_dict()
    d["meta"] = {**d.get("meta", {}), **(extra or {})}
    with open(path, "w") as f:
        f.write("window.LGN_MODEL = ")
        json.dump(d, f, separators=(",", ":"))
        f.write(";\n")


# How each op is built from the builder's gate set (AND, OR, NOT, NAND, NOR, XOR).
# "!a" means the input passes through a NOT gate first.
_BUILDER_RECIPES = {
    1: ("AND", "a", "b"), 7: ("OR", "a", "b"), 8: ("NOR", "a", "b"),
    14: ("NAND", "a", "b"), 6: ("XOR", "a", "b"), 12: ("NOT", "a"),
    2: ("AND", "a", "!b"), 4: ("AND", "!a", "b"),
    11: ("OR", "a", "!b"), 13: ("OR", "!a", "b"),
    9: ("XNOR",),  # NOT(XOR(a, b))
}


def to_builder(net: Netlist, input_labels=None, output_labels=None) -> dict:
    """State for the Logic Circuit Builder: {"inputs": [...], "gates": [...], "outputs": [...]}.

    Ops the builder doesn't have natively are expanded into its gate set,
    so the gate count can be a little higher than the netlist's.
    """
    n_in = net.n_inputs
    input_labels = input_labels or [f"IN{i + 1}" for i in range(n_in)]
    output_labels = output_labels or [f"OUT{k + 1}" for k in range(len(net.outputs))]
    next_id = iter(range(1, 10**9))

    inputs = [{"id": next(next_id), "x": 50, "y": 80 + 80 * i, "value": 0, "label": lab}
              for i, lab in enumerate(input_labels)]
    gates, conn_of, not_cache = [], {}, {}
    column_fill: dict[int, int] = {}

    def place(col):
        row = column_fill.get(col, 0)
        column_fill[col] = row + 1
        return 200 + 150 * col, 60 + 90 * row

    def conn(node):
        if node < 0:
            raise ValueError("the builder has no constant source; this output is constant")
        if node < n_in:
            return {"sourceId": inputs[node]["id"], "type": "input"}
        return conn_of[node]

    def gate(kind, srcs, col):
        x, y = place(col)
        g = {"id": next(next_id), "type": kind, "x": x, "y": y,
             "inputConnections": srcs, "output": 0}
        gates.append(g)
        return {"sourceId": g["id"], "type": "gate"}

    def negated(node, col):
        if node not in not_cache:
            not_cache[node] = gate("NOT", [conn(node)], col)
        return not_cache[node]

    level = {n_in + j: int(lv) for j, lv in enumerate(net.levels)}
    for j, (op, a, b) in enumerate(zip(net.ops.tolist(), net.a.tolist(), net.b.tolist())):
        col = 2 * level[n_in + j]
        recipe = _BUILDER_RECIPES.get(op)
        if recipe is None:
            raise ValueError(f"unexpected op {OP_NAMES[op]} in a simplified netlist")
        if recipe[0] == "XNOR":
            x = gate("XOR", [conn(a), conn(b)], col - 1)
            conn_of[n_in + j] = gate("NOT", [x], col)
            continue
        srcs = []
        for s in recipe[1:]:
            node = a if s.endswith("a") else b
            srcs.append(negated(node, col - 1) if s.startswith("!") else conn(node))
        conn_of[n_in + j] = gate(recipe[0], srcs, col)

    last_col = 200 + 150 * (max(column_fill) + 1 if column_fill else 0)
    outputs = [{"id": next(next_id), "x": last_col, "y": 80 + 80 * k, "value": 0,
                "label": lab, "connection": conn(int(r))}
               for k, (r, lab) in enumerate(zip(net.outputs, output_labels))]
    return {"inputs": inputs, "gates": gates, "outputs": outputs}


def describe(net: Netlist, input_labels=None, output_labels=None) -> str:
    """Human-readable listing of the circuit, one gate per line."""
    n_in = net.n_inputs
    input_labels = input_labels or [f"x{i}" for i in range(n_in)]
    name = lambda r: ("0" if r == -1 else "1" if r == -2 else
                      input_labels[r] if r < n_in else f"g{r - n_in}")
    lines = []
    for j, (op, a, b) in enumerate(zip(net.ops.tolist(), net.a.tolist(), net.b.tolist())):
        args = name(a) if op == 12 else f"{name(a)}, {name(b)}"
        lines.append(f"g{j} = {OP_NAMES[op]}({args})")
    for k, r in enumerate(net.outputs.tolist()):
        lab = output_labels[k] if output_labels else f"y{k}"
        lines.append(f"{lab} = {name(r)}")
    return "\n".join(lines)
