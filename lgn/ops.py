"""The 16 two-input boolean functions and their real-valued relaxations.

Op ids follow the truth-table encoding: bit (3 - (2a + b)) of the id is f(a, b),
so id 1 = 0b0001 is AND (only f(1,1) = 1), id 6 = 0b0110 is XOR, and so on.

Every two-input boolean function has a unique multilinear extension

    f(a, b) = c0 + c1*a + c2*b + c3*a*b

which agrees with f on {0,1}^2 and is a probabilistic interpretation when a and b
are independent probabilities of being 1. This lets a whole layer be evaluated
with four coefficients per gate instead of sixteen separate ops.
"""

import torch

OP_NAMES = [
    "FALSE", "AND", "A_AND_NOT_B", "A",
    "NOT_A_AND_B", "B", "XOR", "OR",
    "NOR", "XNOR", "NOT_B", "A_OR_NOT_B",
    "NOT_A", "NOT_A_OR_B", "NAND", "TRUE",
]
N_OPS = 16

# op ids used directly by name elsewhere
FALSE, AND, A_AND_NOT_B, A, NOT_A_AND_B, B, XOR, OR = range(8)
NOR, XNOR, NOT_B, A_OR_NOT_B, NOT_A, NOT_A_OR_B, NAND, TRUE = range(8, 16)


def truth(op: int, a: int, b: int) -> int:
    """Evaluate op on boolean inputs."""
    return (op >> (3 - (2 * a + b))) & 1


def truth_table(op: int):
    """(f(0,0), f(0,1), f(1,0), f(1,1))."""
    return tuple(truth(op, a, b) for a in (0, 1) for b in (0, 1))


def multilinear_coefficients() -> torch.Tensor:
    """(16, 4) tensor of [c0, c1, c2, c3] for every op."""
    rows = []
    for op in range(N_OPS):
        f00, f01, f10, f11 = truth_table(op)
        rows.append([f00, f10 - f00, f01 - f00, f11 - f10 - f01 + f00])
    return torch.tensor(rows, dtype=torch.float32)
