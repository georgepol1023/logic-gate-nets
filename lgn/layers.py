import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import A, N_OPS, multilinear_coefficients


def random_connections(in_dim: int, out_dim: int, generator: torch.Generator):
    """Pick two inputs for every gate.

    Inputs are drawn from back-to-back random permutations, so every input is
    used (as long as 2 * out_dim >= in_dim) and no input is badly over-used.
    Wiring is fixed at init and never learned: only the choice of gate is.
    """
    n = 2 * out_dim
    reps = -(-n // in_dim)
    idx = torch.cat([torch.randperm(in_dim, generator=generator) for _ in range(reps)])[:n]
    idx = idx[torch.randperm(n, generator=generator)]
    a, b = idx[:out_dim].clone(), idx[out_dim:].clone()
    # avoid wiring both inputs of a gate to the same wire where possible
    same = (a == b).nonzero().flatten()
    if in_dim > 1:
        b[same] = (b[same] + 1) % in_dim
    return a, b


class LogicLayer(nn.Module):
    """A layer of `out_dim` two-input gates, each choosing one of 16 boolean ops.

    Training (`.train()`): each gate outputs a softmax-weighted mix of the 16
    relaxed ops, so the whole thing is differentiable.
    Evaluation (`.eval()`): each gate commits to its most likely op, which makes
    the layer an exact boolean circuit on binary inputs.
    """

    def __init__(self, in_dim: int, out_dim: int, init: str = "residual",
                 generator: torch.Generator | None = None):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        g = generator if generator is not None else torch.Generator().manual_seed(0)

        a, b = random_connections(in_dim, out_dim, g)
        self.register_buffer("idx_a", a)
        self.register_buffer("idx_b", b)
        self.register_buffer("coef", multilinear_coefficients())

        w = torch.randn(out_dim, N_OPS, generator=g)
        if init == "residual":
            # Start every gate close to a pass-through of its first input.
            # Without this, deep stacks start as noise and gradients vanish.
            w[:, A] += 5.0
        elif init != "gaussian":
            raise ValueError(f"unknown init {init!r}")
        self.weights = nn.Parameter(w)

    def op_probs(self) -> torch.Tensor:
        return F.softmax(self.weights, dim=-1)

    def ops(self) -> torch.Tensor:
        """The op each gate commits to when discretized."""
        return self.weights.argmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, self.idx_a]
        b = x[:, self.idx_b]
        if self.training:
            p = self.op_probs()
        else:
            p = F.one_hot(self.ops(), N_OPS).to(x.dtype)
        c = p @ self.coef  # (out_dim, 4)
        return c[:, 0] + c[:, 1] * a + c[:, 2] * b + c[:, 3] * a * b

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


class GroupSum(nn.Module):
    """Split the final layer into `k` equal groups and count the 1s in each.

    The counts are the class scores. `tau` scales them into a range where
    softmax cross-entropy trains well; it doesn't change the argmax.
    """

    def __init__(self, k: int, tau: float):
        super().__init__()
        self.k, self.tau = k, tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.k, -1).sum(-1) / self.tau

    def extra_repr(self) -> str:
        return f"k={self.k}, tau={self.tau}"
