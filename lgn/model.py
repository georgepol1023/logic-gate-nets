import torch
import torch.nn as nn

from .layers import GroupSum, LogicLayer


class LogicNet(nn.Module):
    """A stack of logic layers.

    With `n_classes` set, a GroupSum head turns the last layer into class
    scores. Without it, the last layer's gates are the outputs (used for
    learning exact boolean functions such as an adder).
    """

    def __init__(self, in_dim: int, widths: list[int], n_classes: int | None = None,
                 tau: float = 10.0, init: str = "residual", seed: int = 0):
        super().__init__()
        if n_classes is not None and widths[-1] % n_classes:
            raise ValueError("last layer width must be divisible by n_classes")
        g = torch.Generator().manual_seed(seed)
        dims = [in_dim] + list(widths)
        self.layers = nn.ModuleList(
            LogicLayer(dims[i], dims[i + 1], init=init, generator=g) for i in range(len(widths))
        )
        self.head = GroupSum(n_classes, tau) if n_classes is not None else None
        self.config = dict(in_dim=in_dim, widths=list(widths), n_classes=n_classes,
                           tau=tau, init=init, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x) if self.head is not None else x

    @property
    def n_gates(self) -> int:
        return sum(l.out_dim for l in self.layers)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, hard: bool = True, batch_size: int = 4096) -> torch.Tensor:
        """Run the model in hard (discretized) or soft (relaxed) mode."""
        was_training = self.training
        self.train(not hard)
        out = torch.cat([self(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
        self.train(was_training)
        return out

    def save(self, path):
        torch.save({"config": self.config, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state"])
        return model
