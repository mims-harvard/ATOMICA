"""The MLP head shared by every frozen-embedding probe.

`AtomicaProbeHead` is the head the frozen sequence-model baselines use, plus BatchNorm:

    Linear(d, hidden) -> BatchNorm -> ReLU -> Dropout
    Linear(hidden, hidden) -> BatchNorm -> ReLU -> Dropout
    Linear(hidden, final_hidden) -> BatchNorm -> ReLU -> Dropout
    Linear(final_hidden, num_classes)

Same depth, widths, bottleneck and dropout placement as the baselines' MLP, so the comparison
stays a comparison of representations. BatchNorm is there because the invariant descriptors are
wide and heterogeneously scaled; it complements train-fit z-scoring of the inputs rather than
replacing it, since it sits after the first Linear.

BatchNorm needs at least two samples per batch, so training skips singleton batches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

TASK_TYPES = ("binary", "multiclass", "multilabel")


class AtomicaProbeHead(nn.Module):
    """MLP head over a frozen embedding. Returns logits; activation is applied by the caller."""

    def __init__(self, input_dim: int, num_classes: int, task_type: str = "multiclass",
                 hidden_dim: int = 512, final_hidden_dim: int = 32, dropout: float = 0.3,
                 use_batchnorm: bool = True):
        """`use_batchnorm=False` gives the baselines' MLP exactly, with nothing added.

        Keeping it a swept option rather than always on means every model gets the same choice and
        validation decides per model.
        """
        super().__init__()
        if task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {task_type!r}")
        self.task_type = task_type
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm

        def block(d_in: int, d_out: int) -> list:
            layers: list = [nn.Linear(d_in, d_out)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(d_out))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            return layers

        self.net = nn.Sequential(
            *block(input_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, final_hidden_dim),
            nn.Linear(final_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, apply_activation: bool = False) -> torch.Tensor:
        logits = self.net(x)
        if not apply_activation:
            return logits
        return torch.sigmoid(logits) if self.task_type in ("binary", "multilabel") \
            else torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """The `final_hidden_dim` penultimate activation (everything but the output Linear)."""
        self.eval()
        for layer in list(self.net)[:-1]:
            x = layer(x)
        return x

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits -> the probability form used for seed-ensembling."""
        return torch.sigmoid(logits) if self.task_type in ("binary", "multilabel") \
            else torch.softmax(logits, dim=-1)


def num_outputs(task_type: str, y) -> int:
    """Output width implied by the labels: binary -> 1, multiclass -> max+1, multilabel -> n_labels."""
    import numpy as np
    y = np.asarray(y)
    if task_type == "binary":
        return 1
    if task_type == "multiclass":
        return int(y.max()) + 1
    return int(y.shape[1])
