"""
The single MLP head shared by every frozen-embedding probe (and available for fine-tuning).

`AtomicaProbeHead` is **exactly the head the frozen sequence-model baselines use, plus BatchNorm**:

    Linear(d, hidden) -> BatchNorm -> ReLU -> Dropout
    Linear(hidden, hidden) -> BatchNorm -> ReLU -> Dropout
    Linear(hidden, final_hidden) -> BatchNorm -> ReLU -> Dropout
    Linear(final_hidden, num_classes)

Same depth, same widths, same 32-d bottleneck, same activation/dropout placement as
the frozen sequence-model baselines' MLP; the only architectural difference is the three BatchNorm
layers. That matters because it makes the head comparison a controlled test of BatchNorm rather than of
architecture, which is what lets us claim protocol parity with the baselines.

Why BatchNorm is needed here: the invariant descriptors are wide and heterogeneously scaled (Gram entries
are orders of magnitude larger than the L2-normalized scalar readout). Note it is **complementary to**, not
a replacement for, train-fit z-scoring of the inputs -- BatchNorm sits *after* the first Linear, so it
never normalizes the raw input, and the first weight matrix still has to cope with the raw scale. Using
both was worth +0.02-0.04 validation at every pooling we tried.

Consequence: BatchNorm needs >= 2 samples per batch, so training must skip singleton batches.
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
        """`use_batchnorm=False` gives EXACTLY the baselines' MLP -- same depth, widths,
        bottleneck, and activation/dropout placement, with nothing added.

        The flag exists so BatchNorm can be a swept axis rather than a standing advantage. With it
        fixed on, "our head has BatchNorm and theirs does not" is an uncontrolled difference in every
        comparison; with it swept, every model gets the choice and validation decides per model.
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
