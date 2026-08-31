"""Segment-reduction helpers (``scatter_sum`` / ``scatter_mean`` / ``scatter_min``).

``torch_scatter`` is a compiled CUDA extension whose prebuilt wheels have to
match both the PyTorch version and the host's glibc. That coupling, rather than
anything in ATOMICA itself, is what pins an installation to one CUDA toolchain.

If ``torch_scatter`` is installed it is used unchanged, so existing environments
behave exactly as before. Otherwise the pure-PyTorch implementations below are
used; they are built on ``Tensor.scatter_add_`` and ``Tensor.scatter_reduce_``,
which ship with PyTorch itself and therefore work on any CUDA build.

Only the argument forms ATOMICA uses are supported:
``scatter_*(src, index, dim=..., dim_size=...)`` with a 1-D ``index``.
"""

from typing import Optional, Tuple

import torch

__all__ = [
    "scatter",
    "scatter_sum",
    "scatter_mean",
    "scatter_min",
    "TORCH_SCATTER_AVAILABLE",
]


def _broadcast(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand a 1-D ``index`` to ``src``'s shape, indexing along ``dim``."""
    if dim < 0:
        dim = src.dim() + dim
    out = index
    for _ in range(dim):
        out = out.unsqueeze(0)
    while out.dim() < src.dim():
        out = out.unsqueeze(-1)
    return out.expand_as(src)


def _output_size(src: torch.Tensor, index: torch.Tensor, dim: int,
                 dim_size: Optional[int]) -> list:
    if dim < 0:
        dim = src.dim() + dim
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    index = _broadcast(index, src, dim)
    if out is None:
        out = src.new_zeros(_output_size(src, index, dim, dim_size))
    return out.scatter_add_(dim, index, src)


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                  out: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None) -> torch.Tensor:
    total = _scatter_sum(src, index, dim, out, dim_size)
    if dim < 0:
        dim = src.dim() + dim
    counts = src.new_zeros(total.size(dim)).scatter_add_(
        0, index, torch.ones_like(index, dtype=src.dtype)
    )
    counts = counts.clamp(min=1)
    shape = [1] * total.dim()
    shape[dim] = -1
    return total / counts.view(shape)


def _scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Values-only equivalent of ``torch_scatter.scatter_min``.

    Returns ``(values, argmin)``. The second element is a placeholder of the
    right shape: ATOMICA only ever reads element ``[0]``, and computing a true
    argmin would cost a second pass.
    """
    if out is not None:
        raise NotImplementedError("scatter_min with a preallocated out= is not supported")
    size = _output_size(src, index, dim, dim_size)
    broadcast_index = _broadcast(index, src, dim)
    if src.is_floating_point():
        fill = torch.finfo(src.dtype).max
    else:
        fill = torch.iinfo(src.dtype).max
    values = src.new_full(size, fill)
    values = values.scatter_reduce_(dim if dim >= 0 else src.dim() + dim,
                                    broadcast_index, src,
                                    reduce="amin", include_self=True)
    # torch_scatter reports 0 for segments that no element mapped to.
    values = values.masked_fill(values == fill, 0)
    argmin = torch.zeros(size, dtype=torch.long, device=src.device)
    return values, argmin


def _scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
             out: Optional[torch.Tensor] = None,
             dim_size: Optional[int] = None,
             reduce: str = "sum") -> torch.Tensor:
    if reduce in ("sum", "add"):
        return _scatter_sum(src, index, dim, out, dim_size)
    if reduce == "mean":
        return _scatter_mean(src, index, dim, out, dim_size)
    if reduce == "min":
        return _scatter_min(src, index, dim, out, dim_size)[0]
    raise NotImplementedError(f"reduce='{reduce}' is not supported")


try:  # pragma: no cover - which branch runs depends on the installed extras
    from torch_scatter import scatter, scatter_mean, scatter_min, scatter_sum  # noqa: F401

    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    scatter = _scatter
    scatter_sum = _scatter_sum
    scatter_mean = _scatter_mean
    scatter_min = _scatter_min

    TORCH_SCATTER_AVAILABLE = False
