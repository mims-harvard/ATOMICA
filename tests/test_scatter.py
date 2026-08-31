"""Check the pure-PyTorch scatter helpers against torch_scatter.

ATOMICA uses torch_scatter when it is installed and falls back to
``atomica.utils.scatter``'s own implementations otherwise. These tests run the
two against each other on the argument forms the model uses, so the fallback
cannot silently drift. When torch_scatter is absent the comparisons are skipped
and only the shape/value invariants are checked.
"""

import pytest
import torch

from atomica.utils.scatter import (
    TORCH_SCATTER_AVAILABLE,
    _scatter,
    _scatter_mean,
    _scatter_min,
    _scatter_sum,
)

needs_torch_scatter = pytest.mark.skipif(
    not TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not installed"
)

SHAPES = [
    ((40,), 7),          # 1-D, as used for per-graph lengths and losses
    ((40, 5), 7),        # 2-D, as used for block/graph representations
    ((40, 3, 3), 7),     # 3-D, as used for [N, n_channel, 3] coordinates
]


def _inputs(shape, n_seg, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(*shape, generator=g)
    index = torch.randint(0, n_seg, (shape[0],), generator=g)
    return src, index


@needs_torch_scatter
@pytest.mark.parametrize("shape,n_seg", SHAPES)
def test_sum_matches_torch_scatter(shape, n_seg):
    from torch_scatter import scatter_sum as ref

    src, index = _inputs(shape, n_seg)
    assert torch.allclose(_scatter_sum(src, index, dim=0), ref(src, index, dim=0), atol=1e-6)


@needs_torch_scatter
@pytest.mark.parametrize("shape,n_seg", SHAPES)
def test_mean_matches_torch_scatter(shape, n_seg):
    from torch_scatter import scatter_mean as ref

    src, index = _inputs(shape, n_seg)
    assert torch.allclose(_scatter_mean(src, index, dim=0), ref(src, index, dim=0), atol=1e-6)


@needs_torch_scatter
def test_min_matches_torch_scatter():
    from torch_scatter import scatter_min as ref

    src, index = _inputs((40,), 7)
    assert torch.allclose(_scatter_min(src, index)[0], ref(src, index)[0], atol=1e-6)


@needs_torch_scatter
def test_min_zero_fills_empty_segments_like_torch_scatter():
    from torch_scatter import scatter_min as ref

    src = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 0, 2])  # segment 1 receives nothing
    assert torch.allclose(_scatter_min(src, index, dim_size=4)[0],
                          ref(src, index, dim_size=4)[0], atol=1e-6)


@needs_torch_scatter
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_generic_scatter_matches_torch_scatter(reduce):
    from torch_scatter import scatter as ref

    src, index = _inputs((40, 5), 7)
    got = _scatter(src, index, dim=0, dim_size=9, reduce=reduce)
    want = ref(src, index, dim=0, dim_size=9, reduce=reduce)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-6)


@needs_torch_scatter
def test_default_dim_size_matches_torch_scatter():
    """torch_scatter infers dim_size as index.max() + 1; so must the fallback."""
    from torch_scatter import scatter_sum as ref

    src = torch.randn(10, 4)
    index = torch.tensor([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
    assert _scatter_sum(src, index, dim=0).shape == ref(src, index, dim=0).shape


def test_sum_is_correct_without_reference():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
    assert torch.allclose(_scatter_sum(src, index, dim=0), expected)


def test_mean_is_correct_without_reference():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])
    expected = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
    assert torch.allclose(_scatter_mean(src, index, dim=0), expected)


def test_dim_size_pads_trailing_empty_segments():
    src = torch.ones(3, 2)
    index = torch.tensor([0, 0, 1])
    out = _scatter_sum(src, index, dim=0, dim_size=5)
    assert out.shape == (5, 2)
    assert torch.allclose(out[2:], torch.zeros(3, 2))
