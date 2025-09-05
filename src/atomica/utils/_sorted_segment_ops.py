#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
    Reproducible segment operations (compared to scatter)
    However, the index must be sorted, which might not be satified in most scenarios
"""

from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import segment_csr, scatter_sum
from torch_scatter import segment_sum_csr, segment_mean_csr, segment_max_csr, segment_min_csr


@torch.no_grad()
def _index2indptr(index: Tensor) -> Tensor:
    # e.g. [0, 0, 1, 2, 2, 2, 2, 4, 4, 5]
    shift = F.pad(index[1:] - index[:-1], (1, 0), value=1) # [N], e.g. [1, 0, 1, 1, 0, 0, 0, 2, 0, 1]
    tmp_len = torch.nonzero(shift).flatten() # [tmp_L], e.g. [0, 2, 3, 7, 9], notice there should be another 7 between 3 and 7
    offset = shift[tmp_len] # [tmp_L], e.g. [1, 1, 1, 2, 1]
    L = torch.max(index) + 1 # e.g. 6
    indices = torch.zeros(L, dtype=torch.long, device=index.device)
    indices[offset.cumsum(dim=0) - 1] = 1  # e.g. [1, 1, 1, 0, 1, 1]
    indices = tmp_len.size(0) - (indices + indices.sum() - torch.cumsum(indices, dim=0))  # e.g. [0, 1, 2, 3, 3, 4]
    indptr = F.pad(tmp_len[indices], (0, 1), value=index.size(0))
    return indptr


def _sorted_segment_op(func: Any, src: Tensor, index: Tensor, dim: int=0, dim_size: Optional[int] = None) -> Tensor:
    indptr = _index2indptr(index)
    if dim_size:
        out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    else:
        out = None
    return func(src, indptr, out)


def sorted_segment_sum(src: Tensor, index: Tensor, dim: int=0, dim_size: Optional[int] = None) -> Tensor:
    return _sorted_segment_op(segment_sum_csr, src, index, dim, dim_size)


def sorted_segment_mean(src: Tensor, index: Tensor, dim: int=0, dim_size: Optional[int] = None) -> Tensor:
    return _sorted_segment_op(segment_mean_csr, src, index, dim, dim_size)


def sorted_segment_min(src: Tensor, index: Tensor, dim: int=0, dim_size: Optional[int] = None) -> Tensor:
    return _sorted_segment_op(segment_min_csr, src, index, dim, dim_size)[0]


def sorted_segment_max(src: Tensor, index: Tensor, dim: int=0, dim_size: Optional[int] = None) -> Tensor:
    return _sorted_segment_op(segment_max_csr, src, index, dim, dim_size)[0]


def sorted_segment_softmax(src: Tensor, index: Tensor, dim: int=0) -> Tensor:
    indptr = _index2indptr(index)
    max_value_per_index = segment_max_csr(src, indptr)[0]
    max_per_src_element = max_value_per_index[index]

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = segment_sum_csr(recentered_scores_exp, indptr)
    normalizing_constants = sum_per_index[index]

    return recentered_scores_exp.div(normalizing_constants)


def sorted_segment_std(src: Tensor, index: Tensor, dim: int=0, unbiased: bool=True) -> Tensor:
    indptr = _index2indptr(index)
    ones = torch.ones_like(index)
    count = segment_sum_csr(ones, indptr)

    tmp = segment_sum_csr(src, indptr)
    mean = tmp.div(count)

    var = (src - mean[index])
    var = var * var
    out = segment_sum_csr(var, indptr)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out




if __name__ == '__main__':
    src = torch.randn(10, 3)
    index = torch.tensor([0, 0, 1, 2, 2, 5, 7, 7, 9, 9])

    from torch_scatter import scatter_sum, scatter_mean, scatter_min, scatter_max, scatter_softmax, scatter_std

    pairs = {
        'sum': (scatter_sum, sorted_segment_sum),
        'mean': (scatter_mean, sorted_segment_mean),
        'min': (scatter_min, sorted_segment_min),
        'max': (scatter_max, sorted_segment_max),
        'softmax': (scatter_softmax, sorted_segment_softmax),
        'std': (scatter_std, sorted_segment_std)
    }

    for func in pairs:
        ref, imp = pairs[func]
        ref_val = ref(src, index, dim=0)
        if func in ['min', 'max']:
            ref_val = ref_val[0]
        imp_val = imp(src, index)

        error = torch.abs(ref_val - imp_val).sum()
        assert error < 1e-6, f'{func}: {error}'
        print(f'check for {func} passed, error: {error}')
