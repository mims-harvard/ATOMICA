#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
    Reproducible unsorted segment operations (slower)
"""
from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def unsorted_segment_sum(src: Tensor, index: Tensor, dim: int=0, dim_size: Optional[int] = None) -> Tensor:
    if dim_size is None:
        dim_size = index.max() + 1
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    index = _broadcast(index, src, dim)
    out.scatter_add_(0, index, src)
    return out


def unsorted_segment_mean(src: Tensor, index: Tensor, dim: int=0, dim_size: Optional[int] = None) -> Tensor:
    total = unsorted_segment_sum(src, index, dim, dim_size)
    ones = torch.ones_like(index)
    cnt = unsorted_segment_sum(ones, index, dim, dim_size)
    cnt = _broadcast(cnt, total, dim)
    return total / cnt.clamp(min=1)


def unsorted_segment_std(src: Tensor, index: Tensor, dim: int=0, unbiased: bool=True) -> Tensor:
    ones = torch.ones_like(index)
    count = unsorted_segment_sum(ones, index, dim)

    tmp = unsorted_segment_sum(src, index, dim)
    count = _broadcast(count, tmp, dim)
    mean = tmp.div(count)

    var = (src - mean[index])
    var = var * var
    out = unsorted_segment_sum(var, index, dim)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out




if __name__ == '__main__':
    src = torch.randn(10, 3)
    index = torch.tensor([9, 5, 1, 9, 4, 2, 2, 1, 5, 6])

    from torch_scatter import scatter_sum, scatter_mean, scatter_min, scatter_max, scatter_softmax, scatter_std

    pairs = {
        'sum': (scatter_sum, unsorted_segment_sum),
        'mean': (scatter_mean, unsorted_segment_mean),
        'std': (scatter_std, unsorted_segment_std)
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
