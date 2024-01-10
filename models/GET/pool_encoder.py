#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from .modules.get import GET


class GETPoolEncoder(nn.Module):
    def __init__(self, hidden_size, radial_size, n_channel,
                 n_rbf=1, cutoff=7.0, edge_size=16, n_layers=3,
                 n_head=1, dropout=0.1,
                 z_requires_grad=True, stable=False) -> None:
        super().__init__()

        self.encoder = GET(
            hidden_size, radial_size, n_channel,
            n_rbf, cutoff, edge_size, n_layers,
            n_head, dropout=dropout,
            z_requires_grad=z_requires_grad
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        block_id = torch.arange(0, batch_id.shape[0], device=H.device)
        block_repr, pred_Z = self.encoder(H, Z, block_id, batch_id, edges, edge_attr)
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None