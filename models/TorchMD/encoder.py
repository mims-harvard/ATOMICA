#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from utils.nn_utils import stable_norm

from .torchmd_et import TorchMD_ET


class TorchMDEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3) -> None:
        super().__init__()

        self.num_rbf = 50

        self.encoder = TorchMD_ET(
            hidden_channels=hidden_size,
            num_layers=n_layers,
            num_rbf=self.num_rbf
        )

        if edge_size != 0:
            self.edge_linear = nn.Linear(edge_size, self.num_rbf)

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        with torch.no_grad():  # dimenet cannot handle edges with zero distance (e.g. self-loop)
            not_dist_zero = stable_norm(Z[edges[0]] - Z[edges[1]], dim=-1) > 1e-2
            edges = (edges.T[not_dist_zero]).T
            if edge_attr is not None:
                edge_attr = edge_attr[not_dist_zero]
            del not_dist_zero
        if edge_attr is not None:
            edge_attr = self.edge_linear(edge_attr)
        block_repr, _ = self.encoder(H, Z, batch_id, edges, edge_attr)  # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None