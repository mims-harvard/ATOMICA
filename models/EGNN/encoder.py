#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from .egnn import EGNN

class EGNNEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3) -> None:
        super().__init__()

        self.encoder = EGNN(
            in_node_nf=hidden_size,
            hidden_nf=hidden_size,
            out_node_nf=hidden_size,
            in_edge_nf=edge_size,
            n_layers=n_layers)

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        block_repr, _ = self.encoder(H, Z, edges, edge_attr)  # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None