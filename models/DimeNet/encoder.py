#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from utils.nn_utils import stable_norm

from .dimenet import DimeNet, DimeNetPlusPlus


class DimeNetEncoder(nn.Module):
    def __init__(self, hidden_size, n_layers=3) -> None:
        super().__init__()
        self.encoder = DimeNetPlusPlus(
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_blocks=n_layers,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=hidden_size * 2,
            num_spherical=7,
            num_radial=6
        )
        # self.encoder = DimeNet(hidden_size,
        #                        out_channels=hidden_size,
        #                        num_blocks=n_layers,
        #                        num_bilinear=8,
        #                        num_spherical=7,
        #                        num_radial=6)

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        with torch.no_grad():  # dimenet cannot handle edges with zero distance (e.g. self-loop)
            not_dist_zero = stable_norm(Z[edges[0]] - Z[edges[1]], dim=-1) > 1e-2
            edges = (edges[0][not_dist_zero], edges[1][not_dist_zero])
            del not_dist_zero
        block_repr = self.encoder(H, Z, batch_id, edges)  # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None