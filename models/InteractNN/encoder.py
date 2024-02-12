#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from .interactnn import InteractionModule

class InteractNNEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3, return_noise=False, global_message_passing=False) -> None:
        super().__init__()
        self.encoder = InteractionModule(ns=hidden_size, nv=hidden_size//2, num_conv_layers=n_layers, sh_lmax=2, edge_size=edge_size, return_noise=return_noise)
        self.return_noise = return_noise
        self.global_message_passing = global_message_passing

    def forward(self, H, Z, block_id, batch_id, perturb_mask, edges, edge_attr, global_mask=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        if self.return_noise:
            block_repr, trans_noise, rot_noise, atom_noise = self.encoder(H, Z, batch_id, perturb_mask, edges, edge_attr)  # [Nb, hidden]
        else:
            block_repr = self.encoder(H, Z, batch_id, perturb_mask, edges, edge_attr)  # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        if global_mask is not None: # filter out global nodes if no global message passing
            assert self.global_message_passing == False, "global_message_passing is True so no global_mask should be provided"
            graph_repr = scatter_sum(block_repr[global_mask], batch_id[global_mask], dim=0)
        else:
            assert self.global_message_passing == True, "global_message_passing is False so a global_mask should be provided"
            graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)

        if self.return_noise:
            return H, block_repr, graph_repr, None, trans_noise, rot_noise, atom_noise
        else:
            return H, block_repr, graph_repr, None