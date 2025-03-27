#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .atomica import InteractionModule
from .utils import batchify, unbatchify

class ATOMICAEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3, return_atom_noise=False, return_global_noise=False, 
                 return_torsion_noise=False, dropout=0.0, max_torsion_neighbors=9,
                 max_edge_length=20, max_global_edge_length=20, max_torsion_edge_length=5) -> None:
        super().__init__()
        self.encoder = InteractionModule(ns=hidden_size, nv=hidden_size//2, num_conv_layers=n_layers, sh_lmax=2, edge_size=edge_size, 
                                         return_atom_noise=return_atom_noise, return_global_noise=return_global_noise, 
                                         return_torsion_noise=return_torsion_noise, dropout=dropout, max_torsion_neighbors=max_torsion_neighbors,
                                         max_edge_length=max_edge_length, max_global_edge_length=max_global_edge_length, max_torsion_edge_length=max_torsion_edge_length)
        self.return_noise = any([return_atom_noise, return_global_noise, return_torsion_noise])

    def forward(self, H, Z, batch_id, perturb_mask, edges, edge_attr, tor_edges=None, tor_batch=None):
        if self.return_noise:
            output = self.encoder(H, Z, batch_id, perturb_mask, edges, edge_attr, tor_edges=tor_edges, tor_batch=tor_batch)  # [Nb, hidden]
            block_repr, trans_noise, rot_noise, atom_noise, tor_noise = output
        else:
            block_repr = self.encoder(H, Z, batch_id, perturb_mask, edges, edge_attr)  # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        if self.return_noise:
            return block_repr, trans_noise, rot_noise, atom_noise, tor_noise
        else:
            return block_repr


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.0, num_layers=4):
        super().__init__()
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.graph_repr_fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
    
    def forward(self, block_repr, batch_id):
        block_repr_, batchify_mask = batchify(block_repr, batch_id)
        for layer in range(self.num_layers):
            block_repr_attn, _ = self.attention_layers[layer](block_repr_, block_repr_, block_repr_)
            block_repr_attn = self.dropout(block_repr_attn)
            block_repr_ = block_repr_ + block_repr_attn # residual connection
            block_repr_ = self.norms[layer](block_repr_)

        block_repr = unbatchify(block_repr_, batchify_mask)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = self.graph_repr_fc(graph_repr)
        graph_repr = F.normalize(graph_repr, dim=-1)
        return graph_repr