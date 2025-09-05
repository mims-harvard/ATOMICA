#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from .modules.tools import KNNBatchEdgeConstructor, BlockEmbedding

from .modules.get import GET

'''
Masked 1D & 3D language model
Add noise to ground truth 3D coordination
Add mask to 1D sequence
'''
class GETEncoder(nn.Module):
    def __init__(self, hidden_size, n_channel, radial_size=16,
                 edge_size=16, k_neighbors=9, n_layers=3, dropout=0.1) -> None:
        super().__init__()

        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            num_atom_position=VOCAB.get_num_atom_pos(),
            embed_size=hidden_size
        )

        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=k_neighbors, delete_self_loop=False)
        self.edge_embedding = nn.Embedding(2, edge_size)  # [0 for internal context edges, 1 for interacting edges]
        
        self.encoder = GET(
            hidden_size, radial_size, n_channel,
            edge_size, n_layers, dropout=dropout
        )

        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.out_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )

    def message_passing(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids):
        # batch_id and block_id
        with torch.no_grad():

            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

        H_0 = self.block_embedding(B, A, atom_positions, block_id)  # use segment ids as position encoding
        intra_edges, inter_edges, _, _, _ = self.edge_constructor(B, batch_id, segment_ids, X=Z, block_id=block_id)
        
        edges = torch.cat([intra_edges, inter_edges], dim=1)
        edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        edge_attr = self.edge_embedding(edge_attr)

        H, pred_X = self.encoder(H_0, Z, block_id, edges, edge_attr)

        H = self.out_linear(H)
        
        H = H + (pred_X.mean(-1).mean(-1) * 0).unsqueeze(-1)  # cheat the autograd check

        H_block = scatter_mean(H, block_id, dim=0)
        pred_results = scatter_mean(self.out_ffn(H_block).squeeze(-1), batch_id)

        return pred_results, H_block  # [batch_size], [N, hidden_size]
    

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label):
        pred_results, _ = self.message_passing(Z, B, A, atom_positions, block_lengths, lengths, segment_ids)
        return F.mse_loss(pred_results, label)
    
    def infer(self, batch):
        pred_results, _ = self.message_passing(
                Z=batch['X'], B=batch['B'], A=batch['A'],
                atom_positions=batch['atom_positions'],
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids']
        )
        return pred_results