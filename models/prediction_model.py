#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum

from data.pdb_utils import VOCAB

from .pretrain_model import DenoisePretrainModel, ReturnValue

class PredictionModel(DenoisePretrainModel):
    def __init__(self, model_type, hidden_size, n_channel,
                 n_rbf=1, cutoff=7.0, n_head=1,
                 radial_size=16, edge_size=64, k_neighbors=9,
                 n_layers=3, dropout=0.1, std=10, atom_level=False,
                 hierarchical=False, no_block_embedding=False) -> None:
        super().__init__(
            model_type, hidden_size, n_channel, n_rbf, cutoff, n_head, radial_size, edge_size,
            k_neighbors, n_layers, dropout=dropout, std=std, atom_level=atom_level,
            hierarchical=hierarchical, no_block_embedding=no_block_embedding, denoising=False)
        # del self.sigmas  # no need for noise level

    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')
        model = cls(
            model_type=pretrained_model.model_type,
            hidden_size=pretrained_model.hidden_size,
            n_channel=pretrained_model.n_channel,
            n_rbf=pretrained_model.n_rbf,
            cutoff=pretrained_model.cutoff,
            radial_size=pretrained_model.radial_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=pretrained_model.k_neighbors,
            n_layers=pretrained_model.n_layers,
            n_head=pretrained_model.n_head,
            dropout=pretrained_model.dropout,
            std=pretrained_model.std,
            atom_level=pretrained_model.atom_level,
            hierarchical=pretrained_model.hierarchical,
            no_block_embedding=pretrained_model.no_block_embedding,
            **kwargs
        )
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        return model

    ########## overload ##########
    @torch.no_grad()
    def choose_receptor(self, batch_size, device):
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    @torch.no_grad()
    def perturb(self, Z, block_id, batch_id, batch_size, segment_ids, receptor_segment):
        # do not perturb in prediction model
        return Z, None, None, None, None
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        return_value = super().forward(
            Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label,
            return_noise=return_noise, return_loss=False)
        
        return return_value