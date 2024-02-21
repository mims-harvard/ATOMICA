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
                 hierarchical=False, no_block_embedding=False, global_message_passing=False) -> None:
        super().__init__(
            model_type, hidden_size, n_channel, n_rbf, cutoff, n_head, radial_size, edge_size,
            k_neighbors, n_layers, dropout=dropout, std=std, atom_level=atom_level,
            hierarchical=hierarchical, no_block_embedding=no_block_embedding, global_message_passing=global_message_passing,
            denoising=False, atom_noise=False, translation_noise=False, rotation_noise=False)
        # del self.sigmas  # no need for noise level

    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')
        partial_finetune = kwargs.get('partial_finetune', False)
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
            global_message_passing=kwargs.get('global_message_passing', pretrained_model.global_message_passing),
        )
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)
            model.energy_ffn.requires_grad_(requires_grad=True) # only finetune the energy_ffn
        if pretrained_model.global_message_passing is False and model.global_message_passing is True:
            model.edge_embedding.requires_grad_(requires_grad=True)
            model.encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            model.top_encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        return model

    ########## overload ##########
    @torch.no_grad()
    def choose_receptor(self, batch_size, device):
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    @torch.no_grad()
    def perturb(self, Z, B, block_id, batch_id, bottom_batch_id, batch_size, segment_ids, receptor_segment):
        # do not perturb in prediction model
        return Z, None, None, None, None, None, None, None, None
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False, altered_edges=None, altered_edge_attr=None) -> ReturnValue:
        return_value = super().forward(
            Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label,
            return_noise=return_noise, return_loss=False, altered_edges=altered_edges, altered_edge_attr=altered_edge_attr)
        
        return return_value