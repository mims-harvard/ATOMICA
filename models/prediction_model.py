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

from .pretrain_model import DenoisePretrainModel

PredictionReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 'block_energy',
     'unit_repr', 'block_repr', 'graph_repr', 'graph_unit_repr', 'batch_id', 'block_id'],
)

class PredictionModel(DenoisePretrainModel):
    def __init__(self, hidden_size, edge_size, k_neighbors,
                 n_layers, dropout=0.1, global_message_passing=False, fragmentation_method=None) -> None:
        super().__init__(
            hidden_size=hidden_size, edge_size=edge_size, 
            k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, 
            global_message_passing=global_message_passing,
            atom_noise=False, translation_noise=False, rotation_noise=False, 
            torsion_noise=False, fragmentation_method=fragmentation_method)
        self.energy_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        assert not any([self.atom_noise, self.translation_noise, self.rotation_noise, self.torsion_noise]), 'Prediction model should not have any denoising heads'

    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')
        partial_finetune = kwargs.get('partial_finetune', False)
        model = cls(
            hidden_size=pretrained_model.hidden_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=pretrained_model.k_neighbors,
            n_layers=pretrained_model.n_layers,
            dropout=pretrained_model.dropout,
            fragmentation_method=pretrained_model.fragmentation_method if hasattr(pretrained_model, "fragmentation_method") else None, # for backward compatibility
            global_message_passing=kwargs.get('global_message_passing', pretrained_model.global_message_passing),
        )
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, global_message_passing={model.global_message_passing}, 
               fragmentation_method={model.fragmentation_method}""")
        assert not any([model.atom_noise, model.translation_noise, model.rotation_noise, model.torsion_noise]), "prediction model no noise"
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

    def precalculate_edges(self, batch):
        """
        Returns block level edges and edge_attr if top_level = True
        Returns atom level edges and edge_attr if top_level = False
        """
        Z=batch['X']
        B=batch['B']
        A=batch['A']
        block_lengths=batch['block_lengths']
        lengths=batch['lengths']
        segment_ids=batch['segment_ids']
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            # transform blocks to single units
            bottom_batch_id = batch_id[block_id]  # [Nu]
            bottom_B = B[block_id]  # [Nu]
            bottom_segment_ids = segment_ids[block_id]  # [Nu]
            bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]
            
            top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
            top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)

            top_edges, top_edge_attr, top_edge_mask = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, return_mask=True)
            bottom_edges, bottom_edge_attr, bottom_edge_mask = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z, bottom_block_id, return_mask=True)
        return bottom_edges, bottom_edge_attr, bottom_edge_mask, top_edges, top_edge_attr, top_edge_mask

    ########## overload ##########
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, 
                top_altered_edges=None, top_altered_edge_attr=None,
                bottom_altered_edges=None, bottom_altered_edge_attr=None,) -> PredictionReturnValue:
        # batch_id and block_id
        with torch.no_grad():

            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            # transform blocks to single units
            bottom_batch_id = batch_id[block_id]  # [Nu]
            bottom_B = B[block_id]  # [Nu]
            bottom_segment_ids = segment_ids[block_id]  # [Nu]
            bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]

        # embedding
        bottom_H_0 = self.block_embedding.atom_embedding(A) # FIXME: ablation
        top_H_0 = self.block_embedding.block_embedding(B)

        perturb_mask = None
        perturb_block_mask = None
        # bottom level message passing
        if bottom_altered_edges is not None:
            edges = bottom_altered_edges
            edge_attr = bottom_altered_edge_attr
        else:
            edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z, bottom_block_id)
        atom_mask = A != VOCAB.get_atom_global_idx() if not self.global_message_passing else None
        bottom_block_repr, graph_repr_bottom = self.encoder(bottom_H_0, Z, bottom_batch_id, perturb_mask, edges, edge_attr, global_mask=atom_mask)
        # top level 
        top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        if top_altered_edges is not None:
            edges = top_altered_edges
            edge_attr = top_altered_edge_attr
        else:
            edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
        top_H_0 = top_H_0 + scatter_mean(bottom_block_repr, block_id, dim=0)
        global_mask = B != self.global_block_id if not self.global_message_passing else None
        block_repr, graph_repr = self.top_encoder(top_H_0, top_Z, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)
        bottom_block_repr = torch.concat([bottom_block_repr, block_repr[block_id]], dim=-1) # bottom_block_repr and block_repr may have different dim size for dim=1

        block_energy = self.energy_ffn(block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[B == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, batch_id)
        
        return PredictionReturnValue(
            energy=pred_energy,
            block_energy=block_energy,

            # representations
            unit_repr=bottom_block_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,
            graph_unit_repr=graph_repr_bottom,

            # batch information
            batch_id=batch_id,
            block_id=block_id,
        )


    def infer(self, batch, extra_info=False, top_altered_edges=None, top_altered_edge_attr=None,
                bottom_altered_edges=None, bottom_altered_edge_attr=None):
        self.eval()
        return_value = self.forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            top_altered_edges=top_altered_edges,
            top_altered_edge_attr=top_altered_edge_attr,
            bottom_altered_edges=bottom_altered_edges,
            bottom_altered_edge_attr=bottom_altered_edge_attr,
        )
        if extra_info:
            return -return_value.graph_repr, return_value
        return -return_value.graph_repr