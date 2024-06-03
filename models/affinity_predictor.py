#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import torch

from .prediction_model import PredictionModel, PredictionReturnValue
from .pretrain_model import DenoisePretrainModel


class AffinityPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        nonlinearity = nn.ReLU
        self.energy_ffn = nn.Sequential(
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.energy_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        block_energy = self.energy_ffn(return_value.block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[B == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, return_value.batch_id)
        return F.mse_loss(-pred_energy, label), -pred_energy  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        block_energy = self.energy_ffn(return_value.block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[batch['B'] == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, return_value.batch_id)
        if extra_info:
            return -pred_energy, return_value
        return -pred_energy
    

class AffinityPredictorNoisyNodes(PredictionModel):

    def __init__(self, noisy_nodes_weight, **kwargs) -> None:
        DenoisePretrainModel.__init__(self, **kwargs)
        nonlinearity = nn.ReLU
        self.noisy_nodes_weight = noisy_nodes_weight
        self.energy_ffn = nn.Sequential(
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        model = cls(
            noisy_nodes_weight=kwargs['noisy_nodes_weight'],
            hidden_size=pretrained_model.hidden_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=kwargs.get('k_neighbors', pretrained_model.k_neighbors),
            n_layers=pretrained_model.n_layers,
            dropout=pretrained_model.dropout,
            fragmentation_method=pretrained_model.fragmentation_method if hasattr(pretrained_model, "fragmentation_method") else None, # for backward compatibility
            global_message_passing=kwargs.get('global_message_passing', pretrained_model.global_message_passing),
            atom_noise=pretrained_model.atom_noise, translation_noise=pretrained_model.translation_noise, rotation_noise=pretrained_model.rotation_noise, torsion_noise=pretrained_model.torsion_noise, 
            atom_weight=pretrained_model.atom_weight, translation_weight=pretrained_model.translation_weight, rotation_weight=pretrained_model.rotation_weight, torsion_weight=pretrained_model.torsion_weight
        )
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, global_message_passing={model.global_message_passing}, 
               fragmentation_method={model.fragmentation_method}""")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        if kwargs.get('partial_finetune', False):
            model.requires_grad_(requires_grad=False)
            model.energy_ffn.requires_grad_(requires_grad=True)

        if pretrained_model.global_message_passing is False and model.global_message_passing is True:
            model.edge_embedding.requires_grad_(requires_grad=True)
            model.encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            model.top_encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")

        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, receptor_segment, atom_score, atom_eps, tr_score, 
                tr_eps, rot_score, tor_edges, tor_score, tor_batch, label) -> PredictionReturnValue:
        return_value = DenoisePretrainModel.forward(self, Z, B, A, block_lengths, lengths, segment_ids, receptor_segment, atom_score, atom_eps, tr_score, 
                tr_eps, rot_score, tor_edges, tor_score, tor_batch) # use DenoisePretrainModel.forward()
        block_energy = self.energy_ffn(return_value.block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[B == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, return_value.batch_id)
        pred_loss = F.mse_loss(-pred_energy, label)
        return pred_loss, self.noisy_nodes_weight*return_value.loss, -pred_energy  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
    
    def _toggle_noise(self, return_noise: bool):
        self.encoder.return_noise = return_noise
        self.encoder.encoder.return_torsion_noise = return_noise
        self.encoder.encoder.return_global_noise = return_noise
        self.top_encoder.return_noise = return_noise
        self.top_encoder.encoder.return_global_noise = return_noise

    def infer(self, batch, extra_info=False):
        self.eval()
        self._toggle_noise(False)
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        ) # use PredictionModel.forward()
        block_energy = self.energy_ffn(return_value.block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[batch['B'] == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, return_value.batch_id)
        self._toggle_noise(True)
        if extra_info:
            return -pred_energy, return_value
        return -pred_energy