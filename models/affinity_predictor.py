#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .prediction_model import PredictionModel, ReturnValue


class AffinityPredictor(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        # self.affinity_ffn = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 1, bias=False)
        # )
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)
        energy = return_value.energy
        print("Prediction:", -energy)
        print("Label:", label)
        return F.mse_loss(-energy, label)  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
        aff = scatter_sum(self.affinity_ffn(return_value.block_repr).squeeze(-1), return_value.batch_id)
        return F.mse_loss(aff, label)
    
    def infer(self, batch, return_block_energy=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=None
        )
        if return_block_energy:
            return -return_value.energy, -return_value.block_energy
        return -return_value.energy
        aff = scatter_sum(self.affinity_ffn(return_value.block_repr).squeeze(-1), return_value.batch_id)
        return aff