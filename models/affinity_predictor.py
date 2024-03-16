#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .prediction_model import PredictionModel, PredictionReturnValue


class AffinityPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.affinity_ffn = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 1, bias=False)
        # )
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        energy = return_value.energy
        return F.mse_loss(-energy, label)  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
        aff = scatter_sum(self.affinity_ffn(return_value.block_repr).squeeze(-1), return_value.batch_id)
        return F.mse_loss(aff, label)
    
    def infer(self, batch, extra_info=False, top_altered_edges=None, top_altered_edge_attr=None,
            bottom_altered_edges=None, bottom_altered_edge_attr=None):
        self.eval()
        return_value = super().forward(
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
            return -return_value.energy, return_value
        return -return_value.energy
        aff = scatter_sum(self.affinity_ffn(return_value.block_repr).squeeze(-1), return_value.batch_id)
        return aff