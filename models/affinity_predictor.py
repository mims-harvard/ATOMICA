#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .prediction_model import PredictionModel, PredictionReturnValue


class AffinityPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        energy = return_value.energy
        return F.mse_loss(-energy, label)  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        if extra_info:
            return -return_value.energy, return_value
        return -return_value.energy