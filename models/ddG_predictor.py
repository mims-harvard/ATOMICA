#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .prediction_model import PredictionModel, ReturnValue


class DDGPredictor(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        self.ddg_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, wt, mt, ddg, return_noise=False) -> ReturnValue:
        wt_return_value = super().forward(Z=wt['X'], B=wt['B'], A=wt['A'],
            atom_positions=wt['atom_positions'],
            block_lengths=wt['block_lengths'],
            lengths=wt['lengths'],
            segment_ids=wt['segment_ids'],
            label=None, return_noise=return_noise)

        mt_return_value = super().forward(Z=mt['X'], B=mt['B'], A=mt['A'],
            atom_positions=mt['atom_positions'],
            block_lengths=mt['block_lengths'],
            lengths=mt['lengths'],
            segment_ids=mt['segment_ids'],
            label=None, return_noise=return_noise)
        
        diff = mt_return_value.graph_repr - wt_return_value.graph_repr 
        pred = self.ddg_ffn(diff).squeeze()
        ddg = (ddg > 0).float()
        assert pred.shape == ddg.shape
        loss = F.binary_cross_entropy(pred, ddg)
        return loss
    
    def infer(self, batch):
        self.eval()
        wt, mt, _ = batch

        wt_return_value = super().forward(Z=wt['X'], B=wt['B'], A=wt['A'],
            atom_positions=wt['atom_positions'],
            block_lengths=wt['block_lengths'],
            lengths=wt['lengths'],
            segment_ids=wt['segment_ids'],
            label=None)

        mt_return_value = super().forward(Z=mt['X'], B=mt['B'], A=mt['A'],
            atom_positions=mt['atom_positions'],
            block_lengths=mt['block_lengths'],
            lengths=mt['lengths'],
            segment_ids=mt['segment_ids'],
            label=None)
        
        diff = mt_return_value.graph_repr - wt_return_value.graph_repr 
        pred = self.ddg_ffn(diff).squeeze()
        return pred