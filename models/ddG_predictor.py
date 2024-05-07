#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .prediction_model import PredictionModel, PredictionReturnValue


class DDGPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ddg_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1, bias=False),
            nn.Sigmoid()
        )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        model.ddg_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, wt, mt, ddg) -> PredictionReturnValue:
        wt_return_value = super().forward(
            Z=wt['X'], B=wt['B'], A=wt['A'],
            block_lengths=wt['block_lengths'],
            lengths=wt['lengths'],
            segment_ids=wt['segment_ids'],
        )

        mt_return_value = super().forward(
            Z=mt['X'], B=mt['B'], A=mt['A'],
            block_lengths=mt['block_lengths'],
            lengths=mt['lengths'],
            segment_ids=mt['segment_ids'],
        )
        
        diff = mt_return_value.graph_repr - wt_return_value.graph_repr 
        pred = self.ddg_ffn(diff).squeeze()
        ddg = (ddg > 0).float()
        assert pred.shape == ddg.shape
        loss = F.binary_cross_entropy(pred, ddg)
        return loss
    
    def infer(self, batch):
        self.eval()
        wt, mt, _ = batch

        wt_return_value = super().forward(
            Z=wt['X'], B=wt['B'], A=wt['A'],
            block_lengths=wt['block_lengths'],
            lengths=wt['lengths'],
            segment_ids=wt['segment_ids'],
        )

        mt_return_value = super().forward(
            Z=mt['X'], B=mt['B'], A=mt['A'],
            block_lengths=mt['block_lengths'],
            lengths=mt['lengths'],
            segment_ids=mt['segment_ids'],
        )
        
        diff = mt_return_value.graph_repr - wt_return_value.graph_repr 
        pred = self.ddg_ffn(diff).squeeze()
        return pred