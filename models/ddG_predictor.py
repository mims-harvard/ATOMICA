#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import torch

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
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1, bias=False),
        )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        model.ddg_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, data, ddg) -> PredictionReturnValue:
        return_value = super().forward(
            Z=data['X'], B=data['B'], A=data['A'],
            block_lengths=data['block_lengths'],
            lengths=data['lengths'],
            segment_ids=data['segment_ids'],
        )
        # num_items = return_value.graph_repr.shape[0]//2
        # diff = return_value.graph_repr[:num_items] - return_value.graph_repr[num_items:]
        # pred = self.ddg_ffn(diff).squeeze(dim=1)
        pred = self.ddg_ffn(return_value.graph_repr).squeeze(dim=1)
        assert pred.shape == ddg.shape
        loss = F.mse_loss(pred, ddg)
        return loss
    
    def infer(self, batch):
        self.eval()
        data, _ = batch
        return_value = super().forward(
            Z=data['X'], B=data['B'], A=data['A'],
            block_lengths=data['block_lengths'],
            lengths=data['lengths'],
            segment_ids=data['segment_ids'],
        )
        num_items = return_value.graph_repr.shape[0]//2
        diff = return_value.graph_repr[:num_items] - return_value.graph_repr[num_items:]
        pred = self.ddg_ffn(diff).squeeze(dim=1)
        return pred