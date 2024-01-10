#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .prediction_model import PredictionModel


class GraphMultiBinaryClassifier(PredictionModel):
    def __init__(self, n_task, model_type, hidden_size, n_channel, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        self.n_task = n_task  # how many binary classification tasks?
        # disable energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False

        # binary classification head
        self.class_head = nn.Linear(hidden_size, n_task)

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, None, return_loss=False)
        pred_class = self.class_head(return_value.graph_repr)  # [bs, n_task]
        pred_class = torch.sigmoid(pred_class)
        return pred_class
