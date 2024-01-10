#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

from .prediction_model import PredictionModel, ReturnValue


class GraphClassifier(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, num_class, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        # deactivate energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False
        self.class_ffn = nn.Linear(hidden_size, num_class)
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)
        graph_repr = return_value.graph_repr
        logits = self.class_ffn(graph_repr)  # [bs, num_class]
        return F.cross_entropy(logits, label)
    
    def infer(self, batch):
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=None
        )
        logits = self.class_ffn(return_value.graph_repr)
        logits = F.softmax(logits, dim=-1)
        return logits.argmax(dim=-1), logits