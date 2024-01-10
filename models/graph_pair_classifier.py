#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .prediction_model import PredictionModel, ReturnValue


class GraphPairClassifier(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, num_class, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        # deactivate energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False
        self.class_ffn = nn.Linear(hidden_size * 2, num_class)
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)
        graph_repr = return_value.graph_repr  # [bs * 2], the pairs are adjacent
        graph1, graph2 = graph_repr[0::2], graph_repr[1::2]  # [bs, hidden_size]
        logits = self.class_ffn(torch.cat([graph1, graph2], dim=-1))  # [bs, num_class]
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
        graph_repr = return_value.graph_repr  # [bs * 2], the pairs are adjacent
        graph1, graph2 = graph_repr[0::2], graph_repr[1::2]  # [bs, hidden_size]
        logits = self.class_ffn(torch.cat([graph1, graph2], dim=-1))  # [bs, num_class]
        logits = F.softmax(logits, dim=-1)
        return logits.argmax(dim=-1), logits