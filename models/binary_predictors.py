#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import torch

from .prediction_model import PredictionModel, PredictionReturnValue
from data.pdb_utils import VOCAB
from .InteractNN.utils import batchify, unbatchify


class BinaryPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_attn_layers = 2
        num_heads = 4
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_size, num_heads, dropout=self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range(self.num_attn_layers)
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.pred_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size*2, 1),
        )

    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.pred_ffn.requires_grad_(requires_grad=True)
            model.attn_layers.requires_grad_(requires_grad=True)
            model.norm_layers.requires_grad_(requires_grad=True)
            model.dropout_layers.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids, return_graph_repr=False)
        block_repr = return_value.block_repr # reshape for inactive and active form
        if not self.global_message_passing: # ignore global blocks
            block_repr[B == self.global_block_id, :] = 0

        # apply multihead attention
        block_repr, attn_batch = batchify(block_repr, return_value.batch_id) # (num_batches, max_seq_len, dim)
        block_repr = block_repr.transpose(0, 1) # (max_seq_len, num_batches, dim)
        for i in range(self.num_attn_layers):
            attn_output, _ = self.attn_layers[i](block_repr, block_repr, block_repr)
            attn_output = self.dropout_layers[i](attn_output)
            block_repr = self.norm_layers[i](block_repr + attn_output)
        block_repr = block_repr.transpose(0, 1) # (num_batches, max_seq_len, dim)
        block_repr = unbatchify(block_repr, attn_batch) # (num_items, dim)

        # predict the label
        graph_repr = scatter_sum(block_repr, return_value.batch_id, dim=0)
        graph_repr = graph_repr.reshape(len(label), -1)
        pred = self.pred_ffn(graph_repr).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        return loss, pred 

    
    def infer(self, batch):
        self.eval()
        loss, pred = self.forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
        )
        pred = torch.sigmoid(pred)
        return pred