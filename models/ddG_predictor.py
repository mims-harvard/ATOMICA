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
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)
            model.ddg_ffn.requires_grad_(requires_grad=True)
        return model
    
    def get_pred(self, B, top_Z, lengths, segment_ids):
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

        # embedding
        top_H_0 = self.block_embedding.block_embedding(B)
        perturb_block_mask = None
        
        #top level 
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
        global_mask = B != self.global_block_id if not self.global_message_passing else None
        block_repr, graph_repr = self.top_encoder(top_H_0, top_Z, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)

        # block_energy = self.energy_ffn(block_repr).squeeze(-1)
        # if not self.global_message_passing: # ignore global blocks
        #     block_energy[B == self.global_block_id] = 0
        # pred_energy = scatter_sum(block_energy, batch_id)
        
        num_items = graph_repr.shape[0]//2
        diff = graph_repr[:num_items] - graph_repr[num_items:]
        pred = self.ddg_ffn(diff).squeeze(dim=1)
        return pred
    
    def forward(self, data, ddg) -> PredictionReturnValue:
        B = data['B']
        top_Z = data['Z_block']
        lengths = data['lengths']
        segment_ids = data['segment_ids']
        pred = self.get_pred(B, top_Z, lengths, segment_ids)
        assert pred.shape == ddg.shape
        loss = F.mse_loss(pred, ddg)
        return loss
    
    def infer(self, batch):
        self.eval()
        data, _ = batch
        pred = self.get_pred(
            B = data['B'],
            top_Z = data['Z_block'],
            lengths = data['lengths'],
            segment_ids = data['segment_ids'],
        )
        return pred