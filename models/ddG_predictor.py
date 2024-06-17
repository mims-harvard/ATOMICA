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
        self.esm_projector1 = nn.Sequential(
            nn.Linear(2560, 2560),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2560, 2560),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2560, self.hidden_size),
        )
        # self.esm_projector2 = nn.Sequential(
        #     nn.Linear(2560, 2560),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(2560, 2560),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(2560, self.hidden_size),
        # )
        self.esm_and_block_projector1 = nn.Sequential(
            nn.Linear(2*self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2*self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2*self.hidden_size, self.hidden_size),
        )
        # self.esm_and_block_projector2 = nn.Sequential(
        #     nn.Linear(2*self.hidden_size, 2*self.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(2*self.hidden_size, 2*self.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(2*self.hidden_size, self.hidden_size),
        # )
            
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.ddg_ffn.requires_grad_(requires_grad=True)
        return model
    
    def get_pred(self, B, top_Z, esm_embeddings, lengths, segment_ids):
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

        # embedding
        top_H_0 = self.block_embedding.block_embedding(B)
        esm_embeddings_proj1 = self.esm_projector1(esm_embeddings)
        top_H_0 = self.esm_and_block_projector1(torch.cat([top_H_0, esm_embeddings_proj1], dim=1))
        perturb_block_mask = None
        
        #top level 
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
        global_mask = B != self.global_block_id if not self.global_message_passing else None
        block_repr, _ = self.top_encoder(top_H_0, top_Z, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)
        # esm_embeddings_proj2 = self.esm_projector2(esm_embeddings)
        # block_repr = self.esm_and_block_projector2(torch.cat([block_repr, esm_embeddings_proj2], dim=1))
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = F.normalize(graph_repr, dim=-1)
        # block_energy = self.energy_ffn(block_repr).squeeze(-1)
        # if not self.global_message_passing: # ignore global blocks
        #     block_energy[B == self.global_block_id] = 0
        # pred_energy = scatter_sum(block_energy, batch_id)
        
        num_items = graph_repr.shape[0]//2
        diff = graph_repr[:num_items] - graph_repr[num_items:]  # mt-wt, wt-mt
        forward_pred = self.ddg_ffn(diff).squeeze(dim=1)
        # reverse_pred = self.ddg_ffn(-diff).squeeze(dim=1)
        return forward_pred #, reverse_pred
    
    def forward(self, data, ddg) -> PredictionReturnValue:
        B = data['B']
        top_Z = data['Z_block']
        lengths = data['lengths']
        esm_embeddings = data['esm_embeddings']
        segment_ids = data['segment_ids']
        forward_pred = self.get_pred(B, top_Z, esm_embeddings, lengths, segment_ids)
        loss = F.mse_loss(forward_pred, ddg) # .mse_loss((forward_pred-reverse_pred)/2, ddg) + F.l1_loss(forward_pred, -reverse_pred)
        return loss, forward_pred # return only forward values
    
    def infer(self, batch):
        self.eval()
        data, _ = batch
        forward_pred = self.get_pred(
            B = data['B'],
            top_Z = data['Z_block'],
            esm_embeddings = data['esm_embeddings'],
            lengths = data['lengths'],
            segment_ids = data['segment_ids'],
        )
        return forward_pred