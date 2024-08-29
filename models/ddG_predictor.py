#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import torch

from .prediction_model import PredictionModel, PredictionReturnValue
from data.pdb_utils import VOCAB
from .InteractNN.utils import batchify, unbatchify

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
            nn.Linear(self.hidden_size, 1),
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
    
    def get_pred(self, B, top_Z, esm_embeddings, lengths, segment_ids, mt_block_indexes):
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

        # embedding
        top_H_0 = self.block_embedding.block_embedding(B)
        esm_embeddings_proj1 = self.esm_projector1(esm_embeddings)
        top_H_0 = self.esm_and_block_projector1(torch.cat([top_H_0, esm_embeddings_proj1], dim=1))
        
        #top level 
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          self.global_message_passing, top=True)
        block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)
        num_items = block_repr.shape[0]//2
        diff = block_repr[:num_items] - block_repr[num_items:]  # mt-wt, wt-mt

        # mut_batch_id = batch_id[mt_block_indexes]
        # mut_block_repr = diff[mt_block_indexes]
        # forward_pred = self.ddg_ffn(mut_block_repr)
        # output = scatter_sum(forward_pred, mut_batch_id, dim=0).squeeze(dim=1)

        forward_pred = self.ddg_ffn(diff).squeeze(dim=1)
        forward_output = scatter_sum(forward_pred, batch_id[:num_items], dim=0)

        # reverse_pred = self.ddg_ffn(-diff).squeeze(dim=1)
        # reverse_output = scatter_sum(reverse_pred, batch_id[:num_items], dim=0)
        
        return forward_output #, reverse_output

        # # esm_embeddings_proj2 = self.esm_projector2(esm_embeddings)
        # # block_repr = self.esm_and_block_projector2(torch.cat([block_repr, esm_embeddings_proj2], dim=1))
        # graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        # graph_repr = F.normalize(graph_repr, dim=-1)
        # # block_energy = self.energy_ffn(block_repr).squeeze(-1)
        # # if not self.global_message_passing: # ignore global blocks
        # #     block_energy[B == self.global_block_id] = 0
        # # pred_energy = scatter_sum(block_energy, batch_id)

        num_items = graph_repr.shape[0]//2
        diff = graph_repr[:num_items] - graph_repr[num_items:]  # mt-wt, wt-mt
        forward_pred = self.ddg_ffn(diff).squeeze(dim=1)
        # reverse_pred = self.ddg_ffn(-diff).squeeze(dim=1)
        return forward_pred #, reverse_pred
    
    def forward(self, data, ddg, mt_block_indexes) -> PredictionReturnValue:
        B = data['B']
        top_Z = data['Z_block']
        lengths = data['lengths']
        esm_embeddings = data['esm_embeddings']
        segment_ids = data['segment_ids']
        forward_pred = self.get_pred(B, top_Z, esm_embeddings, lengths, segment_ids, mt_block_indexes)
        loss = F.mse_loss(forward_pred, ddg) #+ F.mse_loss(-reverse_pred, ddg) + F.mse_loss(forward_pred, -reverse_pred) # .mse_loss((forward_pred-reverse_pred)/2, ddg) + F.l1_loss(forward_pred, -reverse_pred)
        return loss, forward_pred # return only forward values
    
    def infer(self, batch):
        self.eval()
        data, _, mt_block_indexes = batch
        forward_pred = self.get_pred(
            B = data['B'],
            top_Z = data['Z_block'],
            esm_embeddings = data['esm_embeddings'],
            lengths = data['lengths'],
            segment_ids = data['segment_ids'],
            mt_block_indexes = mt_block_indexes,
        )
        return forward_pred

class GLOFPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_attn_layers = 8
        num_heads = 4
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_size*2, num_heads, dropout=self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.hidden_size*2)
            for _ in range(self.num_attn_layers)
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.glof_ffn = nn.Sequential(
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
            model.glof_ffn.requires_grad_(requires_grad=True)
        return model
    
    def get_pred(self, B, top_Z, lengths, segment_ids, mt_block_indexes):
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

        top_H_0 = self.block_embedding.block_embedding(B)
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          self.global_message_passing, top=True)
        block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)
        num_items = block_repr.shape[0]//2

        # concat the wt and mt block representations
        wt_mt_block_repr = torch.cat([block_repr[:num_items], block_repr[num_items:]], dim=1)

        # apply multihead attention
        wt_mt_block_repr, batch_mask = batchify(wt_mt_block_repr, batch_id[:num_items]) # (num_batches, max_seq_len, dim)
        wt_mt_block_repr = wt_mt_block_repr.transpose(0, 1) # (max_seq_len, num_batches, dim)
        for i in range(self.num_attn_layers):
            attn_output, _ = self.attn_layers[i](wt_mt_block_repr, wt_mt_block_repr, wt_mt_block_repr)
            attn_output = self.dropout_layers[i](attn_output)
            wt_mt_block_repr = self.norm_layers[i](wt_mt_block_repr + attn_output)
        wt_mt_block_repr = wt_mt_block_repr.transpose(0, 1) # (num_batches, max_seq_len, dim)
        wt_mt_block_repr = unbatchify(wt_mt_block_repr, batch_mask) # (num_items, dim)

        # predict the label
        pred = self.glof_ffn(wt_mt_block_repr[mt_block_indexes]).squeeze(dim=1)
        output = scatter_sum(pred, batch_id[:num_items][mt_block_indexes], dim=0)
        return output 

    
    def forward(self, data, labels, mt_block_indexes) -> PredictionReturnValue:
        B = data['B']
        top_Z = data['Z_block']
        lengths = data['lengths']
        segment_ids = data['segment_ids']
        forward_pred = self.get_pred(B, top_Z, lengths, segment_ids, mt_block_indexes)
        loss = F.binary_cross_entropy_with_logits(forward_pred, labels)
        return loss, forward_pred
    
    def infer(self, batch):
        self.eval()
        data, _, mt_block_indexes = batch
        forward_pred = self.get_pred(
            B = data['B'],
            top_Z = data['Z_block'],
            lengths = data['lengths'],
            segment_ids = data['segment_ids'],
            mt_block_indexes = mt_block_indexes,
        )
        forward_pred = torch.sigmoid(forward_pred)
        return forward_pred