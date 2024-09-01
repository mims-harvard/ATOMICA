#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import torch
from copy import deepcopy

from .prediction_model import PredictionModel
from .pretrain_model import DenoisePretrainModel
from data.pdb_utils import VOCAB
from .InteractNN.utils import batchify, unbatchify


class BinaryPredictor(nn.Module):

    def __init__(self, atom_hidden_size, block_hidden_size, edge_size, k_neighbors,
                 n_layers, dropout=0.0, bottom_global_message_passing=False, global_message_passing=False, fragmentation_method=None,
                 num_heads=4, num_attn_layers=4) -> None:
        super().__init__()
        self.encoder0 = PredictionModel(atom_hidden_size=atom_hidden_size, block_hidden_size=block_hidden_size, edge_size=edge_size,
                                        k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, bottom_global_message_passing=bottom_global_message_passing, 
                                        global_message_passing=global_message_passing, fragmentation_method=fragmentation_method)
        self.encoder1 = PredictionModel(atom_hidden_size=atom_hidden_size, block_hidden_size=block_hidden_size, edge_size=edge_size,    
                                        k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, bottom_global_message_passing=bottom_global_message_passing, 
                                        global_message_passing=global_message_passing, fragmentation_method=fragmentation_method)
        self.num_attn_layers = num_attn_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.block_hidden_size = block_hidden_size
        self.attn_layers0 = nn.ModuleList([
            nn.MultiheadAttention(self.block_hidden_size, num_heads, dropout=self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.norm_layers0 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
            for _ in range(self.num_attn_layers)
        ])
        self.dropout_layers0 = nn.ModuleList([
            nn.Dropout(self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.attn_layers1 = nn.ModuleList([
            nn.MultiheadAttention(self.block_hidden_size, num_heads, dropout=self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.norm_layers1 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
            for _ in range(self.num_attn_layers)
        ])
        self.dropout_layers1 = nn.ModuleList([
            nn.Dropout(self.dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.pred_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.block_hidden_size*2, self.block_hidden_size*2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.block_hidden_size*2, self.block_hidden_size*2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.block_hidden_size*2, 1),
        )
        # self.pred_ffn = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.block_hidden_size, self.block_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.block_hidden_size, self.block_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.block_hidden_size, 1),
        # )

    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        model = cls(
            atom_hidden_size=pretrained_model.atom_hidden_size,
            block_hidden_size=pretrained_model.hidden_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=kwargs.get('k_neighbors', pretrained_model.k_neighbors),
            n_layers=pretrained_model.n_layers,
            dropout=kwargs.get('dropout', pretrained_model.dropout),
            fragmentation_method=pretrained_model.fragmentation_method if hasattr(pretrained_model, "fragmentation_method") else None, # for backward compatibility
            bottom_global_message_passing=kwargs.get('bottom_global_message_passing', pretrained_model.bottom_global_message_passing),
            global_message_passing=kwargs.get('global_message_passing', pretrained_model.global_message_passing),
        )
        print(f"""Pretrained model params: hidden_size={model.encoder0.hidden_size},
               edge_size={model.encoder0.edge_size}, k_neighbors={model.encoder0.k_neighbors}, 
               n_layers={model.encoder0.n_layers}, bottom_global_message_passing={model.encoder0.bottom_global_message_passing},
               global_message_passing={model.encoder0.global_message_passing}, 
               fragmentation_method={model.encoder0.fragmentation_method}""")
        assert not any([model.encoder0.atom_noise, model.encoder0.translation_noise, model.encoder0.rotation_noise, model.encoder0.torsion_noise]), "prediction model no noise"
        model.encoder0.load_state_dict(pretrained_model.state_dict(), strict=False)
        model.encoder1.load_state_dict(deepcopy(pretrained_model.state_dict()), strict=False)

        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)
            model.pred_ffn.requires_grad_(requires_grad=True)
            model.attn_layers0.requires_grad_(requires_grad=True)
            model.norm_layers0.requires_grad_(requires_grad=True)
            model.dropout_layers0.requires_grad_(requires_grad=True)
            model.attn_layers1.requires_grad_(requires_grad=True)
            model.norm_layers1.requires_grad_(requires_grad=True)
            model.dropout_layers1.requires_grad_(requires_grad=True)
        if pretrained_model.global_message_passing is False and model.encoder0.global_message_passing is True:
            model.encoder0.edge_embedding_top.requires_grad_(requires_grad=True)
            model.encoder1.edge_embedding_top.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        if pretrained_model.bottom_global_message_passing is False and model.encoder0.bottom_global_message_passing is True:
            model.encoder0.edge_embedding_bottom.requires_grad_(requires_grad=True)
            model.encoder1.edge_embedding_bottom.requires_grad_(requires_grad=True)
            print("Warning: bottom_global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        return model
    
    def forward_one_encoder(self, Z, B, A, block_lengths, lengths, segment_ids, encoder_id=0):
        if encoder_id == 0:
            encoder = self.encoder0
        else:
            encoder = self.encoder1

        return_value = encoder.forward(Z, B, A, block_lengths, lengths, segment_ids, return_graph_repr=False)
        block_repr = return_value.block_repr
        if not encoder.global_message_passing:
            block_repr[B == encoder.global_block_id, :] = 0
       
        return block_repr, return_value.batch_id
    
    def forward(self, Z0, B0, A0, block_lengths0, lengths0, segment_ids0, 
                Z1, B1, A1, block_lengths1, lengths1, segment_ids1, label):
        block_repr0, batch_id0 = self.forward_one_encoder(Z0, B0, A0, block_lengths0, lengths0, segment_ids0)
        block_repr1, batch_id1 = self.forward_one_encoder(Z1, B1, A1, block_lengths1, lengths1, segment_ids1)

        # apply multihead cross attention
        num_batches = len(label)
        max_seq_len0 = (batch_id0 == torch.arange(num_batches, device=batch_id0.device).unsqueeze(1)).sum(dim=1).max().item()
        max_seq_len1 = (batch_id1 == torch.arange(num_batches, device=batch_id1.device).unsqueeze(1)).sum(dim=1).max().item()
        max_seq_len = max(max_seq_len0, max_seq_len1)
        block_repr0, attn_batch0 = batchify(block_repr0, batch_id0, max_seq_len=max_seq_len) # (num_batches, max_seq_len, dim)
        block_repr0 = block_repr0.transpose(0, 1) # (max_seq_len, num_batches, dim)
        attn_block_repr0 = block_repr0.clone()
        block_repr1, attn_batch1 = batchify(block_repr1, batch_id1, max_seq_len=max_seq_len) # (num_batches, max_seq_len, dim)
        block_repr1 = block_repr1.transpose(0, 1) # (max_seq_len, num_batches, dim)
        attn_block_repr1 = block_repr1.clone()

        for i in range(self.num_attn_layers):
            attn_output, _ = self.attn_layers0[i](block_repr1, attn_block_repr0, attn_block_repr0) # Q, K, V
            attn_output = self.dropout_layers0[i](attn_output)
            attn_block_repr0 = self.norm_layers0[i](attn_block_repr0 + attn_output)
        attn_block_repr0 = attn_block_repr0.transpose(0, 1) # (num_batches, max_seq_len, dim)
        attn_block_repr0 = unbatchify(attn_block_repr0, attn_batch0) # (num_items, dim)

        for i in range(self.num_attn_layers):
            attn_output, _ = self.attn_layers1[i](block_repr0, attn_block_repr1, attn_block_repr1) # Q, K, V
            attn_output = self.dropout_layers1[i](attn_output)
            attn_block_repr1 = self.norm_layers1[i](attn_block_repr1 + attn_output)
        attn_block_repr1 = attn_block_repr1.transpose(0, 1) # (num_batches, max_seq_len, dim)
        attn_block_repr1 = unbatchify(attn_block_repr1, attn_batch1) # (num_items, dim)

        # predict the label
        graph_repr0 = scatter_sum(attn_block_repr0, batch_id0, dim=0)
        graph_repr1 = scatter_sum(attn_block_repr1, batch_id1, dim=0)
        graph_repr = torch.cat([graph_repr0, graph_repr1], dim=1)
        # graph_repr = graph_repr0 - graph_repr1
        pred = self.pred_ffn(graph_repr).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        return loss, pred 

    def infer(self, batch):
        self.eval()
        batch0, batch1, label = batch
        loss, pred = self.forward(
            Z0=batch0['X'], B0=batch0['B'], A0=batch0['A'],
            block_lengths0=batch0['block_lengths'],
            lengths0=batch0['lengths'],
            segment_ids0=batch0['segment_ids'],
            Z1=batch1['X'], B1=batch1['B'], A1=batch1['A'],
            block_lengths1=batch1['block_lengths'],
            lengths1=batch1['lengths'],
            segment_ids1=batch1['segment_ids'],
            label=label,
        )
        pred = torch.sigmoid(pred)
        return pred