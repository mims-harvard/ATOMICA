#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import torch
from copy import deepcopy

from .prediction_model import PredictionModel
from .pretrain_model import DenoisePretrainModel
from data.pdb_utils import VOCAB
from .InteractNN.utils import batchify, unbatchify
from .tools import CrossAttentionWithSpatialEncoding


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
    
    def forward_one_encoder(self, Z, B, A, block_lengths, lengths, segment_ids, encoder_id):
        if encoder_id == 0:
            encoder = self.encoder0
        elif encoder_id == 1:
            encoder = self.encoder1
        else:
            raise ValueError(f"Invalid encoder_id: {encoder_id}")

        return_value = encoder.forward(Z, B, A, block_lengths, lengths, segment_ids, return_graph_repr=False)
        block_repr = return_value.block_repr
        if not encoder.global_message_passing:
            block_repr[B == encoder.global_block_id, :] = 0
       
        return block_repr, return_value.batch_id
    
    def forward(self, Z0, B0, A0, block_lengths0, lengths0, segment_ids0, 
                Z1, B1, A1, block_lengths1, lengths1, segment_ids1, label):
        block_repr0, batch_id0 = self.forward_one_encoder(Z0, B0, A0, block_lengths0, lengths0, segment_ids0, encoder_id=0)
        block_repr1, batch_id1 = self.forward_one_encoder(Z1, B1, A1, block_lengths1, lengths1, segment_ids1, encoder_id=1)

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
            attn_output, _ = self.attn_layers0[i](attn_block_repr0, block_repr1, block_repr1) # Q, K, V
            attn_output = self.dropout_layers0[i](attn_output)
            attn_block_repr0 = self.norm_layers0[i](attn_block_repr0 + attn_output)
        attn_block_repr0 = attn_block_repr0.transpose(0, 1) # (num_batches, max_seq_len, dim)
        attn_block_repr0 = unbatchify(attn_block_repr0, attn_batch0) # (num_items, dim)

        for i in range(self.num_attn_layers):
            attn_output, _ = self.attn_layers1[i](attn_block_repr1, block_repr0, block_repr0) # Q, K, V
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


class BinaryPredictorMSP(nn.Module):

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
            CrossAttentionWithSpatialEncoding(dim_query=self.block_hidden_size, dim_kv=self.block_hidden_size, dim_out=self.block_hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_attn_layers*2)
        ])
        self.norm_layers0 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
            for _ in range(self.num_attn_layers*2)
        ])
        self.attn_layers1 = nn.ModuleList([
            CrossAttentionWithSpatialEncoding(dim_query=self.block_hidden_size, dim_kv=self.block_hidden_size, dim_out=self.block_hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_attn_layers*2)
        ])
        self.norm_layers1 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
            for _ in range(self.num_attn_layers*2)
        ])
        self.atom_attn_layers0 = nn.ModuleList([
            CrossAttentionWithSpatialEncoding(dim_query=self.block_hidden_size, dim_kv=self.block_hidden_size, dim_out=self.block_hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.atom_norm_layers0 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
            for _ in range(self.num_attn_layers)
        ])
        self.atom_attn_layers1 = nn.ModuleList([
            CrossAttentionWithSpatialEncoding(dim_query=self.block_hidden_size, dim_kv=self.block_hidden_size, dim_out=self.block_hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.atom_norm_layers1 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
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
        # self.mut_residual_block_ffn = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.encoder1.hidden_size*2, self.encoder1.hidden_size*2),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.encoder1.hidden_size*2, self.encoder1.hidden_size*2),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.encoder1.hidden_size*2, self.encoder1.hidden_size),
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
            model.attn_layers1.requires_grad_(requires_grad=True)
            model.norm_layers1.requires_grad_(requires_grad=True)
            model.atom_attn_layers0.requires_grad_(requires_grad=True)
            model.atom_norm_layers0.requires_grad_(requires_grad=True)
            model.atom_attn_layers1.requires_grad_(requires_grad=True)
            model.atom_norm_layers1.requires_grad_(requires_grad=True)
            # model.mut_residual_block_ffn.requires_grad_(requires_grad=True)
        if pretrained_model.global_message_passing is False and model.encoder0.global_message_passing is True:
            model.encoder0.edge_embedding_top.requires_grad_(requires_grad=True)
            model.encoder1.edge_embedding_top.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        if pretrained_model.bottom_global_message_passing is False and model.encoder0.bottom_global_message_passing is True:
            model.encoder0.edge_embedding_bottom.requires_grad_(requires_grad=True)
            model.encoder1.edge_embedding_bottom.requires_grad_(requires_grad=True)
            print("Warning: bottom_global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        return model
    
    def forward_one_encoder_bottom(self, Z, B, A, atom_repr, block_lengths, lengths, segment_ids, encoder_id):
        if encoder_id == 0:
            encoder = self.encoder0
        elif encoder_id == 1:
            encoder = self.encoder1
        else:
            raise ValueError(f"Invalid encoder_id: {encoder_id}")

        # batch_id and block_id
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            # transform blocks to single units
            bottom_batch_id = batch_id[block_id]  # [Nu]
            bottom_B = B[block_id]  # [Nu]
            bottom_segment_ids = segment_ids[block_id]  # [Nu]
            bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]

        # bottom level message passing
        edges, edge_attr = encoder.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, 
                                          Z, bottom_block_id, encoder.bottom_global_message_passing, 
                                          top=False)
        atom_repr = encoder.encoder(
            atom_repr, Z, bottom_batch_id, None, edges, edge_attr, 
        )
        return atom_repr
    
    def forward_one_encoder_bottom_to_top(self, Z, B, A, atom_repr, block_lengths, lengths, segment_ids, encoder_id):
        if encoder_id == 0:
            encoder = self.encoder0
        elif encoder_id == 1:
            encoder = self.encoder1
        else:
            raise ValueError(f"Invalid encoder_id: {encoder_id}")

        # batch_id and block_id
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)
        
        # embedding
        top_H_0 = encoder.block_embedding.block_embedding(B)

        # top level message passing
        top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = encoder.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          encoder.global_message_passing, top=True)
        if encoder.bottom_global_message_passing:
            batched_bottom_block_repr, _ = batchify(atom_repr, block_id)
        else:
            atom_mask = A != VOCAB.get_atom_global_idx()
            batched_bottom_block_repr, _ = batchify(atom_repr[atom_mask], block_id[atom_mask])
        
        block_repr_from_bottom = encoder.atom_block_attn(top_H_0.unsqueeze(1), batched_bottom_block_repr)
        top_H_0 = top_H_0 + block_repr_from_bottom.squeeze(1)
        top_H_0 = encoder.atom_block_attn_norm(top_H_0)
        return top_H_0, top_Z, batch_id, edges, edge_attr

    def forward_one_encoder_top(self, top_H_0, top_Z, batch_id, edges, edge_attr, encoder_id):
        if encoder_id == 0:
            encoder = self.encoder0
        elif encoder_id == 1:
            encoder = self.encoder1
        else:
            raise ValueError(f"Invalid encoder_id: {encoder_id}")
        block_repr = encoder.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)
        return block_repr
    
    def forward(self, Z0, B0, A0, block_lengths0, lengths0, segment_ids0, mut_block0,
                Z1, B1, A1, block_lengths1, lengths1, segment_ids1, mut_block1, label):
        
        atom_repr0 = self.encoder0.block_embedding.atom_embedding(A0)
        atom_repr1 = self.encoder1.block_embedding.atom_embedding(A1)

        # apply cross attention on the bottom level
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids0)  # [Nb]
            batch_id[torch.cumsum(lengths0, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch
            block_id = torch.zeros_like(A0) # [Nu]
            block_id[torch.cumsum(block_lengths0, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)
            # transform blocks to single units
            bottom_batch_id0 = batch_id[block_id]  # [Nu]

            batch_id = torch.zeros_like(segment_ids1)  # [Nb]
            batch_id[torch.cumsum(lengths1, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch
            block_id = torch.zeros_like(A1) # [Nu]
            block_id[torch.cumsum(block_lengths1, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)
            # transform blocks to single units
            bottom_batch_id1 = batch_id[block_id]  # [Nu]
        
        atom_repr0, attn_batch0 = batchify(atom_repr0, bottom_batch_id0) # (num_batches, max_seq_len0, dim), (num_items, max_seq_len0)
        Z0_batched, _ = batchify(Z0, bottom_batch_id0) # (num_batches, max_seq_len0, 3)
        atom_repr1, attn_batch1 = batchify(atom_repr1, bottom_batch_id1) # (num_batches, max_seq_len1, dim), (num_items, max_seq_len0)
        Z1_batched, _ = batchify(Z1, bottom_batch_id1) # (num_batches, max_seq_len1, 3)
        
        expanded_Z0 = Z0_batched.unsqueeze(2)  # Shape: (num_batches, max_seq_len0, 1, dim)
        expanded_Z1 = Z1_batched.unsqueeze(1)  # Shape: (num_batches, 1, max_seq_len1, dim)
        atom_pairwise_distances0 = torch.sqrt((expanded_Z0 - expanded_Z1).pow(2).sum(dim=-1)) # (num_batches, max_seq_len0, max_seq_len1)
        atom_pairwise_distances1 = atom_pairwise_distances0.transpose(1, 2) # (num_batches, max_seq_len1, max_seq_len0)
        atom_pairwise_mask0 = attn_batch0.unsqueeze(2) * attn_batch1.unsqueeze(1) # (num_batches, max_seq_len0, max_seq_len1)
        atom_pairwise_mask1 = attn_batch1.unsqueeze(2) * attn_batch0.unsqueeze(1) # (num_batches, max_seq_len1, max_seq_len0)
        
        attn_atom_repr0 = atom_repr0.clone()
        attn_atom_repr1 = atom_repr1.clone()

        for i in range(self.num_attn_layers):
            attn_output = self.atom_attn_layers0[i](attn_atom_repr0, atom_repr1, atom_pairwise_distances0, atom_pairwise_mask0) # Q, KV, pdist
            attn_atom_repr0 = self.atom_norm_layers0[i](attn_atom_repr0 + attn_output)
        attn_atom_repr0 = unbatchify(attn_atom_repr0, attn_batch0) # (num_items, dim)

        for i in range(self.num_attn_layers):
            attn_output = self.atom_attn_layers1[i](attn_atom_repr1, atom_repr0, atom_pairwise_distances1, atom_pairwise_mask1) # Q, KV, pdist
            attn_atom_repr1 = self.atom_norm_layers1[i](attn_atom_repr1 + attn_output)
        attn_atom_repr1 = unbatchify(attn_atom_repr1, attn_batch1) # (num_items, dim)

        atom_repr0, atom_repr1 = attn_atom_repr0, attn_atom_repr1


        atom_repr0 = self.forward_one_encoder_bottom(Z0, B0, A0, atom_repr0, block_lengths0, lengths0, segment_ids0, encoder_id=0)
        atom_repr1 = self.forward_one_encoder_bottom(Z1, B1, A1, atom_repr1, block_lengths1, lengths1, segment_ids1, encoder_id=1)

        block_repr0, top_Z0, batch_id0, edges0, edge_attr0 = self.forward_one_encoder_bottom_to_top(Z0, B0, A0, atom_repr0, block_lengths0, lengths0, segment_ids0, encoder_id=0)
        block_repr1, top_Z1, batch_id1, edges1, edge_attr1 = self.forward_one_encoder_bottom_to_top(Z1, B1, A1, atom_repr1, block_lengths1, lengths1, segment_ids1, encoder_id=1)
        # block_repr1[mut_block1] += self.mut_residual_block_ffn(torch.cat([block_repr1[mut_block1], block_repr0[mut_block0]], dim=1))

        top_Z0_batched, _ = batchify(top_Z0, batch_id0) # (num_batches, max_seq_len0, 3)
        top_Z1_batched, _ = batchify(top_Z1, batch_id1) # (num_batches, max_seq_len1, 3)
        expanded_top_Z0 = top_Z0_batched.unsqueeze(2)  # Shape: (num_batches, max_seq_len0, 1, dim)
        expanded_top_Z1 = top_Z1_batched.unsqueeze(1)  # Shape: (num_batches, 1, max_seq_len1, dim)
        pairwise_distances0 = torch.sqrt((expanded_top_Z0 - expanded_top_Z1).pow(2).sum(dim=-1)) # (num_batches, max_seq_len0, max_seq_len1)
        pairwise_distances1 = pairwise_distances0.transpose(1, 2) # (num_batches, max_seq_len1, max_seq_len0)
        
        # apply multihead cross attention with spatial encoding
        block_repr0, attn_batch0 = batchify(block_repr0, batch_id0) # (num_batches, max_seq_len0, dim), (num_items, max_seq_len0)
        block_repr1, attn_batch1 = batchify(block_repr1, batch_id1) # (num_batches, max_seq_len1, dim), (num_items, max_seq_len0)
        pairwise_mask0 = attn_batch0.unsqueeze(2) * attn_batch1.unsqueeze(1) # (num_batches, max_seq_len0, max_seq_len1)
        pairwise_mask1 = attn_batch1.unsqueeze(2) * attn_batch0.unsqueeze(1) # (num_batches, max_seq_len1, max_seq_len0)
        attn_block_repr0 = block_repr0.clone()
        attn_block_repr1 = block_repr1.clone()

        for i in range(self.num_attn_layers):
            attn_output = self.attn_layers0[i](attn_block_repr0, block_repr1, pairwise_distances0, pairwise_mask0) # Q, KV, pdist
            attn_block_repr0 = self.norm_layers0[i](attn_block_repr0 + attn_output)
        attn_block_repr0 = unbatchify(attn_block_repr0, attn_batch0) # (num_items, dim)

        for i in range(self.num_attn_layers):
            attn_output = self.attn_layers1[i](attn_block_repr1, block_repr0, pairwise_distances1, pairwise_mask1) # Q, KV, pdist
            attn_block_repr1 = self.norm_layers1[i](attn_block_repr1 + attn_output)
        attn_block_repr1 = unbatchify(attn_block_repr1, attn_batch1) # (num_items, dim)

        block_repr0, block_repr1 = attn_block_repr0, attn_block_repr1

        block_repr0 = self.forward_one_encoder_top(block_repr0, top_Z0, batch_id0, edges0, edge_attr0, encoder_id=0)
        block_repr1 = self.forward_one_encoder_top(block_repr1, top_Z1, batch_id1, edges1, edge_attr1, encoder_id=1)

        # apply multihead cross attention with spatial encoding
        block_repr0, attn_batch0 = batchify(block_repr0, batch_id0) # (num_batches, max_seq_len0, dim), (num_items, max_seq_len0)
        block_repr1, attn_batch1 = batchify(block_repr1, batch_id1) # (num_batches, max_seq_len1, dim), (num_items, max_seq_len0)
        pairwise_mask0 = attn_batch0.unsqueeze(2) * attn_batch1.unsqueeze(1) # (num_batches, max_seq_len0, max_seq_len1)
        pairwise_mask1 = attn_batch1.unsqueeze(2) * attn_batch0.unsqueeze(1) # (num_batches, max_seq_len1, max_seq_len0)
        attn_block_repr0 = block_repr0.clone()
        attn_block_repr1 = block_repr1.clone()

        for i in range(self.num_attn_layers, self.num_attn_layers*2):
            attn_output = self.attn_layers0[i](attn_block_repr0, block_repr1, pairwise_distances0, pairwise_mask0) # Q, KV, pdist
            attn_block_repr0 = self.norm_layers0[i](attn_block_repr0 + attn_output)
        attn_block_repr0 = unbatchify(attn_block_repr0, attn_batch0) # (num_items, dim)

        for i in range(self.num_attn_layers, self.num_attn_layers*2):
            attn_output = self.attn_layers1[i](attn_block_repr1, block_repr0, pairwise_distances1, pairwise_mask1) # Q, KV, pdist
            attn_block_repr1 = self.norm_layers1[i](attn_block_repr1 + attn_output)
        attn_block_repr1 = unbatchify(attn_block_repr1, attn_batch1) # (num_items, dim)
        block_repr0, block_repr1 = attn_block_repr0, attn_block_repr1

        # predict the label
        mut_block_repr0 = block_repr0[mut_block0]
        mut_block_repr1 = block_repr1[mut_block1]
        assert batch_id0[mut_block0].equal(batch_id1[mut_block1])
        assert len(batch_id0[mut_block0]) == len(label)
        assert len(batch_id0[mut_block0].unique()) == len(label)
        assert len(batch_id1[mut_block1].unique()) == len(label)

        final_repr = torch.cat([mut_block_repr0, mut_block_repr1], dim=1)
        pred = self.pred_ffn(final_repr).squeeze(dim=1)
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
            mut_block0 = batch0['mut_block_id'],
            Z1=batch1['X'], B1=batch1['B'], A1=batch1['A'],
            block_lengths1=batch1['block_lengths'],
            lengths1=batch1['lengths'],
            segment_ids1=batch1['segment_ids'],
            mut_block1 = batch1['mut_block_id'],
            label=label,
        )
        pred = torch.sigmoid(pred)
        return pred


class BinaryPredictorMSP2(nn.Module):

    def __init__(self, atom_hidden_size, block_hidden_size, edge_size, k_neighbors,
                 n_layers, dropout=0.0, bottom_global_message_passing=False, global_message_passing=False, fragmentation_method=None,
                 num_heads=4, num_attn_layers=4) -> None:
        super().__init__()
        self.encoder = PredictionModel(atom_hidden_size=atom_hidden_size, block_hidden_size=block_hidden_size, edge_size=edge_size,
                                        k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, bottom_global_message_passing=bottom_global_message_passing, 
                                        global_message_passing=global_message_passing, fragmentation_method=fragmentation_method)
        # self.encoder1 = PredictionModel(atom_hidden_size=atom_hidden_size, block_hidden_size=block_hidden_size, edge_size=edge_size,    
        #                                 k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, bottom_global_message_passing=bottom_global_message_passing, 
        #                                 global_message_passing=global_message_passing, fragmentation_method=fragmentation_method)
        self.num_attn_layers = num_attn_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.block_hidden_size = block_hidden_size
        self.attn_layers0 = nn.ModuleList([
            CrossAttentionWithSpatialEncoding(dim_query=self.block_hidden_size, dim_kv=self.block_hidden_size, dim_out=self.block_hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.norm_layers0 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
            for _ in range(self.num_attn_layers)
        ])
        self.attn_layers1 = nn.ModuleList([
            CrossAttentionWithSpatialEncoding(dim_query=self.block_hidden_size, dim_kv=self.block_hidden_size, dim_out=self.block_hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_attn_layers)
        ])
        self.norm_layers1 = nn.ModuleList([
            nn.LayerNorm(self.block_hidden_size)
            for _ in range(self.num_attn_layers)
        ])
        self.residual_atom_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.atom_hidden_size, self.encoder.atom_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.atom_hidden_size, self.encoder.atom_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.atom_hidden_size, self.encoder.atom_hidden_size),
        )
        self.residual_block_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
        )
        self.mut_residual_atom_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.atom_hidden_size, self.encoder.atom_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.atom_hidden_size, self.encoder.atom_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.atom_hidden_size, self.encoder.atom_hidden_size),
        )
        self.mut_residual_block_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
        )
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
        print(f"""Pretrained model params: hidden_size={model.encoder.hidden_size},
               edge_size={model.encoder.edge_size}, k_neighbors={model.encoder.k_neighbors}, 
               n_layers={model.encoder.n_layers}, bottom_global_message_passing={model.encoder.bottom_global_message_passing},
               global_message_passing={model.encoder.global_message_passing}, 
               fragmentation_method={model.encoder.fragmentation_method}""")
        assert not any([model.encoder.atom_noise, model.encoder.translation_noise, model.encoder.rotation_noise, model.encoder.torsion_noise]), "prediction model no noise"
        model.encoder.load_state_dict(pretrained_model.state_dict(), strict=False)

        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)
            model.pred_ffn.requires_grad_(requires_grad=True)
            model.attn_layers0.requires_grad_(requires_grad=True)
            model.norm_layers0.requires_grad_(requires_grad=True)
            model.attn_layers1.requires_grad_(requires_grad=True)
            model.norm_layers1.requires_grad_(requires_grad=True)
            model.residual_atom_ffn.requires_grad_(requires_grad=True)
            model.residual_block_ffn.requires_grad_(requires_grad=True)
        if pretrained_model.global_message_passing is False and model.encoder.global_message_passing is True:
            model.encoder.edge_embedding_top.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        if pretrained_model.bottom_global_message_passing is False and model.encoder.bottom_global_message_passing is True:
            model.encoder.edge_embedding_bottom.requires_grad_(requires_grad=True)
            print("Warning: bottom_global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        return model
    
    def forward_one_encoder_bottom(self, Z, B, A, atom_repr, block_lengths, lengths, segment_ids):
        encoder=self.encoder

        # batch_id and block_id
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            # transform blocks to single units
            bottom_batch_id = batch_id[block_id]  # [Nu]
            bottom_B = B[block_id]  # [Nu]
            bottom_segment_ids = segment_ids[block_id]  # [Nu]
            bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]

        # bottom level message passing
        edges, edge_attr = encoder.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, 
                                          Z, bottom_block_id, encoder.bottom_global_message_passing, 
                                          top=False)
        atom_repr = encoder.encoder(
            atom_repr, Z, bottom_batch_id, None, edges, edge_attr, 
        )
        return atom_repr
    
    def forward_one_encoder_bottom_to_top(self, Z, B, A, atom_repr, block_lengths, lengths, segment_ids):
        encoder=self.encoder

        # batch_id and block_id
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)
        
        # embedding
        top_H_0 = encoder.block_embedding.block_embedding(B)

        # top level message passing
        top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = encoder.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          encoder.global_message_passing, top=True)
        if encoder.bottom_global_message_passing:
            batched_bottom_block_repr, _ = batchify(atom_repr, block_id)
        else:
            atom_mask = A != VOCAB.get_atom_global_idx()
            batched_bottom_block_repr, _ = batchify(atom_repr[atom_mask], block_id[atom_mask])
        
        block_repr_from_bottom = encoder.atom_block_attn(top_H_0.unsqueeze(1), batched_bottom_block_repr)
        top_H_0 = top_H_0 + block_repr_from_bottom.squeeze(1)
        top_H_0 = encoder.atom_block_attn_norm(top_H_0)
        return top_H_0, top_Z, batch_id, edges, edge_attr

    def forward_one_encoder_top(self, top_H_0, top_Z, batch_id, edges, edge_attr):
        encoder=self.encoder
        block_repr = encoder.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)
        return block_repr
    
    def forward(self, Z0, B0, A0, block_lengths0, lengths0, segment_ids0, mut_block0,
                Z1, B1, A1, block_lengths1, lengths1, segment_ids1, mut_block1, label):
        
        atom_repr0 = self.encoder.block_embedding.atom_embedding(A0)
        atom_repr1 = self.encoder.block_embedding.atom_embedding(A1)
        # apply cross attention on the bottom level
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids1)  # [Nb]
            batch_id[torch.cumsum(lengths1, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch
            block_id = torch.zeros_like(A1) # [Nu]
            block_id[torch.cumsum(block_lengths1, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)
            # transform blocks to single units
            bottom_batch_id1 = batch_id[block_id]  # [Nu]

        atom_repr1 += self.residual_atom_ffn(atom_repr1)
        atom_repr1[torch.isin(bottom_batch_id1, mut_block1)] += self.mut_residual_atom_ffn(atom_repr1[torch.isin(bottom_batch_id1, mut_block1)])

        atom_repr0 = self.forward_one_encoder_bottom(Z0, B0, A0, atom_repr0, block_lengths0, lengths0, segment_ids0)
        atom_repr1 = self.forward_one_encoder_bottom(Z1, B1, A1, atom_repr1, block_lengths1, lengths1, segment_ids1)

        block_repr0, top_Z0, batch_id0, edges0, edge_attr0 = self.forward_one_encoder_bottom_to_top(Z0, B0, A0, atom_repr0, block_lengths0, lengths0, segment_ids0)
        block_repr1, top_Z1, batch_id1, edges1, edge_attr1 = self.forward_one_encoder_bottom_to_top(Z1, B1, A1, atom_repr1, block_lengths1, lengths1, segment_ids1)
        block_repr1 += self.residual_block_ffn(block_repr1)
        block_repr1[mut_block1] += self.mut_residual_block_ffn(block_repr1[mut_block1])

        top_Z0_batched, _ = batchify(top_Z0, batch_id0) # (num_batches, max_seq_len0, 3)
        top_Z1_batched, _ = batchify(top_Z1, batch_id1) # (num_batches, max_seq_len1, 3)
        expanded_top_Z0 = top_Z0_batched.unsqueeze(2)  # Shape: (num_batches, max_seq_len0, 1, dim)
        expanded_top_Z1 = top_Z1_batched.unsqueeze(1)  # Shape: (num_batches, 1, max_seq_len1, dim)
        pairwise_distances0 = torch.sqrt((expanded_top_Z0 - expanded_top_Z1).pow(2).sum(dim=-1)) # (num_batches, max_seq_len0, max_seq_len1)
        pairwise_distances1 = pairwise_distances0.transpose(1, 2) # (num_batches, max_seq_len1, max_seq_len0)

        block_repr0 = self.forward_one_encoder_top(block_repr0, top_Z0, batch_id0, edges0, edge_attr0)
        block_repr1 = self.forward_one_encoder_top(block_repr1, top_Z1, batch_id1, edges1, edge_attr1)

        # apply multihead cross attention with spatial encoding
        block_repr0, attn_batch0 = batchify(block_repr0, batch_id0) # (num_batches, max_seq_len0, dim), (num_items, max_seq_len0)
        block_repr1, attn_batch1 = batchify(block_repr1, batch_id1) # (num_batches, max_seq_len1, dim), (num_items, max_seq_len0)
        pairwise_mask0 = attn_batch0.unsqueeze(2) * attn_batch1.unsqueeze(1) # (num_batches, max_seq_len0, max_seq_len1)
        pairwise_mask1 = attn_batch1.unsqueeze(2) * attn_batch0.unsqueeze(1) # (num_batches, max_seq_len1, max_seq_len0)
        attn_block_repr0 = block_repr0.clone()
        attn_block_repr1 = block_repr1.clone()

        for i in range(0, self.num_attn_layers):
            attn_output = self.attn_layers0[i](attn_block_repr0, block_repr1, pairwise_distances0, pairwise_mask0) # Q, KV, pdist
            attn_block_repr0 = self.norm_layers0[i](attn_block_repr0 + attn_output)
        attn_block_repr0 = unbatchify(attn_block_repr0, attn_batch0) # (num_items, dim)

        for i in range(0, self.num_attn_layers):
            attn_output = self.attn_layers1[i](attn_block_repr1, block_repr0, pairwise_distances1, pairwise_mask1) # Q, KV, pdist
            attn_block_repr1 = self.norm_layers1[i](attn_block_repr1 + attn_output)
        attn_block_repr1 = unbatchify(attn_block_repr1, attn_batch1) # (num_items, dim)
        block_repr0, block_repr1 = attn_block_repr0, attn_block_repr1

        # predict the label
        mut_block_repr0 = block_repr0[mut_block0]
        mut_block_repr1 = block_repr1[mut_block1]
        assert batch_id0[mut_block0].equal(batch_id1[mut_block1])
        assert len(batch_id0[mut_block0]) == len(label)
        assert len(batch_id0[mut_block0].unique()) == len(label)
        assert len(batch_id1[mut_block1].unique()) == len(label)

        final_repr = torch.cat([mut_block_repr0, mut_block_repr1], dim=1)
        pred = self.pred_ffn(final_repr).squeeze(dim=1)
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
            mut_block0 = batch0['mut_block_id'],
            Z1=batch1['X'], B1=batch1['B'], A1=batch1['A'],
            block_lengths1=batch1['block_lengths'],
            lengths1=batch1['lengths'],
            segment_ids1=batch1['segment_ids'],
            mut_block1 = batch1['mut_block_id'],
            label=label,
        )
        pred = torch.sigmoid(pred)
        return pred