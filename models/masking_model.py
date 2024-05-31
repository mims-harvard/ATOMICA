#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from .prediction_model import PredictionModel, PredictionReturnValue
import torch


class MaskedNodeModel(PredictionModel):

    def __init__(self, num_layers, num_nodes, **kwargs) -> None:
        super().__init__(**kwargs)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, dropout=self.dropout))
            #layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        # layers.append(nn.Linear(self.hidden_size, num_nodes))
        # self.masked_ffn = nn.Sequential(*layers)
        self.attention_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(self.hidden_size, num_nodes)

    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model = torch.load(pretrain_ckpt, map_location='cpu')
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        model = cls(
            hidden_size=pretrained_model.hidden_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=kwargs.get('k_neighbors', pretrained_model.k_neighbors),
            n_layers=pretrained_model.n_layers,
            dropout=pretrained_model.dropout,
            fragmentation_method=pretrained_model.fragmentation_method if hasattr(pretrained_model, "fragmentation_method") else None, # for backward compatibility
            global_message_passing=kwargs.get('global_message_passing', pretrained_model.global_message_passing),
            num_layers=kwargs['num_layers'],
            num_nodes=kwargs['num_nodes'],
        )
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, global_message_passing={model.global_message_passing}, 
               fragmentation_method={model.fragmentation_method}""")
        assert not any([model.atom_noise, model.translation_noise, model.rotation_noise, model.torsion_noise]), "prediction model no noise"
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        if pretrained_model.global_message_passing is False and model.global_message_passing is True:
            model.edge_embedding.requires_grad_(requires_grad=True)
            model.encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            model.top_encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, masked_blocks, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        #logits = self.masked_ffn(return_value.block_repr[masked_blocks])
        attn_output = return_value.block_repr[masked_blocks]
        for i in range(0, len(self.attention_layers), 3):
            attn_layer = self.attention_layers[i]
            #relu = self.attention_layers[i+1]
            #dropout = self.attention_layers[i+2]
            
            attn_output, _ = attn_layer(attn_output, attn_output, attn_output)
            #attn_output = relu(attn_output)
            #attn_output = dropout(attn_output)
        logits = self.output_layer(attn_output)
        return F.cross_entropy(logits, label), F.softmax(logits, dim=1)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        logits = self.masked_ffn(return_value.block_repr[batch['masked_blocks']])
        pred_label = F.softmax(logits, dim=1)
        if extra_info:
            return pred_label, return_value
        return pred_label