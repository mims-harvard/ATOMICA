#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from .prediction_model import PredictionModel, PredictionReturnValue
import torch

class ClassifierModel(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classifier_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 1),
        )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)
            model.classifier_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        logit = self.classifier_ffn(return_value.graph_repr).flatten()
        return F.binary_cross_entropy_with_logits(logit, label)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        logit = self.classifier_ffn(return_value.graph_repr)
        pred_label = F.sigmoid(logit)
        if extra_info:
            return pred_label, return_value
        return pred_label


class MultiClassClassifierModel(PredictionModel):

    def __init__(self, num_classes, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size*4),
            nn.ReLU(),
            nn.Linear(self.hidden_size*4, self.hidden_size*4),
            nn.ReLU(),
            nn.Linear(self.hidden_size*4, self.num_classes),
        )
    
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
            num_classes=kwargs['num_classes'],
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

        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)
            model.classifier_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        logits = self.classifier_ffn(return_value.graph_repr)
        return F.cross_entropy(logits, label)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        logits = self.classifier_ffn(return_value.graph_repr)
        pred_label = F.softmax(logits, dim=1)
        if extra_info:
            return pred_label, return_value
        return pred_label


class RegressionPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        nonlinearity = nn.ReLU
        self.energy_ffn = nn.Sequential(
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.energy_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        pred_energy = self.energy_ffn(return_value.graph_repr).squeeze(-1)
        return F.mse_loss(pred_energy, label), pred_energy  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        pred_energy = self.energy_ffn(return_value.graph_repr).squeeze(-1)
        if extra_info:
            return pred_energy, return_value
        return pred_energy
    