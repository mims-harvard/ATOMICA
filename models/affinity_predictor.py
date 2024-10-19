#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import torch

from data.pdb_utils import VOCAB
from .pretrain_model import DenoisePretrainModel
from .InteractNN.utils import batchify
from .prediction_model import PredictionModel, PredictionReturnValue


class AffinityPredictor(PredictionModel):

    def __init__(self, num_affinity_pred_layers, nonlinearity, affinity_pred_dropout, affinity_pred_hidden_size, 
                 num_projector_layers, projector_hidden_size, projector_dropout, 
                 block_embedding_size=None, block_embedding0_size=None, block_embedding1_size=None, # =1536 ESM3 embedding size
                   **kwargs) -> None:
        super().__init__(**kwargs)

        layers = [nonlinearity, nn.Dropout(affinity_pred_dropout), nn.Linear(self.hidden_size, affinity_pred_hidden_size)]
        for _ in range(0, num_affinity_pred_layers-2):
            layers.extend([nonlinearity, nn.Dropout(affinity_pred_dropout), nn.Linear(affinity_pred_hidden_size, affinity_pred_hidden_size)])
        layers.extend([nonlinearity, nn.Dropout(affinity_pred_dropout), nn.Linear(affinity_pred_hidden_size, 1)])
        self.energy_ffn = nn.Sequential(*layers)

        # same block embedding for all blocks
        self.block_embedding_size = block_embedding_size
        if self.block_embedding_size:
            params = (nonlinearity, block_embedding_size, projector_dropout, projector_hidden_size, num_projector_layers)
            block_projector, block_mixing = self.init_block_embedding(*params)
            self.pre_projector = nn.Sequential(*block_projector)
            self.pre_mixing_ffn = nn.Sequential(*block_mixing)
            block_projector, block_mixing = self.init_block_embedding(*params)
            self.post_projector = nn.Sequential(*block_projector)
            self.post_mixing_ffn = nn.Sequential(*block_mixing)

        # different block embedidng for segment 0 and 1
        self.block_embedding0_size = block_embedding0_size
        self.block_embedding1_size = block_embedding1_size
        if self.block_embedding0_size and self.block_embedding1_size:
            params0 = (nonlinearity, block_embedding0_size, projector_dropout, projector_hidden_size, num_projector_layers)
            params1 = (nonlinearity, block_embedding1_size, projector_dropout, projector_hidden_size, num_projector_layers)

            block_projector0, block_mixing0 = self.init_block_embedding(*params0)
            self.pre_projector0 = nn.Sequential(*block_projector0)
            self.pre_mixing_ffn0 = nn.Sequential(*block_mixing0)

            block_projector1, block_mixing1 = self.init_block_embedding(*params1)
            self.pre_projector1 = nn.Sequential(*block_projector1)
            self.pre_mixing_ffn1 = nn.Sequential(*block_mixing1)

            block_projector0, block_mixing0 = self.init_block_embedding(*params0)
            self.post_projector0 = nn.Sequential(*block_projector0)
            self.post_mixing_ffn0 = nn.Sequential(*block_mixing0)

            block_projector1, block_mixing1 = self.init_block_embedding(*params1)
            self.post_projector1 = nn.Sequential(*block_projector1)
            self.post_mixing_ffn1 = nn.Sequential(*block_mixing1)

        self.attention_pooling.requires_grad_(requires_grad=False) # pooling is not used in affinity prediction
    
    def init_block_embedding(self, nonlinearity: nn.Module, block_embedding_size: int, projector_dropout: float, projector_hidden_size: int, num_projector_layers: int):
        projector_layers = [nonlinearity, nn.Dropout(projector_dropout), nn.Linear(block_embedding_size, projector_hidden_size)]
        for _ in range(0, num_projector_layers-2):
            projector_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(projector_hidden_size, projector_hidden_size)])
        projector_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(projector_hidden_size, self.hidden_size)])

        mixing_layers = [nonlinearity, nn.Dropout(projector_dropout), nn.Linear(2*self.hidden_size, 2*self.hidden_size)]
        for _ in range(0, num_projector_layers-2):
            mixing_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(2*self.hidden_size, 2*self.hidden_size)])
        mixing_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(2*self.hidden_size, self.hidden_size)])
        return projector_layers, mixing_layers

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
            nonlinearity=kwargs['nonlinearity'],
            num_affinity_pred_layers=kwargs['num_affinity_pred_layers'],
            affinity_pred_dropout=kwargs['affinity_pred_dropout'],
            affinity_pred_hidden_size=kwargs['affinity_pred_hidden_size'],
            num_projector_layers=kwargs['num_projector_layers'],
            projector_dropout=kwargs['projector_dropout'],
            projector_hidden_size=kwargs['projector_hidden_size'],
            block_embedding_size=kwargs['block_embedding_size'],
            block_embedding0_size=kwargs['block_embedding0_size'],
            block_embedding1_size=kwargs['block_embedding1_size'],
        )
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, bottom_global_message_passing={model.bottom_global_message_passing},
               global_message_passing={model.global_message_passing}, 
               fragmentation_method={model.fragmentation_method}""")
        assert not any([model.atom_noise, model.translation_noise, model.rotation_noise, model.torsion_noise]), "prediction model no noise"
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)

        if pretrained_model.global_message_passing is False and model.global_message_passing is True:
            model.edge_embedding_top.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        
        if pretrained_model.bottom_global_message_passing is False and model.bottom_global_message_passing is True:
            model.edge_embedding_bottom.requires_grad_(requires_grad=True)
            print("Warning: bottom_global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        model.attention_pooling.requires_grad_(requires_grad=False) # pooling is not used in affinity prediction
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.energy_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label, block_embeddings, block_embeddings0, block_embeddings1) -> PredictionReturnValue:
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

        # embedding
        bottom_H_0 = self.block_embedding.atom_embedding(A)
        top_H_0 = self.block_embedding.block_embedding(B)
        if self.block_embedding_size:
            block_embeddings_all = self.pre_projector(block_embeddings)
            top_H_0 = self.pre_mixing_ffn(torch.cat([top_H_0, block_embeddings_all], dim=-1))
        elif self.block_embedding0_size and self.block_embedding1_size:
            block_embeddings_segment0 = self.pre_projector0(block_embeddings0)
            block_embeddings_segment1 = self.pre_projector1(block_embeddings1)
            top_H_0_segment0 = self.pre_mixing_ffn0(torch.cat([top_H_0[segment_ids==0], block_embeddings_segment0], dim=-1))
            top_H_0_segment1 = self.pre_mixing_ffn1(torch.cat([top_H_0[segment_ids==1], block_embeddings_segment1], dim=-1))
            top_H_0 = torch.cat([top_H_0_segment0, top_H_0_segment1], dim=0)

        # bottom level message passing
        edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, 
                                          Z, bottom_block_id, self.bottom_global_message_passing, 
                                          top=False)
        bottom_block_repr = self.encoder(
            bottom_H_0, Z, bottom_batch_id, None, edges, edge_attr, 
        )
        
        # top level message passing
        top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          self.global_message_passing, top=True)
        if self.bottom_global_message_passing:
            batched_bottom_block_repr, _ = batchify(bottom_block_repr, block_id)
        else:
            atom_mask = A != VOCAB.get_atom_global_idx()
            batched_bottom_block_repr, _ = batchify(bottom_block_repr[atom_mask], block_id[atom_mask])
        block_repr_from_bottom = self.atom_block_attn(top_H_0.unsqueeze(1), batched_bottom_block_repr)
        top_H_0 = top_H_0 + block_repr_from_bottom.squeeze(1)
        top_H_0 = self.atom_block_attn_norm(top_H_0)

        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)

        if self.block_embedding_size:
            block_embeddings_all = self.post_projector(block_embeddings)
            block_repr = self.post_mixing_ffn(torch.cat([block_repr, block_embeddings_all], dim=-1))
        elif self.block_embedding0_size and self.block_embedding1_size:
            block_embeddings_segment0 = self.post_projector0(block_embeddings0)
            block_embeddings_segment1 = self.post_projector1(block_embeddings1)
            block_repr_segment0 = self.post_mixing_ffn0(torch.cat([block_repr[segment_ids==0], block_embeddings_segment0], dim=-1))
            block_repr_segment1 = self.post_mixing_ffn1(torch.cat([block_repr[segment_ids==1], block_embeddings_segment1], dim=-1))
            block_repr = torch.cat([block_repr_segment0, block_repr_segment1], dim=0)

        block_energy = self.energy_ffn(block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[B == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, batch_id)
        return F.mse_loss(pred_energy, label), pred_energy  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)

    def infer(self, batch):
        self.eval()
        loss, pred_energy = self.forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
            block_embeddings0=batch.get('block_embeddings0', None),
            block_embeddings1=batch.get('block_embeddings1', None),
        )
        return pred_energy


class BlockAffinityPredictor(PredictionModel):

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
        # self.energy_ffn1 = nn.Sequential(
        #     nonlinearity(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nonlinearity(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nonlinearity(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, 1)
        # )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        model = super().load_from_pretrained(pretrain_ckpt, **kwargs)
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.energy_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label) -> PredictionReturnValue:
        # batch_id and block_id
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

        # embedding
        top_H_0 = self.block_embedding.block_embedding(B)
        top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          self.global_message_passing, top=True)
        block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)
        block_energy = self.energy_ffn(block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[B == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, batch_id)
        return F.mse_loss(pred_energy, label), pred_energy  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)


    def infer(self, batch, extra_info=False):
        self.eval()
        loss, pred = self.forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
        )
        if extra_info:
            raise NotImplementedError
        return pred
    

class AffinityPredictorNoisyNodes(PredictionModel):

    def __init__(self, noisy_nodes_weight, **kwargs) -> None:
        DenoisePretrainModel.__init__(self, **kwargs)
        nonlinearity = nn.ReLU
        self.noisy_nodes_weight = noisy_nodes_weight
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
        # self.energy_ffn1 = nn.Sequential(
        #     nonlinearity(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nonlinearity(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nonlinearity(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, 1)
        # )
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        model = cls(
            noisy_nodes_weight=kwargs['noisy_nodes_weight'],
            atom_hidden_size=pretrained_model.atom_hidden_size,
            block_hidden_size=pretrained_model.hidden_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=kwargs.get('k_neighbors', pretrained_model.k_neighbors),
            n_layers=pretrained_model.n_layers,
            dropout=kwargs.get('dropout', pretrained_model.dropout),
            fragmentation_method=pretrained_model.fragmentation_method if hasattr(pretrained_model, "fragmentation_method") else None, # for backward compatibility
            bottom_global_message_passing=kwargs.get('bottom_global_message_passing', pretrained_model.global_message_passing),
            global_message_passing=kwargs.get('global_message_passing', pretrained_model.global_message_passing),
            atom_noise=pretrained_model.atom_noise, translation_noise=pretrained_model.translation_noise, rotation_noise=pretrained_model.rotation_noise, torsion_noise=pretrained_model.torsion_noise, 
            atom_weight=pretrained_model.atom_weight, translation_weight=pretrained_model.translation_weight, rotation_weight=pretrained_model.rotation_weight, torsion_weight=pretrained_model.torsion_weight
        )
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, bottom_global_message_passing={model.bottom_global_message_passing},
               global_message_passing={model.global_message_passing}, 
               fragmentation_method={model.fragmentation_method}""")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        if kwargs.get('partial_finetune', False):
            model.requires_grad_(requires_grad=False)
            model.energy_ffn.requires_grad_(requires_grad=True)

        if pretrained_model.global_message_passing is False and model.global_message_passing is True:
            model.edge_embedding_top.requires_grad_(requires_grad=True)
            model.top_encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")

        if pretrained_model.bottom_global_message_passing is False and model.bottom_global_message_passing is True:
            model.edge_embedding_bottom.requires_grad_(requires_grad=True)
            model.encoder.encoder.edge_embedder.requires_grad_(requires_grad=True)
            print("Warning: bottom_global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")

        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, receptor_segment, atom_score, atom_eps, tr_score, 
                tr_eps, rot_score, tor_edges, tor_score, tor_batch, label) -> PredictionReturnValue:
        return_value = DenoisePretrainModel.forward(self, Z, B, A, block_lengths, lengths, segment_ids, receptor_segment, atom_score, atom_eps, tr_score, 
                tr_eps, rot_score, tor_edges, tor_score, tor_batch) # use DenoisePretrainModel.forward()
        block_energy = self.energy_ffn(return_value.block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[B == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, return_value.batch_id)

        # rec_block_energy = self.energy_ffn(return_value.block_repr[segment_ids == 0]).squeeze(-1)
        # lig_block_energy = self.energy_ffn1(return_value.block_repr[segment_ids == 1]).squeeze(-1)
        # if not self.global_message_passing: # ignore global blocks
        #     rec_block_energy[B[segment_ids == 0] == self.global_block_id] = 0
        #     lig_block_energy[B[segment_ids == 1] == self.global_block_id] = 0
        # pred_energy_rec = scatter_sum(rec_block_energy, return_value.batch_id[segment_ids == 0])
        # pred_energy_lig = scatter_sum(lig_block_energy, return_value.batch_id[segment_ids == 1])
        # pred_energy = pred_energy_rec + pred_energy_lig
        pred_loss = F.mse_loss(pred_energy, label)
        return pred_loss, self.noisy_nodes_weight*return_value.loss, pred_energy  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
    
    def _toggle_noise(self, return_noise: bool):
        self.encoder.return_noise = return_noise
        self.encoder.encoder.return_torsion_noise = return_noise
        self.encoder.encoder.return_global_noise = return_noise

    def infer(self, batch, extra_info=False):
        self.eval()
        self._toggle_noise(False)
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        ) # use PredictionModel.forward()
        # rec_block_energy = self.energy_ffn(return_value.block_repr[batch['segment_ids'] == 0]).squeeze(-1)
        # lig_block_energy = self.energy_ffn1(return_value.block_repr[batch['segment_ids'] == 1]).squeeze(-1)
        # if not self.global_message_passing: # ignore global blocks
        #     rec_block_energy[batch['B'][batch['segment_ids'] == 0] == self.global_block_id] = 0
        #     lig_block_energy[batch['B'][batch['segment_ids'] == 1] == self.global_block_id] = 0
        # pred_energy_rec = scatter_sum(rec_block_energy, return_value.batch_id[batch['segment_ids'] == 0])
        # pred_energy_lig = scatter_sum(lig_block_energy, return_value.batch_id[batch['segment_ids'] == 1])
        # pred_energy = pred_energy_rec + pred_energy_lig
        block_energy = self.energy_ffn(return_value.block_repr).squeeze(-1)
        if not self.global_message_passing: # ignore global blocks
            block_energy[batch['B'] == self.global_block_id] = 0
        pred_energy = scatter_sum(block_energy, return_value.batch_id)
        self._toggle_noise(True)
        if extra_info:
            return pred_energy, return_value
        return pred_energy