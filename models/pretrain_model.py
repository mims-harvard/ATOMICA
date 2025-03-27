# Source https://github.com/THUNLP-MT/GET

from collections import namedtuple
from copy import deepcopy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from data.pdb_utils import VOCAB
from data.dataset import MODALITIES
from .tools import BlockEmbedding, KNNBatchEdgeConstructor
from .ATOMICA.encoder import ATOMICAEncoder, AttentionPooling
from .tools import CrossAttention
from .ATOMICA.utils import batchify


ReturnValue = namedtuple(
    'ReturnValue',
    ['unit_repr', 'block_repr', 'graph_repr', 'batch_id', 'block_id', 
     'loss', 'atom_loss', 'atom_base', 'tor_loss', 'tor_base', 'rotation_loss', 
     'translation_loss', 'rotation_base', 'translation_base', 'masked_loss', 'pred_blocks'],
    )


def construct_edges(edge_constructor, B, batch_id, segment_ids, X, block_id, complexity=-1):
    if complexity == -1:  # don't do splicing
        intra_edges, inter_edges, global_normal_edges, global_global_edges, _ = edge_constructor(B, batch_id, segment_ids, X=X, block_id=block_id)
        return intra_edges, inter_edges, global_normal_edges, global_global_edges
    # do splicing
    offset, bs_id_start, bs_id_end = 0, 0, 0
    mini_intra_edges, mini_inter_edges, mini_global_global_edges, mini_global_normal_edges = [], [], [], []
    with torch.no_grad():
        batch_size = batch_id.max() + 1
        unit_batch_id = batch_id[block_id]
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id, dim=0)

        while bs_id_end < batch_size:
            bs_id_start = bs_id_end
            bs_id_end += 1
            while bs_id_end + 1 <= batch_size and \
                  (lengths[bs_id_start:bs_id_end + 1] * lengths[bs_id_start:bs_id_end + 1].max()).sum() < complexity:
                bs_id_end += 1
            # print(bs_id_start, bs_id_end, lengths[bs_id_start:bs_id_end], (lengths[bs_id_start:bs_id_end] * lengths[bs_id_start:bs_id_end].max()).sum())
            
            block_is_in = (batch_id >= bs_id_start) & (batch_id < bs_id_end)
            unit_is_in = (unit_batch_id >= bs_id_start) & (unit_batch_id < bs_id_end)
            B_mini, batch_id_mini, segment_ids_mini = B[block_is_in], batch_id[block_is_in], segment_ids[block_is_in]
            X_mini, block_id_mini = X[unit_is_in], block_id[unit_is_in]

            intra_edges, inter_edges, global_normal_edges, global_global_edges, _ = edge_constructor(
                B_mini, batch_id_mini - bs_id_start, segment_ids_mini, X=X_mini, block_id=block_id_mini - offset)

            if not hasattr(edge_constructor, 'given_intra_edges'):
                mini_intra_edges.append(intra_edges + offset)
            if not hasattr(edge_constructor, 'given_inter_edges'):
                mini_inter_edges.append(inter_edges + offset)
            if global_global_edges is not None:
                mini_global_global_edges.append(global_global_edges + offset)
            if global_normal_edges is not None:
                mini_global_normal_edges.append(global_normal_edges + offset)
            offset += B_mini.shape[0]

        if hasattr(edge_constructor, 'given_intra_edges'):
            intra_edges = edge_constructor.given_intra_edges
        else:
            intra_edges = torch.cat(mini_intra_edges, dim=1)
        if hasattr(edge_constructor, 'given_inter_edges'):
            inter_edges = edge_constructor.given_inter_edges
        else:
            inter_edges = torch.cat(mini_inter_edges, dim=1)
        if global_global_edges is not None:
            global_global_edges = torch.cat(mini_global_global_edges, dim=1)
        if global_normal_edges is not None:
            global_normal_edges = torch.cat(mini_global_normal_edges, dim=1)

    return intra_edges, inter_edges, global_normal_edges, global_global_edges

class DenoisePretrainModel(nn.Module):

    def __init__(self, atom_hidden_size, block_hidden_size, edge_size=16, k_neighbors=9, n_layers=3,
                 dropout=0.0, bottom_global_message_passing=False, global_message_passing=False, fragmentation_method=None,
                 atom_noise=True, translation_noise=True, rotation_noise=True, torsion_noise=True, num_masked_block_classes=None, 
                 atom_weight=1, translation_weight=1, rotation_weight=1, torsion_weight=1, mask_weight=1, modality_embedding=False) -> None:
        super().__init__()

        # model parameters
        self.atom_hidden_size = atom_hidden_size
        self.hidden_size = block_hidden_size
        self.edge_size = edge_size
        self.n_layers = n_layers
        self.dropout = dropout

        # edge parameters
        self.k_neighbors = k_neighbors

        # message passing parameters
        self.global_message_passing = global_message_passing
        self.bottom_global_message_passing = bottom_global_message_passing

        # block embedding parameters
        self.fragmentation_method = fragmentation_method
        VOCAB.load_tokenizer(fragmentation_method)

        # Denoising parameters
        self.atom_noise = atom_noise
        self.translation_noise = translation_noise
        self.rotation_noise = rotation_noise
        self.torsion_noise = torsion_noise
        self.atom_weight = atom_weight
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.torsion_weight = torsion_weight
        self.mask_weight = mask_weight
        self.mse_loss = nn.MSELoss()

        self.global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)

        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            atom_embed_size=atom_hidden_size,
            block_embed_size=block_hidden_size,
            no_block_embedding=False,
        )

        self.use_modality_embedding = modality_embedding
        if self.use_modality_embedding:
            self.modality_embedding = nn.Embedding(len(MODALITIES), block_hidden_size)

        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=k_neighbors,
            global_message_passing=self.global_message_passing or self.bottom_global_message_passing,
            global_node_id_vocab=[self.global_block_id, VOCAB.get_atom_global_idx()], # global edges are only constructed for the global block, but not the global atom
            delete_self_loop=True)
        self.edge_embedding_bottom = nn.Embedding(4, edge_size)  # [intra / inter / global_global / global_normal]
        self.edge_embedding_top = nn.Embedding(4, edge_size)  # [intra / inter / global_global / global_normal]
        
        self.encoder = ATOMICAEncoder(
            atom_hidden_size, edge_size, n_layers=n_layers, dropout=dropout,
            return_atom_noise=atom_noise, return_global_noise=translation_noise or rotation_noise,
            return_torsion_noise=torsion_noise, max_torsion_neighbors=k_neighbors, 
            max_edge_length=5, max_global_edge_length=20, max_torsion_edge_length=5
        )
        self.top_encoder = ATOMICAEncoder(
            block_hidden_size, edge_size, n_layers=n_layers, dropout=dropout, max_edge_length=5
        )
        self.atom_block_attn = CrossAttention(block_hidden_size, atom_hidden_size, block_hidden_size, num_heads=4, dropout=dropout)
        self.atom_block_attn_norm = nn.LayerNorm(block_hidden_size)
        self.attention_pooling = AttentionPooling(block_hidden_size, num_heads=4, dropout=dropout, num_layers=4)

        if self.atom_noise:
            self.top_scale_noise_ffn = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(block_hidden_size, block_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(block_hidden_size, 1, bias=False)
            )
        if self.translation_noise:
            self.top_translation_scale_ffn = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(block_hidden_size, block_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(block_hidden_size, 1, bias=False)
            )
        if self.rotation_noise:
            self.top_rotation_scale_ffn = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(block_hidden_size, block_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(block_hidden_size, 1, bias=False)
            )
        self.num_masked_block_classes = num_masked_block_classes
        if num_masked_block_classes is not None:
            self.masked_ffn = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(self.dropout),
                nn.ReLU(),
                nn.Linear(self.hidden_size, num_masked_block_classes),
                nn.Dropout(self.dropout),
                nn.ReLU(),
                nn.Linear(num_masked_block_classes, num_masked_block_classes),
                nn.Dropout(self.dropout),
            )
            self.masking_objective = True
        else:
            self.masking_objective = False        


    def get_config(self):
        return {
            'atom_hidden_size': self.atom_hidden_size,
            'block_hidden_size': self.hidden_size,
            'edge_size': self.edge_size,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'k_neighbors': self.k_neighbors,
            'global_message_passing': self.global_message_passing,
            'bottom_global_message_passing': self.bottom_global_message_passing,
            'fragmentation_method': self.fragmentation_method,
            'atom_noise': self.atom_noise,
            'translation_noise': self.translation_noise,
            'rotation_noise': self.rotation_noise,
            'torsion_noise': self.torsion_noise,
            'atom_weight': self.atom_weight,
            'translation_weight': self.translation_weight,
            'rotation_weight': self.rotation_weight,
            'torsion_weight': self.torsion_weight,
            'mask_weight': self.mask_weight,
            'modality_embedding': self.use_modality_embedding,
            'num_masked_block_classes': self.num_masked_block_classes,
            'model_type': self.__class__.__name__,
        }
    
    @classmethod
    def load_from_config_and_weights(cls, config_path, weights_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert config['model_type'] == cls.__name__, f"Model type {config['model_type']} does not match {cls.__name__}"
        del config['model_type']
        model = DenoisePretrainModel(**config)
        model.load_state_dict(torch.load(weights_path))
        return model


    def get_edges(self, B, batch_id, segment_ids, Z, block_id, global_message_passing, top):
        intra_edges, inter_edges, global_normal_edges, global_global_edges = construct_edges(
                    self.edge_constructor, B, batch_id, segment_ids, Z, block_id, complexity=2000**2)
        if global_message_passing:
            edges = torch.cat([intra_edges, inter_edges, global_normal_edges, global_global_edges], dim=1)
            edge_attr = torch.cat([
                torch.zeros_like(intra_edges[0]),
                torch.ones_like(inter_edges[0]),
                torch.ones_like(global_normal_edges[0]) * 2,
                torch.ones_like(global_global_edges[0]) * 3])
        else:
            edges = torch.cat([intra_edges, inter_edges], dim=1)
            edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        
        if top:
            edge_attr = self.edge_embedding_top(edge_attr)
        else:
            edge_attr = self.edge_embedding_bottom(edge_attr)

        return edges, edge_attr
    

    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, 
                receptor_segment=None, atom_score=None, atom_eps=None, tr_score=None, 
                tr_eps=None, rot_score=None,tor_edges=None, tor_score=None, tor_batch=None,
                masked_blocks=None, masked_labels=None, modality=None,
                ) -> ReturnValue:
        with torch.no_grad():
            assert tor_edges.shape[1] == tor_score.shape[0], f"tor_edges {tor_edges.shape} and tor_score {tor_score.shape} should have the same length"
            assert self.atom_noise or self.translation_noise or self.rotation_noise or self.torsion_noise, 'At least one type of noise should be enabled, otherwise the model is not denoising'

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

        Z_perturbed = Z

        # embedding
        bottom_H_0 = self.block_embedding.atom_embedding(A)
        top_H_0 = self.block_embedding.block_embedding(B)
        if self.use_modality_embedding:
            top_H_0[B == self.global_block_id] += self.modality_embedding(modality[batch_id[B == self.global_block_id]])
        # encoding
        perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        perturb_mask = perturb_block_mask[block_id]  # [Nu]
        
        # bottom level message passing
        edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, 
                                          Z_perturbed, bottom_block_id, self.bottom_global_message_passing, 
                                          top=False)
        bottom_block_repr, trans_noise, rot_noise, pred_noise, tor_noise = self.encoder(
            bottom_H_0, Z_perturbed, bottom_batch_id, perturb_mask, edges, edge_attr, 
            tor_edges=tor_edges, tor_batch=tor_batch)
        
        # top level message passing 
        top_Z = scatter_mean(Z_perturbed, block_id, dim=0)  # [Nb, n_channel, 3]
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
        
        if self.global_message_passing:
            graph_repr = self.attention_pooling(block_repr, batch_id)
        else:
            global_mask = B != self.global_block_id
            graph_repr = self.attention_pooling(block_repr[global_mask], batch_id[global_mask])
        
        noise_loss = torch.tensor(0.0).cuda()
        # Atom denoising loss
        if self.atom_noise:
            pred_noise_scale_top = self.top_scale_noise_ffn(block_repr)[block_id]
            pred_noise = pred_noise * pred_noise_scale_top

            # pred_noise = torch.clamp(pred_noise, min=-1, max=1)  # [Nu, n_channel, 3]
            atom_loss = F.mse_loss(atom_eps[bottom_batch_id][perturb_mask].unsqueeze(-1) * pred_noise[perturb_mask], 
                                   atom_score[perturb_mask], reduction='none')  # [Nperturb, 3]
            atom_loss = atom_loss.sum(dim=-1)  # [Nperturb]
            atom_loss = scatter_mean(atom_loss, batch_id[block_id][perturb_mask])  # [batch_size]
            atom_loss = atom_loss.mean()  # [1]

            noise_loss += self.atom_weight * atom_loss
            atom_base = scatter_mean((atom_score[perturb_mask]**2).mean(dim=-1), batch_id[block_id][perturb_mask]).mean() # [1]
        else:
            atom_loss = torch.tensor(0.0)
            atom_base = torch.tensor(0.0)
        
        # Torsion denoising loss
        if self.torsion_noise:
            tor_loss = F.mse_loss(tor_noise, tor_score, reduction='none') # [n_tor_edges]
            tor_loss = scatter_mean(tor_loss, tor_batch, dim=0) # [batch_size]
            tor_loss = tor_loss.mean() # [1]
            noise_loss += self.torsion_weight * tor_loss
            tor_base = (tor_score**2).mean() # [1]
        else:
            tor_loss = torch.tensor(0.0)
            tor_base = torch.tensor(0.0)

        # Global translation loss
        if self.translation_noise:
            trans_noise_scale_top = self.top_translation_scale_ffn(graph_repr)
            trans_noise = trans_noise * trans_noise_scale_top # [batch, 3]
            tloss = self.mse_loss(tr_eps.unsqueeze(-1) * trans_noise, -tr_score)
            translation_base = (tr_score**2).mean() # [1]
            noise_loss += self.translation_weight * tloss
        else:
            tloss = torch.tensor(0.0)
            translation_base = torch.tensor(0.0)

        # Global rotation loss
        if self.rotation_noise:
            rot_noise_scale_top = self.top_rotation_scale_ffn(graph_repr)
            rot_noise = rot_noise * rot_noise_scale_top
            wloss = self.mse_loss(rot_noise, rot_score)
            rotation_base = (rot_score**2).mean() # [1]
            noise_loss += self.rotation_weight * wloss
        else:
            wloss = torch.tensor(0.0)
            rotation_base = torch.tensor(0.0)
        
        # Masking loss
        if self.masking_objective:
            logits = self.masked_ffn(block_repr[masked_blocks])
            masked_loss = F.cross_entropy(logits, masked_labels)
            noise_loss += masked_loss * self.mask_weight
            pred_blocks = F.softmax(logits, dim=1)
        else:
            masked_loss = torch.tensor(0.0)
            pred_blocks = None

        return ReturnValue(
            # representations
            unit_repr=bottom_block_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=noise_loss,

            atom_loss=atom_loss,
            atom_base=atom_base,

            tor_loss=tor_loss,
            tor_base=tor_base,

            rotation_loss=wloss,
            rotation_base=rotation_base,

            translation_loss=tloss,
            translation_base=translation_base,

            masked_loss=masked_loss,
            pred_blocks=pred_blocks,
        )


class DenoisePretrainModelWithBlockEmbedding(DenoisePretrainModel):
    def __init__(self, atom_hidden_size, block_hidden_size, edge_size=16, k_neighbors=9, n_layers=3,
                 dropout=0.0, bottom_global_message_passing=False, global_message_passing=False, fragmentation_method=None,
                 atom_noise=True, translation_noise=True, rotation_noise=True, torsion_noise=True, num_masked_block_classes=None, 
                 atom_weight=1, translation_weight=1, rotation_weight=1, torsion_weight=1, mask_weight=1, modality_embedding=False,
                 num_projector_layers=3, projector_hidden_size=32, projector_dropout=0,
                 block_embedding_size=None, block_embedding0_size=None, block_embedding1_size=None) -> None:
        super().__init__(
            atom_hidden_size, block_hidden_size, edge_size, k_neighbors, n_layers, dropout, bottom_global_message_passing, global_message_passing, fragmentation_method,
            atom_noise, translation_noise, rotation_noise, torsion_noise, num_masked_block_classes, 
            atom_weight, translation_weight, rotation_weight, torsion_weight, mask_weight, modality_embedding,
        )
        self.num_projector_layers = num_projector_layers
        self.projector_hidden_size = projector_hidden_size
        self.projector_dropout = projector_dropout

        # same block embedding for all blocks
        nonlinearity = nn.ReLU()
        self.block_embedding_size = block_embedding_size
        if self.block_embedding_size:
            params = (nonlinearity, block_embedding_size, projector_dropout, projector_hidden_size, num_projector_layers)
            block_projector, block_mixing = self.init_block_embedding(*params)
            self.pre_projector = nn.Sequential(*block_projector)
            self.pre_mixing_ffn = nn.Sequential(*block_mixing)

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
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'block_embedding_size': self.block_embedding_size,
            'block_embedding0_size': self.block_embedding0_size,
            'block_embedding1_size': self.block_embedding1_size,
            'num_projector_layers': self.num_projector_layers,
            'projector_dropout': self.projector_dropout,
            'projector_hidden_size': self.projector_hidden_size,
        })
        return config_dict
    
    @classmethod
    def load_from_config_and_weights(cls, config_path, weights_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert config['model_type'] == cls.__name__, f"Model type {config['model_type']} does not match {cls.__name__}"
        del config['model_type']
        model = DenoisePretrainModelWithBlockEmbedding(**config)
        model.load_state_dict(torch.load(weights_path))
        return model
    
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
    

    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, 
                receptor_segment=None, atom_score=None, atom_eps=None, tr_score=None, 
                tr_eps=None, rot_score=None,tor_edges=None, tor_score=None, tor_batch=None,
                masked_blocks=None, masked_labels=None, modality=None,
                block_embeddings=None, block_embeddings0=None, block_embeddings1=None,
                ) -> ReturnValue:
        with torch.no_grad():
            assert tor_edges.shape[1] == tor_score.shape[0], f"tor_edges {tor_edges.shape} and tor_score {tor_score.shape} should have the same length"
            assert self.atom_noise or self.translation_noise or self.rotation_noise or self.torsion_noise, 'At least one type of noise should be enabled, otherwise the model is not denoising'

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

        Z_perturbed = Z

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

        if self.use_modality_embedding:
            top_H_0[B == self.global_block_id] += self.modality_embedding(modality[batch_id[B == self.global_block_id]])
        # encoding
        perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        perturb_mask = perturb_block_mask[block_id]  # [Nu]
        
        # bottom level message passing
        edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, 
                                          Z_perturbed, bottom_block_id, self.bottom_global_message_passing, 
                                          top=False)
        bottom_block_repr, trans_noise, rot_noise, pred_noise, tor_noise = self.encoder(
            bottom_H_0, Z_perturbed, bottom_batch_id, perturb_mask, edges, edge_attr, 
            tor_edges=tor_edges, tor_batch=tor_batch)
        
        # top level message passing 
        top_Z = scatter_mean(Z_perturbed, block_id, dim=0)  # [Nb, n_channel, 3]
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
        
        if self.global_message_passing:
            graph_repr = self.attention_pooling(block_repr, batch_id)
        else:
            global_mask = B != self.global_block_id
            graph_repr = self.attention_pooling(block_repr[global_mask], batch_id[global_mask])
        
        noise_loss = torch.tensor(0.0).cuda()
        # Atom denoising loss
        if self.atom_noise:
            pred_noise_scale_top = self.top_scale_noise_ffn(block_repr)[block_id]
            pred_noise = pred_noise * pred_noise_scale_top

            # pred_noise = torch.clamp(pred_noise, min=-1, max=1)  # [Nu, n_channel, 3]
            atom_loss = F.mse_loss(atom_eps[bottom_batch_id][perturb_mask].unsqueeze(-1) * pred_noise[perturb_mask], 
                                   atom_score[perturb_mask], reduction='none')  # [Nperturb, 3]
            atom_loss = atom_loss.sum(dim=-1)  # [Nperturb]
            atom_loss = scatter_mean(atom_loss, batch_id[block_id][perturb_mask])  # [batch_size]
            atom_loss = atom_loss.mean()  # [1]

            noise_loss += self.atom_weight * atom_loss
            atom_base = scatter_mean((atom_score[perturb_mask]**2).mean(dim=-1), batch_id[block_id][perturb_mask]).mean() # [1]
        else:
            atom_loss = torch.tensor(0.0)
            atom_base = torch.tensor(0.0)
        
        # Torsion denoising loss
        if self.torsion_noise:
            tor_loss = F.mse_loss(tor_noise, tor_score, reduction='none') # [n_tor_edges]
            tor_loss = scatter_mean(tor_loss, tor_batch, dim=0) # [batch_size]
            tor_loss = tor_loss.mean() # [1]
            noise_loss += self.torsion_weight * tor_loss
            tor_base = (tor_score**2).mean() # [1]
        else:
            tor_loss = torch.tensor(0.0)
            tor_base = torch.tensor(0.0)

        # Global translation loss
        if self.translation_noise:
            trans_noise_scale_top = self.top_translation_scale_ffn(graph_repr)
            trans_noise = trans_noise * trans_noise_scale_top # [batch, 3]
            tloss = self.mse_loss(tr_eps.unsqueeze(-1) * trans_noise, -tr_score)
            translation_base = (tr_score**2).mean() # [1]
            noise_loss += self.translation_weight * tloss
        else:
            tloss = torch.tensor(0.0)
            translation_base = torch.tensor(0.0)

        # Global rotation loss
        if self.rotation_noise:
            rot_noise_scale_top = self.top_rotation_scale_ffn(graph_repr)
            rot_noise = rot_noise * rot_noise_scale_top
            wloss = self.mse_loss(rot_noise, rot_score)
            rotation_base = (rot_score**2).mean() # [1]
            noise_loss += self.rotation_weight * wloss
        else:
            wloss = torch.tensor(0.0)
            rotation_base = torch.tensor(0.0)
        
        # Masking loss
        if self.masking_objective:
            logits = self.masked_ffn(block_repr[masked_blocks])
            masked_loss = F.cross_entropy(logits, masked_labels)
            noise_loss += masked_loss * self.mask_weight
            pred_blocks = F.softmax(logits, dim=1)
        else:
            masked_loss = torch.tensor(0.0)
            pred_blocks = None

        return ReturnValue(
            # representations
            unit_repr=bottom_block_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=noise_loss,

            atom_loss=atom_loss,
            atom_base=atom_base,

            tor_loss=tor_loss,
            tor_base=tor_base,

            rotation_loss=wloss,
            rotation_base=rotation_base,

            translation_loss=tloss,
            translation_base=translation_base,

            masked_loss=masked_loss,
            pred_blocks=pred_blocks,
        )