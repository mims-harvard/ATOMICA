#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from data.pdb_utils import VOCAB
from .tools import BlockEmbedding, KNNBatchEdgeConstructor
from .InteractNN.encoder import InteractNNEncoder


ReturnValue = namedtuple(
    'ReturnValue',
    ['unit_repr', 'block_repr', 'graph_repr', 'graph_unit_repr', 'batch_id', 'block_id', 
     'loss', 'atom_loss', 'atom_base', 'tor_loss', 'tor_base', 'rotation_loss', 
     'translation_loss', 'rotation_base', 'translation_base'],
    )


def construct_edges(edge_constructor, B, batch_id, segment_ids, X, block_id, complexity=-1):
    if complexity == -1:  # don't do splicing
        intra_edges, inter_edges, global_global_edges, global_normal_edges, _ = edge_constructor(B, batch_id, segment_ids, X=X, block_id=block_id)
        return intra_edges, inter_edges, global_global_edges, global_normal_edges
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

            intra_edges, inter_edges, global_global_edges, global_normal_edges, _ = edge_constructor(
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

    return intra_edges, inter_edges, global_global_edges, global_normal_edges


class DenoisePretrainModel(nn.Module):

    def __init__(self, hidden_size, edge_size=16, k_neighbors=9, n_layers=3,
                 dropout=0.1, global_message_passing=False, fragmentation_method=None,
                 atom_noise=True, translation_noise=True, rotation_noise=True, torsion_noise=True, 
                 atom_weight=1, translation_weight=1, rotation_weight=1, torsion_weight=1) -> None:
        super().__init__()

        # model parameters
        self.hidden_size = hidden_size
        self.edge_size = edge_size
        self.n_layers = n_layers
        self.dropout = dropout

        # edge parameters
        self.k_neighbors = k_neighbors

        # message passing parameters
        self.global_message_passing = global_message_passing

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
        self.mse_loss = nn.MSELoss()

        self.global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)

        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            embed_size=hidden_size,
            no_block_embedding=False,
        )

        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=k_neighbors,
            global_message_passing=global_message_passing,
            global_node_id_vocab=[self.global_block_id, VOCAB.get_atom_global_idx()], # global edges are only constructed for the global block, but not the global atom
            delete_self_loop=True)
        self.edge_embedding = nn.Embedding(4, edge_size)  # [intra / inter / global_global / global_normal]
        
        self.encoder = InteractNNEncoder(
            hidden_size, edge_size, n_layers=n_layers, 
            return_atom_noise=atom_noise, return_global_noise=translation_noise or rotation_noise,
            return_torsion_noise=torsion_noise, global_message_passing=global_message_passing)
        self.top_encoder = deepcopy(self.encoder)

        # Denoising heads
        if self.torsion_noise:
            self.top_encoder.encoder.remove_torsion_denoiser() # torsion noise is only applied to the bottom level
            self.top_encoder.return_noise = any([self.top_encoder.encoder.return_atom_noise, self.top_encoder.encoder.return_global_noise])
        
        if self.atom_noise:
            self.bottom_scale_noise_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2*hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False)
            )
            self.top_scale_noise_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        if self.translation_noise:
            self.bottom_translation_scale_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False)
            )
            self.top_translation_scale_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        if self.rotation_noise:
            self.bottom_rotation_scale_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False)
            )
            self.top_rotation_scale_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False)
            )

    def get_edges(self, B, batch_id, segment_ids, Z, block_id):
        intra_edges, inter_edges, global_global_edges, global_normal_edges = construct_edges(
                    self.edge_constructor, B, batch_id, segment_ids, Z, block_id, complexity=2000**2)
        if self.global_message_passing:
            edges = torch.cat([intra_edges, inter_edges, global_global_edges, global_normal_edges], dim=1)
            edge_attr = torch.cat([
                torch.zeros_like(intra_edges[0]),
                torch.ones_like(inter_edges[0]),
                torch.ones_like(global_global_edges[0]) * 2,
                torch.ones_like(global_normal_edges[0]) * 3])
        else:
            edges = torch.cat([intra_edges, inter_edges], dim=1)
            edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        edge_attr = self.edge_embedding(edge_attr)

        return edges, edge_attr
    

    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, 
                receptor_segment=None, atom_score=None, atom_eps=None, tr_score=None, 
                tr_eps=None, rot_score=None,tor_edges=None, tor_score=None, tor_batch=None,
                ) -> ReturnValue:
        # batch_id and block_id
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
        bottom_H_0 = self.block_embedding.atom_embedding(A) # FIXME: ablation
        top_H_0 = self.block_embedding.block_embedding(B)

        # encoding
        perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        perturb_mask = perturb_block_mask[block_id]  # [Nu]
        
        # bottom level message passing
        edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z_perturbed, bottom_block_id)
        atom_mask = A != VOCAB.get_atom_global_idx() if not self.global_message_passing else None
        bottom_block_repr, graph_repr_bottom, trans_noise, rot_noise, pred_noise, tor_noise = self.encoder(
            bottom_H_0, Z_perturbed, bottom_batch_id, perturb_mask, edges, edge_attr, 
            tor_edges=tor_edges, tor_batch=tor_batch, global_mask=atom_mask)
        
        # top level message passing
        top_Z = scatter_mean(Z_perturbed, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
        top_H_0 = top_H_0 + scatter_mean(bottom_block_repr, block_id, dim=0)
        global_mask = B != self.global_block_id if not self.global_message_passing else None
        if self.top_encoder.return_noise:
            block_repr, graph_repr, trans_noise_top, rot_noise_top, pred_noise_top, _ = self.top_encoder(
                top_H_0, top_Z, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)
        else:
            # For the hierarchical denoising model, if torsion noise only and no global translation or rotation, the top encoder is not required to return noise
            block_repr, graph_repr = self.top_encoder(
                top_H_0, top_Z, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)
        bottom_block_repr = torch.concat([bottom_block_repr, block_repr[block_id]], dim=-1) # bottom_block_repr and block_repr may have different dim size for dim=1
        
        noise_loss = torch.tensor(0.0).cuda()

        # Atom denoising loss
        if self.atom_noise:
            pred_noise_scale = self.bottom_scale_noise_ffn(bottom_block_repr)
            pred_noise = pred_noise * pred_noise_scale

            # pred_noise = torch.clamp(pred_noise, min=-1, max=1)  # [Nu, n_channel, 3]
            atom_loss = F.mse_loss(atom_eps[bottom_batch_id][perturb_mask].unsqueeze(-1) * pred_noise[perturb_mask], atom_score[perturb_mask], reduction='none')  # [Nperturb, 3]
            atom_loss = atom_loss.sum(dim=-1)  # [Nperturb]
            atom_loss = scatter_mean(atom_loss, batch_id[block_id][perturb_mask])  # [batch_size] # FIXME: used to be scatter_sum
            atom_loss = atom_loss.mean()  # [1]

            top_atom_noise = scatter_mean(atom_score, block_id, dim=0)  # [Nb, 3]
            pred_noise_scale_top = self.top_scale_noise_ffn(block_repr)
            pred_noise_top = pred_noise_top * pred_noise_scale_top
            # pred_noise_top = torch.clamp(pred_noise_top, min=-1, max=1)  # [Nu, n_channel, 3]
            atom_loss_top = F.mse_loss(atom_eps[batch_id][perturb_block_mask].unsqueeze(-1) * pred_noise_top[perturb_block_mask], top_atom_noise[perturb_block_mask], reduction='none')
            atom_loss_top = atom_loss_top.sum(dim=-1)  # [Nperturb]
            atom_loss_top = scatter_mean(atom_loss_top, batch_id[perturb_block_mask])  # [batch_size] # FIXME: used to be scatter_sum
            atom_loss += atom_loss_top.mean()
            noise_loss += self.atom_weight * atom_loss
            atom_base = scatter_mean((atom_score[perturb_mask]**2).mean(dim=-1), batch_id[block_id][perturb_mask]).mean() * 2 # [1], 2 for bottom and top level
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
            trans_noise_scale = self.bottom_translation_scale_ffn(graph_repr_bottom)
            trans_noise = trans_noise * trans_noise_scale
            tloss = self.mse_loss(tr_eps.unsqueeze(-1) * trans_noise, -tr_score)
            
            trans_noise_scale_top = self.top_translation_scale_ffn(graph_repr)
            trans_noise_top = trans_noise_top * trans_noise_scale_top
            tloss_top = self.mse_loss(tr_eps.unsqueeze(-1) * trans_noise_top, -tr_score)

            tloss += tloss_top
            translation_base = (tr_score**2).mean() * 2 # [1], 2 for bottom and top level
            noise_loss += self.translation_weight * tloss
        else:
            tloss = torch.tensor(0.0)
            translation_base = torch.tensor(0.0)

        # Global rotation loss
        if self.rotation_noise:
            rot_noise_scale = self.bottom_rotation_scale_ffn(graph_repr_bottom)
            rot_noise = rot_noise * rot_noise_scale
            wloss = self.mse_loss(rot_noise, rot_score)
            
            rot_noise_scale_top = self.top_rotation_scale_ffn(graph_repr)
            rot_noise_top = rot_noise_top * rot_noise_scale_top
            wloss_top = self.mse_loss(rot_noise_top, rot_score)

            wloss += wloss_top
            rotation_base = (rot_score**2).mean() * 2 # [1], 2 for bottom and top level
            noise_loss += self.rotation_weight * wloss
        else:
            wloss = torch.tensor(0.0)
            rotation_base = torch.tensor(0.0)
        

        return ReturnValue(
            # representations
            unit_repr=bottom_block_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,
            graph_unit_repr=graph_repr_bottom,

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
        )