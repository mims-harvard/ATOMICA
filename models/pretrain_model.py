#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum
import random
import plotly.graph_objects as go

from data.pdb_utils import VOCAB

from .GET.modules.tools import BlockEmbedding, KNNBatchEdgeConstructor


ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 'block_energy', 'noise', 'noise_level',
     'unit_repr', 'block_repr', 'graph_repr', 'graph_unit_repr',
     'batch_id', 'block_id',
     'loss', 'noise_loss', 'noise_level_loss', 'align_loss', 'atom_loss', 'atom_base', 
     'tor_loss', 'tor_base',
     'rotation_loss', 'translation_loss', 'rotation_base', 'translation_base'],
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


def _expansion(theta, sigma, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2) * np.sin(theta * (l + 1 / 2)) / np.sin(theta / 2)
    return p

def _density(expansion, theta):
    density = expansion * (1 - np.cos(theta)) / np.pi
    density = np.clip(density, 0, 1000)
    return density / density.sum()

def _score(exp, theta, sigma, L=2000):
    dSigma = 0
    for l in range(L):
        hi = np.sin(theta * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(theta * (l + 1 / 2))
        lo = np.sin(theta / 2)
        dlo = 1 / 2 * np.cos(theta / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2) * (lo * dhi - hi * dlo) / (lo ** 2)
    return dSigma / exp + np.sin(theta) / (1 - np.cos(theta))

class DenoisePretrainModel(nn.Module):

    def __init__(self, model_type, hidden_size, n_channel,
                 n_rbf=1, cutoff=7.0, n_head=1,
                 radial_size=16, edge_size=16, k_neighbors=9, n_layers=3,
                 dropout=0.1, std=10, global_message_passing=False,
                 atom_level=False, hierarchical=False, no_block_embedding=False, 
                 denoising=True, atom_noise=True, translation_noise=True, 
                 rotation_noise=True, torsion_noise=True, rot_sigma=1.5, fragmentation_method=None) -> None:
        super().__init__()

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_channel = n_channel
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.n_head = n_head
        self.radial_size = radial_size
        self.edge_size = edge_size
        self.k_neighbors = k_neighbors
        self.n_layers = n_layers
        self.dropout = dropout
        self.std = std
        self.global_message_passing = global_message_passing
        self.atom_level = atom_level
        self.hierarchical = hierarchical
        self.no_block_embedding = no_block_embedding
        self.denoising = denoising
        self.atom_noise = atom_noise
        self.translation_noise = translation_noise
        self.rotation_noise = rotation_noise
        self.torsion_noise = torsion_noise
        self.mse_loss = nn.MSELoss()
        self.fragmentation_method = fragmentation_method

        VOCAB.load_tokenizer(fragmentation_method)

        if self.denoising:
            assert self.atom_noise or self.translation_noise or self.rotation_noise or self.torsion_noise, 'At least one type of noise should be enabled, otherwise the model is not denoising'

        self.theta_range = np.linspace(0.1, np.pi/4, 100)
        self.sigma_range = np.linspace(0.1, rot_sigma, 100)
        self.expansion = [_expansion(self.theta_range, sigma) for sigma in self.sigma_range]
        self.density = [_density(exp, self.theta_range) for exp in self.expansion]
        self.score = [_score(exp, self.theta_range, sigma) for exp, sigma in zip(self.expansion, self.sigma_range)]

        assert not (self.hierarchical and self.atom_level), 'Hierarchical model is incompatible with atom-level model'

        self.global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)

        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            num_atom_position=VOCAB.get_num_atom_pos(),
            embed_size=hidden_size,
            no_block_embedding=no_block_embedding
        )

        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=k_neighbors,
            global_message_passing=global_message_passing,
            global_node_id_vocab=[self.global_block_id, VOCAB.get_atom_global_idx()], # global edges are only constructed for the global block, but not the global atom
            delete_self_loop=True)
        self.edge_embedding = nn.Embedding(4, edge_size)  # [intra / inter / global_global / global_normal]
        
        z_requires_grad = False
        if model_type == 'GET':
            from .GET.encoder import GETEncoder
            self.encoder = GETEncoder(
                hidden_size, radial_size, n_channel,
                n_rbf, cutoff, edge_size, n_layers,
                n_head, dropout=dropout,
                z_requires_grad=z_requires_grad
            )
        elif model_type == 'GETPool':
            from .GET.pool_encoder import GETPoolEncoder
            self.encoder = GETPoolEncoder(
                hidden_size, radial_size, n_channel,
                n_rbf, cutoff, edge_size, n_layers,
                n_head, dropout=dropout,
                z_requires_grad=z_requires_grad
            )
        elif model_type == 'InteractNN':
            from .InteractNN.encoder import InteractNNEncoder
            self.encoder = InteractNNEncoder(
                hidden_size, edge_size, n_layers=n_layers, 
                return_atom_noise=atom_noise, return_global_noise=translation_noise or rotation_noise,
                return_torsion_noise=torsion_noise, global_message_passing=global_message_passing)
        elif model_type == 'SchNet':
            from .SchNet.encoder import SchNetEncoder
            self.encoder = SchNetEncoder(hidden_size, edge_size, n_layers)
        elif model_type == 'EGNN':
            from .EGNN.encoder import EGNNEncoder
            self.encoder = EGNNEncoder(hidden_size, edge_size, n_layers)
        elif model_type == 'DimeNet':
            from .DimeNet.encoder import DimeNetEncoder
            self.encoder = DimeNetEncoder(hidden_size, n_layers)
        elif model_type == 'TorchMD':
            from .TorchMD.encoder import TorchMDEncoder
            self.encoder = TorchMDEncoder(hidden_size, edge_size, n_layers)
        else:
            raise NotImplementedError(f'Model type {model_type} not implemented!')
        
        if self.hierarchical:
            self.top_encoder = deepcopy(self.encoder)
            if self.torsion_noise and model_type == 'InteractNN':
                self.top_encoder.encoder.return_torsion_noise = False # torsion noise is only applied to the bottom level
                self.top_encoder.return_noise = any([self.top_encoder.encoder.return_atom_noise, self.top_encoder.encoder.return_global_noise])
        
        if not self.denoising:
            self.energy_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        else:
            if self.atom_noise:
                self.bottom_scale_noise_ffn = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(2*hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, 1, bias=False)
                )
                if self.hierarchical:
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
                if self.hierarchical:
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
                if self.hierarchical:
                    self.top_rotation_scale_ffn = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                        nn.Linear(hidden_size, 1, bias=False)
                    )
                

    @torch.no_grad()
    def choose_receptor(self, batch_size, device):
        segment_retain = (torch.randn((batch_size, ), device=device) > 0).long()  # [bs], 0 or 1
        return segment_retain
    
    
    @torch.no_grad()
    def rigid_transform(self, Z, rotation_matrices, translation_vectors, perturb_mask, batch_id, batch_size):
        # TODO: SPEED THIS UP!!!! NEED TO RESHAPE Z INTO [B, N, 3] AND USE BATCHED MATRIX MULTIPLICATION
        w = rotation_matrices # [B, 3], B = batch_size
        c = w.norm(dim=-1, keepdim=True)  # [B, 1]
        c1 = torch.sin(c) / c.clamp(min=1e-6) # [B, 1]
        c2 = (1 - torch.cos(c)) / (c ** 2).clamp(min=1e-6) # [B, 1]
        for i in range(batch_size):
            mask = batch_id == i
            mask = torch.logical_and(mask, perturb_mask)
            wi = w[i].unsqueeze(0)  # [1, 3]
            if self.rotation_noise:
                Z[mask] = Z[mask] + c1[i] * torch.cross(wi, Z[mask]) + c2[i] * torch.cross(wi, torch.cross(wi, Z[mask]))
            if self.translation_noise:
                Z[mask] = Z[mask] + translation_vectors[i]
        return Z

    @torch.no_grad()
    def perturb(self, Z, B, block_id, batch_id, bottom_batch_id, batch_size, segment_ids, receptor_segment):
        perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        perturb_mask = perturb_block_mask[block_id]  # [Nu]
        assert torch.any(perturb_mask), 'No perturbable nodes!'

        # Random rigid transform
        sidx = [random.randint(0, 99) for _ in range(batch_size)]  # 0 is padding
        tidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in sidx]
        theta = torch.tensor([self.theta_range[i] for i in tidx]).float().cuda()
        w = torch.randn(batch_size, 3).cuda()
        hat_w = F.normalize(w, dim=-1)
        w = hat_w * theta.unsqueeze(-1) # [batch_size,3]
        eps = np.random.uniform(0.1, 1.0, size=batch_size)
        eps = torch.tensor(eps).float().cuda().unsqueeze(-1)
        hat_t = torch.randn(batch_size, 3).cuda() * eps
        # Apply
        center = Z[(B[block_id] == self.global_block_id) & perturb_mask]  # [bs] center of perturbed segment
        Z_perturbed = Z - center[batch_id][block_id]
        Z_perturbed = self.rigid_transform(Z_perturbed, w, hat_t, perturb_mask, bottom_batch_id, batch_size)
        Z_perturbed = Z_perturbed + center[batch_id][block_id]

        # Apply atom level coordinate noise
        if self.atom_noise:
            atom_noise = torch.clamp(torch.randn_like(Z), min=-1, max=1)  # [Nu, channel, 3]
            atom_noise[~perturb_mask] = 0  # only one side of the complex is perturbed
            Z_perturbed = Z_perturbed + atom_noise
        else:
            atom_noise = None
        return Z_perturbed, hat_w, hat_t, eps, sidx, tidx, atom_noise, perturb_mask, perturb_block_mask

    @torch.no_grad()
    def update_global_block(self, Z, B, block_id):
        is_global = B[block_id] == self.global_block_id  # [Nu]
        scatter_ids = torch.cumsum(is_global.long(), dim=0) - 1  # [Nu]
        not_global = ~is_global
        centers = scatter_mean(Z[not_global], scatter_ids[not_global], dim=0)  # [Nglobal, n_channel, 3], Nglobal = batch_size * 2
        Z = Z.clone()
        Z[is_global] = centers
        return Z, not_global

    def pred_noise_from_energy(self, energy, Z):
        dy = grad(
            [energy.sum()],  
            [Z],
            create_graph=self.training,
            retain_graph=self.training,
        )[0]
        pred_noise = (-dy).view(-1, self.n_channel, 3).contiguous() # the direction of the gradients is where the energy drops the fastest. Noise adopts the opposite direction
        return pred_noise

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
    
    def inertia(self, X, mask, batch_id):
        # X: [Nu, 3], mask: [Nu], batch_id: [Nu]
        inner = (X ** 2).sum(dim=-1) # [Nu]
        inner = inner[...,None,None] * torch.eye(3).to(X)[None,...]  # [Nu,3,3]
        outer = X.unsqueeze(-2) * X.unsqueeze(-1)  # [Nu,3,3]
        inertia = (inner - outer) * mask[...,None,None] # [Nu,3,3]
        return 0.1 * scatter_sum(inertia, batch_id, dim=0)  # [B,3,3]

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, 
                receptor_segment, atom_score, atom_eps, tr_score, tr_eps, rot_score, 
                tor_edges, tor_score, tor_batch, label, return_noise=True, return_loss=True) -> ReturnValue:
        # batch_id and block_id
        with torch.no_grad():

            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            if self.atom_level:  # this is for ablation
                # transform blocks to single units
                batch_id = batch_id[block_id]  # [Nu]
                segment_ids = segment_ids[block_id]  # [Nu]
                B = B[block_id]  # [Nu]
                block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]
            elif self.hierarchical:
                # transform blocks to single units
                bottom_batch_id = batch_id[block_id]  # [Nu]
                bottom_B = B[block_id]  # [Nu]
                bottom_segment_ids = segment_ids[block_id]  # [Nu]
                bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]
            
            batch_size = lengths.shape[0]
            # select receptor
            # receptor_segment = self.choose_receptor(batch_size, batch_id.device)
            # perturbation
            assert Z.shape[1] == 1, "n_channel must be 1"
            Z = Z.squeeze() # [Nu, n_channel, 3] -> [Nu, 3], n_channel == 1
            Z_perturbed = Z.clone()
            # Z_perturbed, hat_w, hat_t, eps, sidx, tidx, atom_noise, perturb_mask, perturb_block_mask = self.perturb(Z, B, block_id, batch_id, bottom_batch_id, batch_size, segment_ids, receptor_segment)
            # Z_perturbed, not_global = self.update_global_block(Z_perturbed, B, block_id)
            Z_perturbed = Z_perturbed.unsqueeze(1)  # [Nu, 1, 3]
            # FIXME: update global atom block A == VOCAB.get_atom_global_idx())

        # Z_perturbed.requires_grad_(True)
        Z_perturbed = Z.unsqueeze(1)
        perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        perturb_mask = perturb_block_mask[block_id]  # [Nu]

        # embedding
        if self.hierarchical:
            bottom_H_0 = self.block_embedding.atom_embedding(A) #  + self.block_embedding.position_embedding(atom_positions) # FIXME: ablation
            top_H_0 = 0 if self.block_embedding.no_block_embedding else self.block_embedding.block_embedding(B)
        else:
            H_0 = self.block_embedding(B, A, atom_positions, block_id)

        # encoding
        if self.hierarchical:
            if self.denoising:
                # bottom level message passing
                edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z_perturbed, bottom_block_id)
                atom_mask = A != VOCAB.get_atom_global_idx() if not self.global_message_passing else None
                _, bottom_block_repr, graph_repr_bottom, _, trans_noise, rot_noise, pred_noise, tor_noise = self.encoder(
                    bottom_H_0, Z_perturbed, bottom_block_id, bottom_batch_id, perturb_mask, edges, edge_attr, 
                    tor_edges=tor_edges, tor_batch=tor_batch, global_mask=atom_mask)
                # top level 
                top_Z = scatter_mean(Z_perturbed, block_id, dim=0)  # [Nb, n_channel, 3]
                top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
                edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
                top_H_0 = top_H_0 + scatter_mean(bottom_block_repr, block_id, dim=0)
                global_mask = B != self.global_block_id if not self.global_message_passing else None
                if self.top_encoder.return_noise:
                    _, block_repr, graph_repr, _, trans_noise_top, rot_noise_top, pred_noise_top, _ = self.top_encoder(
                        top_H_0, top_Z, top_block_id, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)
                else:
                    # For the hierarchical denoising model, if torsion noise only and no global translation or rotation, the top encoder is not required to return noise
                    _, block_repr, graph_repr, _ = self.top_encoder(
                        top_H_0, top_Z, top_block_id, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)
                bottom_block_repr = torch.concat([bottom_block_repr, block_repr[block_id]], dim=-1) # bottom_block_repr and block_repr may have different dim size for dim=1
            else:
                # bottom level message passing
                edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z_perturbed, bottom_block_id)
                atom_mask = A != VOCAB.get_atom_global_idx() if not self.global_message_passing else None
                _, bottom_block_repr, graph_repr_bottom, _ = self.encoder(bottom_H_0, Z_perturbed, bottom_block_id, bottom_batch_id, perturb_mask, edges, edge_attr, global_mask=atom_mask)
                #top level 
                top_Z = scatter_mean(Z_perturbed, block_id, dim=0)  # [Nb, n_channel, 3]
                top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
                edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
                top_H_0 = top_H_0 + scatter_mean(bottom_block_repr, block_id, dim=0)
                global_mask = B != self.global_block_id if not self.global_message_passing else None
                _, block_repr, graph_repr, _ = self.top_encoder(top_H_0, top_Z, top_block_id, batch_id, perturb_block_mask, edges, edge_attr, global_mask=global_mask)
                bottom_block_repr = torch.concat([bottom_block_repr, block_repr[block_id]], dim=-1) # bottom_block_repr and block_repr may have different dim size for dim=1
        else:
            edges, edge_attr = self.get_edges(B, batch_id, segment_ids, Z_perturbed, block_id)
            atom_mask = A != VOCAB.get_atom_global_idx() if not self.global_message_passing else None
            if self.denoising:
                bottom_block_repr, block_repr, graph_repr, _, trans_noise, rot_noise, pred_noise, tor_noise = self.encoder(H_0, Z_perturbed, block_id, perturb_mask, batch_id, edges, edge_attr, global_mask=atom_mask)
            else:
                bottom_block_repr, block_repr, graph_repr, _ = self.encoder(H_0, Z_perturbed, block_id, batch_id, perturb_mask, edges, edge_attr, global_mask=atom_mask)

        # predict energy
        if self.denoising:
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
                if self.hierarchical:
                    top_atom_noise = scatter_mean(atom_score, block_id, dim=0)  # [Nb, 3]
                    pred_noise_scale_top = self.top_scale_noise_ffn(block_repr)
                    pred_noise_top = pred_noise_top * pred_noise_scale_top
                    # pred_noise_top = torch.clamp(pred_noise_top, min=-1, max=1)  # [Nu, n_channel, 3]
                    atom_loss_top = F.mse_loss(atom_eps[batch_id][perturb_block_mask].unsqueeze(-1) * pred_noise_top[perturb_block_mask], top_atom_noise[perturb_block_mask], reduction='none')
                    atom_loss_top = atom_loss_top.sum(dim=-1)  # [Nperturb]
                    atom_loss_top = scatter_mean(atom_loss_top, batch_id[perturb_block_mask])  # [batch_size] # FIXME: used to be scatter_sum
                    atom_loss += atom_loss_top.mean()
                noise_loss += atom_loss
                atom_base = scatter_mean((atom_score[perturb_mask]**2).mean(dim=-1), batch_id[block_id][perturb_mask]).mean() * (2 if self.hierarchical else 1) # [1]
            else:
                atom_loss = torch.tensor(0.0)
                atom_base = torch.tensor(0.0)
            
            if self.torsion_noise:
                tor_loss = F.mse_loss(tor_noise, tor_score, reduction='none') # [n_tor_edges]
                tor_loss = scatter_mean(tor_loss, tor_batch, dim=0) # [batch_size]
                tor_loss = tor_loss.mean() # [1]
                noise_loss += tor_loss
                tor_base = (tor_score**2).mean() # [1]
            else:
                tor_loss = torch.tensor(0.0)
                tor_base = torch.tensor(0.0)

            # Global translation loss
            if self.translation_noise:
                trans_noise_scale = self.bottom_translation_scale_ffn(graph_repr_bottom)
                trans_noise = trans_noise * trans_noise_scale
                tloss = self.mse_loss(tr_eps.unsqueeze(-1) * trans_noise, -tr_score)
                if self.hierarchical:
                    trans_noise_scale_top = self.top_translation_scale_ffn(graph_repr)
                    trans_noise_top = trans_noise_top * trans_noise_scale_top
                    tloss_top = self.mse_loss(tr_eps.unsqueeze(-1) * trans_noise_top, -tr_score)
                else:
                    tloss_top = torch.tensor(0.0).cuda()
                tloss += tloss_top
                translation_base = (tr_score**2).mean() * (2 if self.hierarchical else 1)
                noise_loss += tloss
            else:
                tloss = torch.tensor(0.0)
                translation_base = torch.tensor(0.0)

            # Global rotation loss
            if self.rotation_noise:
                # score = torch.tensor([self.score[i][j] for i,j in zip(sidx, tidx)]).float().cuda()
                rot_noise_scale = self.bottom_rotation_scale_ffn(graph_repr_bottom)
                rot_noise = rot_noise * rot_noise_scale
                wloss = self.mse_loss(rot_noise, rot_score)
                if self.hierarchical:
                    rot_noise_scale_top = self.top_rotation_scale_ffn(graph_repr)
                    rot_noise_top = rot_noise_top * rot_noise_scale_top
                    wloss_top = self.mse_loss(rot_noise_top, rot_score)
                else:
                    wloss_top = torch.tensor(0.0).cuda()
                wloss += wloss_top
                rotation_base = (rot_score**2).mean() * (2 if self.hierarchical else 1)
                noise_loss += wloss
            else:
                wloss = torch.tensor(0.0)
                rotation_base = torch.tensor(0.0)
            
            loss = noise_loss
            align_loss, noise_level_loss = 0, 0
            block_energy = None
            pred_energy = None
        else:
            pred_noise = None,
            noise_loss, align_loss, noise_level_loss, loss = None, None, None, None
            atom_loss, tloss, wloss = None, None, None
            translation_base, rotation_base = None, None
            block_energy = self.energy_ffn(block_repr).squeeze(-1)
            if not self.global_message_passing: # ignore global blocks
                block_energy[B == self.global_block_id] = 0
            pred_energy = scatter_sum(block_energy, batch_id)
        
        return ReturnValue(
            # denoising variables
            energy=pred_energy,
            block_energy=block_energy,
            noise=pred_noise,
            noise_level=0,
            # noise_level=torch.argmax(pred_noise_level, dim=-1),

            # representations
            unit_repr=bottom_block_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,
            graph_unit_repr=graph_repr_bottom if self.hierarchical else None,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=loss,
            noise_loss=noise_loss,
            noise_level_loss=noise_level_loss,
            align_loss=align_loss,
            atom_loss=atom_loss,
            atom_base=atom_base,
            tor_loss=tor_loss,
            tor_base=tor_base,
            rotation_loss=wloss,
            translation_loss=tloss,
            translation_base=translation_base,
            rotation_base=rotation_base,
        )