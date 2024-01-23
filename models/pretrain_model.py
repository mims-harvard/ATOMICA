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

from data.pdb_utils import VOCAB

from .GET.modules.tools import BlockEmbedding, KNNBatchEdgeConstructor


ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 'noise', 'noise_level',
     'unit_repr', 'block_repr', 'graph_repr',
     'batch_id', 'block_id',
     'loss', 'noise_loss', 'noise_level_loss', 'align_loss'],
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

    def __init__(self, model_type, hidden_size, n_channel,
                 n_rbf=1, cutoff=7.0, n_head=1,
                 radial_size=16, edge_size=64, k_neighbors=9, n_layers=3,
                 sigma_begin=10, sigma_end=0.01, n_noise_level=50,
                 dropout=0.1, std=10, global_message_passing=True,
                 atom_level=False, hierarchical=False, no_block_embedding=False, denoising=True) -> None:
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
            global_node_id_vocab=[self.global_block_id],
            delete_self_loop=False)
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
            self.encoder = InteractNNEncoder(hidden_size, edge_size, n_layers, return_noise=denoising)
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
        
        self.energy_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.encoder.out_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        # self.noise_level_ffn = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, n_noise_level)
        # )

        # TODO: add zero noise level
        # sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), n_noise_level)), dtype=torch.float)
        # self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # [n_noise_level]

    @torch.no_grad()
    def choose_receptor(self, batch_size, device):
        segment_retain = (torch.randn((batch_size, ), device=device) > 0).long()  # [bs], 0 or 1
        return segment_retain

    @torch.no_grad()
    def normalize(self, Z, B, block_id, batch_id, segment_ids, receptor_segment):
        # centering
        center = Z[(B[block_id] == self.global_block_id) & (segment_ids[block_id] == receptor_segment[batch_id][block_id])]  # [bs]
        Z = Z - center[batch_id][block_id]
        # normalize
        Z = Z / self.std
        return Z

    @torch.no_grad()
    def perturb(self, Z, block_id, batch_id, batch_size, segment_ids, receptor_segment):
        # noise_level = torch.randint(0, self.sigmas.shape[0], (batch_size,), device=Z.device)
        # # noise_level = torch.ones((batch_size, ), device=Z.device, dtype=torch.long) * (self.sigmas.shape[0] - 1)
        # used_sigmas = self.sigmas[noise_level][batch_id]  # [Nb]
        # used_sigmas = used_sigmas[block_id]  # [Nu]
        noise_level = None

        # randomly select one side to perturb (segment type 0 or segment type 1)
        perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        perturb_mask = perturb_block_mask[block_id]  # [Nu]
        assert torch.any(perturb_mask), 'No perturbable nodes!'

        # used_sigmas[~perturb_mask] = 0  # only one side of the complex is perturbed

        eps = np.random.uniform(0.1, 1.0, size=(Z.shape[0], Z.shape[1]))
        eps = torch.tensor(eps).float().cuda().unsqueeze(-1)
        noise = torch.randn(Z.shape).cuda() #  * eps
        noise[~perturb_mask] = 0  # only one side of the complex is perturbed

        Z_perturbed = Z + noise # * used_sigmas.unsqueeze(-1).unsqueeze(-1) # FIXME: I think this is wrong
        noise_level = eps
        return Z_perturbed, noise, noise_level, perturb_mask, perturb_block_mask
    
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
        def delete_self_loop(edges):
            return edges[:, edges[0] != edges[1]]
        intra_edges = delete_self_loop(intra_edges)
        inter_edges = delete_self_loop(inter_edges)
        if global_global_edges is not None:
            global_global_edges = delete_self_loop(global_global_edges)
        if global_normal_edges is not None:
            global_normal_edges = delete_self_loop(global_normal_edges)
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

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=True, return_loss=True) -> ReturnValue:

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
            receptor_segment = self.choose_receptor(batch_size, batch_id.device)
            # normalize
            Z = self.normalize(Z, B, block_id, batch_id, segment_ids, receptor_segment)
            # perturbation
            Z_perturbed, noise, noise_level, perturb_mask, perturb_block_mask = self.perturb(Z, block_id, batch_id, batch_size, segment_ids, receptor_segment)
            noise[A == VOCAB.get_atom_global_idx()] = 0 # global nodes are not perturbed
            Z_perturbed = Z + noise
            perturb_mask = torch.logical_and(perturb_mask, A != VOCAB.get_atom_global_idx())
            Z_perturbed, not_global = self.update_global_block(Z_perturbed, B, block_id)

        Z_perturbed.requires_grad_(True)

        # embedding
        if self.hierarchical:
            bottom_H_0 = self.block_embedding.atom_embedding(A) #  + self.block_embedding.position_embedding(atom_positions) # FIXME: ablation
            top_H_0 = 0 if self.block_embedding.no_block_embedding else self.block_embedding.block_embedding(B)
        else:
            H_0 = self.block_embedding(B, A, atom_positions, block_id)

        # encoding
        if self.hierarchical:
            if self.denoising:
                edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z_perturbed, bottom_block_id)
                unit_repr, _, _, pred_Z, pred_noise = self.encoder(bottom_H_0, Z_perturbed, bottom_block_id, bottom_batch_id, edges, edge_attr)
                
                #top level 
                top_Z = scatter_mean(Z_perturbed if pred_Z is None else pred_Z, block_id, dim=0)  # [Nb, n_channel, 3]
                top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
                edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
                top_H_0 = top_H_0 + scatter_mean(unit_repr, block_id, dim=0)
                _, block_repr, graph_repr, _, pred_noise_top = self.top_encoder(top_H_0, top_Z, top_block_id, batch_id, edges, edge_attr)
                unit_repr = torch.concat([unit_repr, block_repr[block_id]], dim=-1) # unit_repr and block_repr may have different dim size for dim=1
            else:
                # bottom level message passing
                edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z_perturbed, bottom_block_id)
                unit_repr, _, _, pred_Z = self.encoder(bottom_H_0, Z_perturbed, bottom_block_id, bottom_batch_id, edges, edge_attr)

                # top level message passing
                top_Z = scatter_mean(Z_perturbed if pred_Z is None else pred_Z, block_id, dim=0)  # [Nb, n_channel, 3]
                top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
                edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
                top_H_0 = top_H_0 + scatter_mean(unit_repr, block_id, dim=0)
                _, block_repr, graph_repr, _ = self.top_encoder(top_H_0, top_Z, top_block_id, batch_id, edges, edge_attr)
                unit_repr = torch.concat([unit_repr, block_repr[block_id]], dim=-1)
        else:
            edges, edge_attr = self.get_edges(B, batch_id, segment_ids, Z_perturbed, block_id)
            if self.denoising:
                unit_repr, block_repr, graph_repr, pred_Z, pred_noise = self.encoder(H_0, Z_perturbed, block_id, batch_id, edges, edge_attr)
            else:
                unit_repr, block_repr, graph_repr, pred_Z = self.encoder(H_0, Z_perturbed, block_id, batch_id, edges, edge_attr)

        # predict energy
        # must be sum instead of mean! mean will make the gradient (predicted noise) pretty small, and the score net will easily converge to 0

        # predict noise level
        # pred_noise_level = self.noise_level_ffn(graph_repr)  # [batch_size, n_noise_level]

        if self.denoising:
            # use denoising head
            pred_energy = None
            pred_noise = pred_noise.view(-1, self.n_channel, 3)  # [Nu, n_channel, 3]
            pred_noise = torch.clamp(pred_noise, min=-1, max=1)  # [Nu, n_channel, 3]
            if self.hierarchical:
                pred_noise_top = pred_noise_top.view(-1, self.n_channel, 3)  # [Nb, n_channel, 3]
                pred_noise_top = torch.clamp(pred_noise_top, min=-1, max=1)  # [Nb, n_channel, 3]
                top_noise = scatter_mean(noise, block_id, dim=0)  # [Nb, n_channel, 3]
        else:
            # old GET code
            pred_energy = scatter_sum(self.energy_ffn(block_repr).squeeze(-1), batch_id)
            if return_noise or return_loss:
                # predict noise
                pred_noise = self.pred_noise_from_energy(pred_energy, Z_perturbed)
            else:
                pred_noise = None
        
        pred_energy = scatter_sum(self.energy_ffn(block_repr).squeeze(-1), batch_id)
        pred_noise = self.pred_noise_from_energy(graph_repr.sum(dim=-1).sum(), Z_perturbed)
        pred_noise_top = torch.zeros_like(pred_noise_top)

        if return_loss:
            # print(pred_noise[perturb_mask][:10])
            perturb_mask = torch.logical_and(perturb_mask, not_global)  # do not calculate denoising loss on global nodes
            # noise loss
            # noise_loss = F.mse_loss(pred_noise[perturb_mask]*noise_level[perturb_mask], noise[perturb_mask]/noise_level[perturb_mask], reduction='none')  # [Nperturb, n_channel, 3]
            # noise_loss = (pred_noise[perturb_mask]*noise_level[perturb_mask]-noise[perturb_mask]/noise_level[perturb_mask])**2
            print("pred_noise", pred_noise[perturb_mask].max().item(), pred_noise[perturb_mask].min().item(), pred_noise[perturb_mask].mean().item())
            noise_loss = (pred_noise[perturb_mask]-noise[perturb_mask])**2
            noise_loss = noise_loss.sum(dim=-1).sum(dim=-1)  # [Nperturb]
            noise_loss = scatter_mean(noise_loss, batch_id[block_id][perturb_mask])  # [batch_size] # FIXME: used to be scatter_sum
            noise_loss = noise_loss.mean()  # [1]
            if self.hierarchical:
                noise_loss_top = F.mse_loss(pred_noise_top[perturb_block_mask], top_noise[perturb_block_mask], reduction='none')
                noise_loss_top = noise_loss_top.sum(dim=-1).sum(dim=-1)  # [Nperturb]
                noise_loss_top = scatter_mean(noise_loss_top, batch_id[perturb_block_mask])  # [batch_size] # FIXME: used to be scatter_sum
                # noise_loss += 0.5 * noise_loss_top.mean()
            assert not torch.isnan(noise_loss), f'noise_loss is nan!'

            # # align loss
            # align_loss = F.mse_loss(Z_perturbed[perturb_mask] - pred_Z[perturb_mask], noise[perturb_mask], reduction='none')  # [Nperturb, n_channel, 3]
            # align_loss = align_loss.sum(dim=-1).sum(dim=-1)  # [Nperturb]
            # align_loss = scatter_sum(align_loss, batch_id[block_id][perturb_mask])  # [batch_size]
            # align_loss = align_loss.mean()
            align_loss = 0

            # # noise level loss
            # noise_level_loss = F.cross_entropy(pred_noise_level, noise_level)
            noise_level_loss = 0

            # # punishments for trivial solution
            # punish_loss = torch.abs(pred_noise[perturb_mask] ** 2 - 1).sum()
            # print(len(pred_noise[perturb_mask]) * 3, punish_loss)

            # total loss
            loss = noise_loss # + noise_level_loss # + align_loss

        else:
            noise_loss, align_loss, noise_level_loss, loss = None, None, None, None

        return ReturnValue(

            # denoising variables
            energy=pred_energy,
            noise=pred_noise,
            noise_level=0,
            # noise_level=torch.argmax(pred_noise_level, dim=-1),

            # representations
            unit_repr=unit_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=loss,
            noise_loss=noise_loss,
            noise_level_loss=noise_level_loss,
            align_loss=align_loss
        )