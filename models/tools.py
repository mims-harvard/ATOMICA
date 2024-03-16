#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_min, scatter_mean

from utils.nn_utils import BatchEdgeConstructor, _knn_edges, print_cuda_memory


def _unit_edges_from_block_edges(unit_block_id, block_src_dst, Z=None, k=None):
    '''
    :param unit_block_id [N], id of block of each unit. Assume X is sorted so that block_id starts from 0 to Nb - 1
    :param block_src_dst: [Eb, 2], all edges (block level), represented in (src, dst)
    '''
    block_n_units = scatter_sum(torch.ones_like(unit_block_id), unit_block_id)  # [Nb], number of units in each block
    block_offsets = F.pad(torch.cumsum(block_n_units[:-1], dim=0), (1, 0), value=0)  # [Nb]
    edge_n_units = block_n_units[block_src_dst]  # [Eb, 2], number of units at two end of the block edges
    edge_n_pairs = edge_n_units[:, 0] * edge_n_units[:, 1]  # [Eb], number of unit-pairs in each edge

    # block edge id for unit pairs
    edge_id = torch.zeros(edge_n_pairs.sum(), dtype=torch.long, device=edge_n_pairs.device)  # [Eu], which edge each unit pair belongs to
    edge_start_index = torch.cumsum(edge_n_pairs, dim=0)[:-1]  # [Eb - 1], start index of each edge (without the first edge as it starts with 0) in unit_src_dst
    edge_id[edge_start_index] = 1
    edge_id = torch.cumsum(edge_id, dim=0)  # [Eu], which edge each unit pair belongs to, start from 0, end with Eb - 1

    # get unit-pair src-dst indexes
    unit_src_dst = torch.ones_like(edge_id)  # [Eu]
    unit_src_dst[edge_start_index] = -(edge_n_pairs[:-1] - 1)  # [Eu], e.g. [1,1,1,-2,1,1,1,1,-4,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    del edge_start_index  # release memory
    if len(unit_src_dst) > 0:
        unit_src_dst[0] = 0 # [Eu], e.g. [0,1,1,-2,1,1,1,1,-4,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    unit_src_dst = torch.cumsum(unit_src_dst, dim=0)  # [Eu], e.g. [0,1,2,0,1,2,3,4,0,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    unit_dst_n = edge_n_units[:, 1][edge_id]  # [Eu], each block edge has m*n unit pairs, here n refers to the number of units in the dst block
    # turn 1D indexes to 2D indexes (TODO: this block is memory-intensive)
    unit_src = torch.div(unit_src_dst, unit_dst_n, rounding_mode='floor') + block_offsets[block_src_dst[:, 0][edge_id]] # [Eu]
    unit_dst = torch.remainder(unit_src_dst, unit_dst_n)  # [Eu], e.g. [0,1,2,0,0,0,0,0,0,1] for block-pair shape 1*3, 5*1, 1*2
    unit_dist_local = unit_dst
    # release some memory
    del unit_dst_n, unit_src_dst  # release memory
    unit_edge_src_start = (unit_dst == 0)
    unit_dst = unit_dst + block_offsets[block_src_dst[:, 1][edge_id]]  # [Eu]
    del block_offsets, block_src_dst # release memory
    unit_edge_src_id = unit_edge_src_start.long()
    if len(unit_edge_src_id) > 1:
        unit_edge_src_id[0] = 0
    unit_edge_src_id = torch.cumsum(unit_edge_src_id, dim=0)  # [Eu], e.g. [0,0,0,1,2,3,4,5,6,6] for the above example

    if k is None:
        return (unit_src, unit_dst), (edge_id, unit_edge_src_start, unit_edge_src_id)

    # sparsify, each atom is connected to the nearest k atoms in the other block in the same block edge

    D = torch.norm(Z[unit_src] - Z[unit_dst], dim=-1) # [Eu, n_channel]
    D = D.sum(dim=-1) # [Eu]
    
    max_n = torch.max(scatter_sum(torch.ones_like(unit_edge_src_id), unit_edge_src_id))
    k = min(k, max_n)

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = unit_edge_src_id.max() + 1
    # src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    dist = torch.norm(Z[unit_src] - Z[unit_dst], dim=-1).sum(-1) # [Eu]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(unit_edge_src_id, unit_dist_local)] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k, dim=-1, largest=False)  # [N, topk]
    del dist_mat

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k)
    unit_edge_src_start = torch.zeros_like(src).bool() # [N, k]
    unit_edge_src_start[:, 0] = True
    src, dst = src.flatten(), dst.flatten()
    unit_edge_src_start = unit_edge_src_start.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)
    unit_edge_src_start = unit_edge_src_start.masked_select(is_valid)

    # extract row, col and edge id
    mat = torch.ones(N, max_n, device=unit_src.device, dtype=unit_src.dtype) * -1
    mat[(unit_edge_src_id, unit_dist_local)] = unit_src
    unit_src = mat[(src, dst)]
    mat[(unit_edge_src_id, unit_dist_local)] = unit_dst
    unit_dst = mat[(src, dst)]
    mat[(unit_edge_src_id, unit_dist_local)] = edge_id
    edge_id = mat[(src, dst)]

    unit_edge_src_id = src
    
    return (unit_src, unit_dst), (edge_id, unit_edge_src_start, unit_edge_src_id)


def _block_edge_dist(X, block_id, src_dst):
    '''
    Several units constitute a block.
    This function calculates the distance of edges between blocks
    The distance between two blocks are defined as the minimum distance of unit-pairs between them.

    :param X: [N, 3], coordinates.
    :param block_id [N], id of block of each unit. Assume X is sorted so that block_id starts from 0 to Nb - 1
    :param src_dst: [Eb, 2], all edges (block level) that needs distance calculation, represented in (src, dst)
    '''
    (unit_src, unit_dst), (edge_id, _, _) = _unit_edges_from_block_edges(block_id, src_dst)
    # calculate unit-pair distances
    src_x, dst_x = X[unit_src], X[unit_dst]  # [Eu, 3]
    dist = torch.norm(src_x - dst_x, dim=-1)  # [Eu]
    dist = scatter_min(dist, edge_id)[0]  # [Eb]

    return dist


class KNNBatchEdgeConstructor(BatchEdgeConstructor):
    def __init__(self, k_neighbors, global_message_passing=True, global_node_id_vocab=[], delete_self_loop=True) -> None:
        super().__init__(global_node_id_vocab, delete_self_loop)
        self.k_neighbors = k_neighbors
        self.global_message_passing = global_message_passing

    def _construct_intra_edges(self, S, batch_id, segment_ids, **kwargs):
        all_intra_edges = super()._construct_intra_edges(S, batch_id, segment_ids)
        X, block_id = kwargs['X'], kwargs['block_id']
        # knn
        src_dst = all_intra_edges.T
        dist = _block_edge_dist(X, block_id, src_dst)
        intra_edges = _knn_edges(
            dist, src_dst, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return intra_edges
    
    def _construct_inter_edges(self, S, batch_id, segment_ids, **kwargs):
        all_inter_edges = super()._construct_inter_edges(S, batch_id, segment_ids)
        X, block_id = kwargs['X'], kwargs['block_id']
        # knn
        src_dst = all_inter_edges.T
        dist = _block_edge_dist(X, block_id, src_dst)
        inter_edges = _knn_edges(
            dist, src_dst, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inter_edges
    
    def _construct_global_edges(self, S, batch_id, segment_ids, **kwargs):
        if self.global_message_passing:
            return super()._construct_global_edges(S, batch_id, segment_ids, **kwargs)
        else:
            return None, None

    def _construct_seq_edges(self, S, batch_id, segment_ids, **kwargs):
        return None


# embedding of blocks (for proteins, it is residue).
class BlockEmbedding(nn.Module):
    '''
    [atom embedding + block embedding + atom position embedding]
    '''
    def __init__(self, num_block_type, num_atom_type, num_atom_position, embed_size, no_block_embedding=False):
        super().__init__()
        if not no_block_embedding:
            self.block_embedding = nn.Embedding(num_block_type, embed_size)
        self.no_block_embedding = no_block_embedding
        self.atom_embedding = nn.Embedding(num_atom_type, embed_size)
        # self.position_embedding = nn.Embedding(num_atom_position, embed_size)
    
    def forward(self, B, A, atom_positions, block_id):
        '''
        :param B: [Nb], block (residue) types
        :param A: [Nu], unit (atom) types
        :param atom_positions: [Nu], unit (atom) position encoding
        :param block_id: [Nu], block id of each unit
        '''
        atom_embed = self.atom_embedding(A) # + self.position_embedding(atom_positions) # FIXME: ablation
        if self.no_block_embedding:
            return atom_embed
        block_embed = self.block_embedding(B[block_id])
        return atom_embed + block_embed


if __name__ == '__main__':
    # test block edge distance
    X = torch.randn(12, 2, 3)
    block_id = torch.tensor([0,0,1,1,1,1,2,2,2,3,4,4], dtype=torch.long)
    src_dst = torch.tensor([[0, 1], [2,3], [1,3], [2,4]], dtype=torch.long)

    gt = []
    for src, dst in src_dst:
        src_X = torch.stack([x for x, b in zip(X, block_id) if b == src], dim=0)
        dst_X = torch.stack([x for x, b in zip(X, block_id) if b == dst], dim=0)
        dist = src_X.unsqueeze(1) - dst_X.unsqueeze(0)
        dist = torch.norm(dist, dim=-1)
        dist = torch.min(dist)
        gt.append(dist)

    for d, gt_d in zip(gt, _block_edge_dist(X, block_id, src_dst)):
        assert d == gt_d