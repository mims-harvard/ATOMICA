#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from train import create_dataset
from data.pdb_utils import VOCAB
from src.edgeshaper import edgeshaper
from models.prediction_model import PredictionModel
from models.pretrain_model import DenoisePretrainModel
from trainers.abs_trainer import Trainer
from data.dataset import PDBDataset, DynamicBatchWrapper
import os
import numpy as np
from numpy.random import default_rng


def parse():
    parser = argparse.ArgumentParser(description='Run EdgeSHAPer on model outputs')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the test set')
    parser.add_argument('--task', type=str, default=None, choices=['PPA', 'PLA', 'LEP', 'PDBBind', 'NL', 'PLA_PS', 'LEP_PS', 'DDG', 'mdrdb'],
                        help='PPA: protein-protein affinity, ' + \
                             'PLA: protein-ligand affinity (small molecules), ' + \
                             'LEP: ligand efficacy prediction, ')
    parser.add_argument('--max_n_vertex_per_batch', type=int, default=5000, help='Max number of vertex per batch for running EdgeSHAPer inference')
    parser.add_argument('--num_monte_carlo_steps', type=int, default=10, help='Number of Monte Carlo steps for EdgeSHAPer inference')
    parser.add_argument('--fragment', type=str, default=None, help='fragmentation of small molecules')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers to use')
    parser.add_argument('--bottom_level', default=False, action='store_true', help='Use bottom level edges')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


def edgeshaper_batched(
    model, data, edges, edge_features, edge_mask, top_level=True, P=None, monte_carlo_steps=100, seed = 42, device = "cpu", max_n_vertex_per_batch=500,
):
    """ Compute Shapley values approximation for edge importance in GNNs
    For model outputs which are vectors it uses the L2 norm to compute the marginal contribution.
    Args:
        model (Torch.NN.Module): Torch GNN model used.
        nodes (tensor, [Nnodes, node_dim]): tensor containing node features of the graph to explain 
        edges (tensor, [2, Nedges]): edge index of the graph to be explained. 
        top_level (bool, optional): states if the edges are defined are the block level (True) 
            or if the edges are defined at the atom level (False)
        monte_carlo_steps (int): number of Monte Carlo sampling steps to perform.
        P (float, optional): probablity of an edge to exist in the random graph. Defaults to the original graph density.
        seed (float, optional): seed used for the random number generator.
        device (string, optional): states if using ```cuda``` or ```cpu```.
    Returns:
        list: list of Shapley values for the edges computed by EdgeSHAPer. The order of the edges is the same as in ```edges```.
    """
    
    edge_weight = edge_features

    rng = default_rng(seed = seed)
    model.eval()
    phi_edges = []

    num_nodes = data['B'].shape[0] if top_level else data['X'].shape[0]
    num_edges = edges.shape[1]
    print(edge_mask.sum())
    
    if P is None:
        max_num_edges = num_nodes*(num_nodes-1)
        graph_density = num_edges/max_num_edges
        P = graph_density

    for j in tqdm(range(num_edges), desc="Edge Shapley"):
        marginal_contrib = 0
        minus_edges, minus_edge_feat = [], []
        plus_edges, plus_edge_feat = [], []
        if edge_mask[j] == 0:
            phi_edges.append(0)
            continue

        # Sample random graphs
        for _ in range(monte_carlo_steps):
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]
            
            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(edges, dim = 1, index = retained_indices_plus)
            edge_weight_j_plus = torch.index_select(edge_weight, dim = 0, index = retained_indices_plus)
            plus_edges.append(E_j_plus)
            plus_edge_feat.append(edge_weight_j_plus)

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(edges, dim = 1, index = retained_indices_minus)
            edge_weight_j_minus = torch.index_select(edge_weight, dim = 0, index = retained_indices_minus)
            minus_edges.append(E_j_minus)
            minus_edge_feat.append(edge_weight_j_minus)

            # TODO: filter out corresponding edges in the bottom level if top level
            # TODO: to save time consider deleting only intermolecular edges?

        # Compute marginal contributions
        batch_size = max_n_vertex_per_batch//data['B'].shape[0]
        for start_idx in range(0, monte_carlo_steps, batch_size):
            end_idx = min(start_idx + batch_size, monte_carlo_steps)
            batch = PDBDataset.collate_fn([data for _ in range(start_idx, end_idx)])
            batch = Trainer.to_device(batch, device)
            if top_level:
                V_j_plus = model.infer(
                    batch, top_altered_edges=collate_altered_edges(num_nodes, [plus_edges[i] for i in range(start_idx, end_idx)]), 
                    top_altered_edge_attr=torch.cat([plus_edge_feat[i] for i in range(start_idx, end_idx)], dim=0)
                )
                V_j_minus = model.infer(
                    batch, top_altered_edges=collate_altered_edges(num_nodes, [minus_edges[i] for i in range(start_idx, end_idx)]), 
                    top_altered_edge_attr=torch.cat([minus_edge_feat[i] for i in range(start_idx, end_idx)], dim=0)
                )
            else:
                V_j_plus = model.infer(
                    batch, bottom_altered_edges=collate_altered_edges(num_nodes, [plus_edges[i] for i in range(start_idx, end_idx)]), 
                    bottom_altered_edge_attr=torch.cat([plus_edge_feat[i] for i in range(start_idx, end_idx)], dim=0)
                )
                V_j_minus = model.infer(
                    batch, bottom_altered_edges=collate_altered_edges(num_nodes, [minus_edges[i] for i in range(start_idx, end_idx)]), 
                    bottom_altered_edge_attr=torch.cat([minus_edge_feat[i] for i in range(start_idx, end_idx)], dim=0)
                )
            if V_j_plus.dim() == 1: # scalar outputs
                marginal_contrib += (V_j_plus - V_j_minus).sum().item()
            else: # vector outputs, e.g. for graph embeddings, use L2 norm
                marginal_contrib += (torch.norm(V_j_plus-V_j_minus, dim=-1)).sum().item()

        phi_edges.append(marginal_contrib/monte_carlo_steps)
        
    return phi_edges


def collate_altered_edges(num_nodes, list_of_altered_edges):
    collated_edges = []
    for idx, altered_edges in enumerate(list_of_altered_edges):
        collated_edges.append(altered_edges + num_nodes*idx)
    collated_edges = torch.cat(collated_edges, dim=1)
    assert collated_edges.max() < num_nodes*len(list_of_altered_edges), f"Edge index out of range, {collated_edges.max()} > {num_nodes*len(list_of_altered_edges)}"
    return collated_edges


def main(args):
    VOCAB.load_tokenizer(args.fragment)
    # load model
    model_ = torch.load(args.ckpt, map_location='cpu')
    if isinstance(model_, DenoisePretrainModel):
        model = PredictionModel.load_from_pretrained(args.ckpt)
    print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    test_set = create_dataset(args.task, args.test_set, fragment=args.fragment)
    test_loader = DataLoader(test_set, batch_size=1,
                             num_workers=args.num_workers,
                             collate_fn=test_set.collate_fn)
    items = test_set.indexes
    
    # save path
    if args.save_path is None:
        save_path = '.'.join(args.ckpt.split('.')[:-1]) + '_results.jsonl'
    else:
        save_path = args.save_path

    fout = open(save_path, 'w')

    idx = 0
    for batch in tqdm(test_loader, desc="Inference"):
        with torch.no_grad():
            # This line is for testing, can delete
            assert np.isclose(batch['label'].item(), items[idx]['label']), f"Mismatch between GT: {items[idx]['label']}, and label: {batch['label'].item()}"
            
            batch = Trainer.to_device(batch, device)
            edges, edge_attr, edge_mask = model.precalculate_edges(batch, top_level=not args.bottom_level)
            edges_explanations = edgeshaper_batched(
                model, batch, edges, edge_attr, edge_mask, top_level=not args.bottom_level, 
                monte_carlo_steps=args.num_monte_carlo_steps, 
                max_n_vertex_per_batch=args.max_n_vertex_per_batch,
                device="cuda:0")
            results = model.infer(batch)
            results = results.tolist()
            for pred_label in results:
                item_id = items[idx]['id']
                gt = items[idx]['label'] if 'label' in items[idx] else items[idx]['affinity']['neglog_aff']
                out_dict = {
                    'id': item_id,
                    'label': pred_label,
                    'task': args.task,
                    'gt': gt
                }
                fout.write(json.dumps(out_dict) + '\n')
                idx += 1
            
            SAVE_PATH = './output/'+ str(item_id) + "/"
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)

            with open(SAVE_PATH + str(item_id) + "_statistics.txt", "w+") as f:
                f.write("Interaction name: " + str(item_id) + "\n\n")
                f.write("GT Affinity: " + str(gt) + "\n")
                f.write("Predicted value: " + str(pred_label) + "\n\n")
                f.write("Shapley values for edges: \n\n")
                for e in range(len(edges_explanations)):
                    f.write("(" + str(edges[0][e].item()) + "," + str(edges[1][e].item()) + "): " + str(edges_explanations[e]) + "\n")
            print("Edgeshaper esults saved to: " + SAVE_PATH)
    fout.close()

if __name__ == '__main__':
    main(parse())