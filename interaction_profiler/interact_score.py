from copy import deepcopy
import numpy as np
import json
import torch
import os
import sys
from tqdm import tqdm

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..',
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.pdb_utils import VOCAB
from data.dataset import PDBDataset
from trainers.abs_trainer import Trainer
from models.prediction_model import PredictionModel


def mask_block(data, block_idx):
    data = deepcopy(data)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    data["B"][block_idx] = VOCAB.symbol_to_idx(VOCAB.MASK)
    block_start = sum(data["block_lengths"][:block_idx])
    block_end = block_start + data["block_lengths"][block_idx]
    data['block_lengths'][block_idx] = 1
    data['X'] = data['X'][:block_start] + [np.mean(data["X"][block_start:block_end], axis=0).tolist()] + data['X'][block_end:]
    data['A'] = data['A'][:block_start] + [VOCAB.get_atom_mask_idx()] + data['A'][block_end:]
    data['atom_positions'] = data['atom_positions'][:block_start] + [VOCAB.get_atom_pos_mask_idx()] + data['atom_positions'][block_end:]
    return data


def get_residue_model_scores(model, data):
    cos_distances = []
    block_idx = []
    for i in range(0, len(data['B'])):
        if data['B'][i] == VOCAB.symbol_to_idx(VOCAB.GLB):
            continue
        cos_distances.append(get_residue_model_score(model, data, i))
        block_idx.append(i)
    return cos_distances, block_idx

def get_residue_model_score(model, data, block_idx):
    with torch.no_grad():
        model.eval()
        masked_data = mask_block(data, block_idx)
        batch = PDBDataset.collate_fn([data, masked_data])
        batch = Trainer.to_device(batch, "cuda")
        output = model(batch["X"], batch["B"], batch["A"], batch['block_lengths'], batch['lengths'], batch['segment_ids'])
        cos_distance = torch.nn.functional.cosine_similarity(output.graph_repr[0], output.graph_repr[1], dim=-1).item()
    return cos_distance


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Get InteractScores")
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    parser.add_argument("--output_path", type=str, help="Output json file for importance scores")
    parser.add_argument("--model_ckpt", type=str, help="Path to the model checkpoint")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = PredictionModel.load_from_pretrained(args.model_ckpt)
    model = model.to("cuda")

    dataset = PDBDataset(args.data_path)
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        cos_distances, block_idx = get_residue_model_scores(model, dataset[i])
        output = {
            "id": dataset.indexes[i],
            "cos_distances": cos_distances,
            "block_idx": block_idx,
        }
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(output) + '\n')
            
    print("Finished!")
            
