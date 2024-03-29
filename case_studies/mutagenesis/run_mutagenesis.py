import os
import sys
import torch
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import random
import pandas as pd

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from plip.structure.preparation import PDBComplex
from data.dataset import PDBDataset
from data.pdb_utils import VOCAB
from models.prediction_model import PredictionModel
from trainers.abs_trainer import Trainer
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.dataset import blocks_interface, blocks_to_data
from data.pdb_utils import VOCAB


def process_datum(rec_file, lig_file, fragmentation_method):
    list_of_blocks1, indexes = pdb_to_list_blocks(rec_file, return_indexes=True)
    try:
        blocks2 = mol2_to_blocks(lig_file, fragment=fragmentation_method)
    except Exception as e:
        blocks2 = mol2_to_blocks(lig_file, fragment=None)
    blocks1, _, indexes1, _ = blocks_interface(list_of_blocks1[0], blocks2, dist_th=6, return_indexes=True)
    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    pocket_residues = [res for i, res in enumerate(indexes[0]) if i in indexes1.tolist()]
    pocket_residues = [(int(res.split("_")[-1]), VOCAB.symbol_to_abrv(b.symbol)) for b,res in zip(blocks1, pocket_residues)]
    return data, pocket_residues


def mask_block(data, block_idx):
    data = deepcopy(data)
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
        if data["segment_ids"][i] == 1: # ignore ligand atoms
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


def get_plip_residues(rec_file, lig_file):
    my_mol = PDBComplex()
    file_dir = os.path.dirname(rec_file)
    tmp_file = f"{file_dir}/tmp.pdb"
    with open(rec_file, "r") as f:
        with open(tmp_file, "w") as f2:
            for line in f:
                if line == "END\n" or line == "\n":
                    continue
                f2.write(line)

    # Load the .mol2 file
    mol = Chem.MolFromMol2File(lig_file)

    # Convert to .pdb format
    pdb_block = Chem.MolToPDBBlock(mol)
    pdb_block = pdb_block.split('\n')[1:]

    with open(tmp_file, "a") as f2:
        for line in pdb_block:
            f2.write(line + "\n")
    my_mol.load_pdb(tmp_file) # Load the PDB file into PLIP class
    my_mol.analyze()
    unique_res = set()
    for interaction in my_mol.interaction_sets.keys():
        lig, ch, _ = interaction.split(":")
        for itype in my_mol.interaction_sets[interaction].all_itypes:
            itype_type = itype.__class__.__name__
            if itype_type == "waterbridge":
                continue
            unique_res.add((itype.resnr, itype.restype))
    os.remove(tmp_file)
    return unique_res


def compare_residues(rec_file, lig_file, model, fragmentation_method, topk):
    data, pocket_residues = process_datum(rec_file, lig_file, fragmentation_method)
    if len(pocket_residues) < topk:
        pdb_id = os.path.basename(os.path.dirname(rec_file))
        print(f"pdb_id: {pdb_id} has less than k={topk} residues in the pocket, num_residues={len(pocket_residues)}. Skipping...")
        return None, None
    if len(set(pocket_residues)) != len(pocket_residues):
        pdb_id = os.path.basename(os.path.dirname(rec_file))
        print(f"pdb_id: {pdb_id} has duplicate residues {pocket_residues}. Skipping...")
        return None, None
    try:
        plip_residues = get_plip_residues(rec_file, lig_file)
    except Exception as e:
        pdb_id = os.path.basename(os.path.dirname(rec_file))
        print(f"Error processing pdb_id: {pdb_id}, error: {e}. Skipping...")
        return None, None

    cos_distances, block_idx = get_residue_model_scores(model, data)
    
    topk_model_res = []
    for _, i in sorted(zip(cos_distances, block_idx))[:topk]:
        topk_model_res.append(pocket_residues[i-1])
        assert pocket_residues[i-1][-1] == VOCAB.idx_to_abrv(data['B'][i]), "block type mismatch"

    rand_model_res = random.sample(pocket_residues, topk)

    topk_jaccard = jaccard_index(topk_model_res, plip_residues)
    rand_jaccard = jaccard_index(rand_model_res, plip_residues)
    return topk_jaccard, rand_jaccard



def jaccard_index(set1, set2):
    if type(set1) == list:
        set1_ = set(set1)
        assert len(set1) == len(set1_), "list1 contains duplicates"
        set1 = set1_
    if type(set2) == list:
        set2_ = set(set2)
        assert len(set2) == len(set2_), "list2 contains duplicates"
        set2 = set2_
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


if __name__ == '__main__':
    model = PredictionModel.load_from_pretrained("/n/holyscratch01/mzitnik_lab/afang/GET/pretrain/models/InteractNN-torsion/version_21/checkpoint/epoch3_step88430.ckpt")
    model = model.to("cuda")

    data_dir = "/n/holyscratch01/mzitnik_lab/afang/data/pdbbind/PDBbind_v2020_PL_refined/refined-set"
    pdb_ids = [pdb_id for pdb_id in os.listdir(data_dir) if len(pdb_id)==4]

    topk = 10

    topk_jaccards = []
    rand_jaccards = []
    for pdb_id in tqdm(pdb_ids):
        rec_file = f"{data_dir}/{pdb_id}/{pdb_id}_pocket.pdb"
        lig_file = f"{data_dir}/{pdb_id}/{pdb_id}_ligand.mol2"

        topk_jaccard, rand_jaccard = compare_residues(rec_file, lig_file, model, model.fragmentation_method, topk)
        topk_jaccards.append(topk_jaccard)
        rand_jaccards.append(rand_jaccard)
    
    df = pd.DataFrame({"pdb_id":pdb_ids, "topk_jaccards": topk_jaccards, "rand_jaccards": rand_jaccards})
    df.to_csv(f"./mutagenesis_scores_topk_{topk}.csv", index=False)
    print(f"Saved results to ./mutagenesis_scores_topk_{topk}.csv")
