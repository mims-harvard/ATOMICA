import os
import sys
import torch
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import random
import pickle
import pandas as pd
import json

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
from data.dataset import blocks_interface, blocks_to_data, BlockGeoAffDataset
from data.pdb_utils import VOCAB


def process_datum(rec_file, lig_file, fragmentation_method):
    """
    Returns: data, pocket_residues [(residue_number, residue_abrv)]
    """
    list_of_blocks1, indexes = pdb_to_list_blocks(rec_file, return_indexes=True)
    try:
        blocks2 = mol2_to_blocks(lig_file, fragment=fragmentation_method)
    except Exception as e:
        blocks2 = mol2_to_blocks(lig_file, fragment=None)
    blocks1, _, indexes1, _ = blocks_interface(list_of_blocks1[0], blocks2, dist_th=6, return_indexes=True)

    data = blocks_to_data(blocks1, blocks2)
    pocket_residues = [res for i, res in enumerate(indexes[0]) if i in indexes1.tolist()]
    pocket_residues = [(int(res.split("_")[-1]), VOCAB.symbol_to_abrv(b.symbol)) for b,res in zip(blocks1, pocket_residues)]
    return data, pocket_residues


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


def get_plip_residues(pdb_file):
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file) # Load the PDB file into PLIP class
    my_mol.analyze()
    unique_res = set()
    for interaction in my_mol.interaction_sets.keys():
        lig, ch, _ = interaction.split(":")
        for itype in my_mol.interaction_sets[interaction].all_itypes:
            itype_type = itype.__class__.__name__
            if itype_type == "waterbridge":
                continue
            unique_res.add((itype.reschain, itype.resnr, itype.restype, itype_type))
    return unique_res


def get_plip_residues_PL(rec_file, lig_file):
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
    my_mol = PDBComplex()
    my_mol.load_pdb(tmp_file) # Load the PDB file into PLIP class
    my_mol.analyze()
    unique_res = set()
    for interaction in my_mol.interaction_sets.keys():
        lig, ch, _ = interaction.split(":")
        for itype in my_mol.interaction_sets[interaction].all_itypes:
            itype_type = itype.__class__.__name__
            if itype_type == "waterbridge":
                continue
            unique_res.add((itype.resnr, itype.restype, itype_type))
    os.remove(tmp_file)
    return unique_res


def compare_residues(data, pocket_residues, plip_residues, model, topk):
    cos_distances, block_idx = get_residue_model_scores(model, data)
    
    topk_model_res = []
    for _, i in sorted(zip(cos_distances, block_idx))[:topk]:
        topk_model_res.append(pocket_residues[i-1])
        assert pocket_residues[i-1][-1] == VOCAB.idx_to_abrv(data['B'][i]), "block type mismatch"

    topk_jaccards = []
    rand_jaccards = []
    for k in range(1, topk+1):
        rand_model_res = random.sample(pocket_residues, k)
        topk_jaccard = jaccard_index(topk_model_res[:k], plip_residues)
        rand_jaccard = jaccard_index(rand_model_res, plip_residues)
        topk_jaccards.append(topk_jaccard)
        rand_jaccards.append(rand_jaccard)
    return topk_jaccards, rand_jaccards, cos_distances

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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run mutagenesis case study")
    parser.add_argument("--task", type=str, default="PN", help="Task to run mutagenesis on, options: PL, PN")
    parser.add_argument("--topk", type=int, default=5, help="Top k residues to consider")
    parser.add_argument("--out_dir", type=str, help="Output folder for importance scores")
    parser.add_argument("--model_ckpt", type=str, help="Path to the model checkpoint")
    return parser.parse_args()


def save_json(out_json, out_dir):
    pdb_id = out_json["pdb_id"]
    with open(f"{out_dir}/{pdb_id}.json", "w") as f:
        json.dump(out_json, f)
                  

if __name__ == '__main__':
    args = parse_args()
    task = args.task
    topk = args.topk

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = PredictionModel.load_from_pretrained(args.model_ckpt)
    model = model.to("cuda")

    if task == "PL":
        data_dir = "/n/holyscratch01/mzitnik_lab/afang/data/pdbbind/PDBbind_v2020_PL_refined/refined-set"
        pdb_ids = [pdb_id for pdb_id in os.listdir(data_dir) if len(pdb_id)==4]

        for pdb_id in tqdm(pdb_ids):
            rec_file = f"{data_dir}/{pdb_id}/{pdb_id}_pocket.pdb"
            lig_file = f"{data_dir}/{pdb_id}/{pdb_id}_ligand.mol2"
            print(f"Processing {pdb_id} ...")

            out_json = {"pdb_id": pdb_id, "pocket_residues": [], "plip_residues": [], "mutagenesis_scores": [], f"top{topk}_jaccard": None, f"rand{topk}_jaccard": None}

            data, pocket_residues = process_datum(rec_file, lig_file, model.fragmentation_method)
            out_json["pocket_residues"] = pocket_residues


            if len(pocket_residues) <= topk:
                print(f"pdb_id: {pdb_id} has <= {topk} residues in the pocket, num_residues={len(pocket_residues)}. Skipping...")
                save_json(out_json, args.out_dir)
                continue
            if len(set(pocket_residues)) != len(pocket_residues):
                print(f"pdb_id: {pdb_id} has duplicate residues {pocket_residues}. Skipping...")
                save_json(out_json, args.out_dir)
                continue

            try:
                plip_residues_with_type = get_plip_residues_PL(rec_file, lig_file)
                plip_residues = set([res[:-1] for res in plip_residues_with_type])
                out_json["plip_residues"] = list(plip_residues_with_type)
            except Exception as e:
                print(f"Error processing pdb_id: {pdb_id}, error: {e}. Skipping...")
                save_json(out_json, args.out_dir)
                continue
            
            topk_jaccard, rand_jaccard, scores = compare_residues(data, pocket_residues, plip_residues, model, topk)
            out_json["mutagenesis_scores"] = scores
            out_json[f"top{topk}_jaccard"] = topk_jaccard
            out_json[f"rand{topk}_jaccard"] = rand_jaccard
            save_json(out_json, args.out_dir)

    elif task in ["PN", "NL"]:
        data_file = f"/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/{task}/processed/PDBbind.pkl"
        data_dir = f"/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/{task}/"
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)
        dataset = BlockGeoAffDataset(data_file)

        pdb_ids = []
        assert len(dataset) == len(raw_data), "dataset length mismatch"
        for data_idx in tqdm(range(len(dataset))):
            assert raw_data[data_idx]["id"] == dataset.indexes[data_idx]['id'], "id mismatch"
            pdb_id = raw_data[data_idx]["id"]
            print(f"Processing {pdb_id} ...")

            out_json = {"pdb_id": pdb_id, "pocket_residues": [], "plip_residues": [], "mutagenesis_scores": [], f"top{topk}_jaccard": None, f"rand{topk}_jaccard": None}

            pocket_residues = list(raw_data[data_idx]['atoms_interface1'][['chain', 'residue', 'resname']].drop_duplicates().itertuples(index=False, name=None))
            out_json["pocket_residues"] = pocket_residues
            data = dataset[data_idx]
            if len(pocket_residues) < topk:
                print(f"data_idx: {data_idx} has less than k={topk} residues in the pocket, num_residues={len(pocket_residues)}. Skipping...")
                save_json(out_json, args.out_dir)
                continue
            if len(set(pocket_residues)) != len(pocket_residues):
                print(f"data_idx: {data_idx} has duplicate residues {pocket_residues}. Skipping...")
                save_json(out_json, args.out_dir)
                continue
            try:
                plip_residues_with_type = get_plip_residues(f"{data_dir}/{pdb_id}.ent.pdb")
                plip_residues = set([res[:-1] for res in plip_residues_with_type])
            except Exception as e:
                print(f"Error processing pdb_id: {pdb_id}, error: {e}. Skipping...")
                save_json(out_json, args.out_dir)
                continue

            filtered_plip_residues = set(plip_residues).intersection(set(pocket_residues)) # there may be multiple interfaces in the PN files, only keep the interface the data has
            out_json["plip_residues"] = [x for x in plip_residues_with_type if x[:-1] in filtered_plip_residues]
            if len(filtered_plip_residues) == 0:
                print(f"pdb_id: {pdb_id} no PLIP residues in the pocket. Skipping...")
                save_json(out_json, args.out_dir)
                continue
            
            try:
                topk_jaccard, rand_jaccard, scores = compare_residues(data, pocket_residues, filtered_plip_residues, model, topk)
            except Exception as e:
                print(f"Error computing jaccard similarity metric for pdb_id: {pdb_id}, error: {e}. Skipping...")
                save_json(out_json, args.out_dir)
                continue
            out_json["mutagenesis_scores"] = scores
            out_json[f"top{topk}_jaccard"] = topk_jaccard
            out_json[f"rand{topk}_jaccard"] = rand_jaccard
            save_json(out_json, args.out_dir)