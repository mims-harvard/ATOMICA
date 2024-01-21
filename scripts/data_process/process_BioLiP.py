#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import pickle
import argparse

import numpy as np

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from data.pdb_utils import Residue, VOCAB
from data.dataset import blocks_interface, blocks_to_data
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.converter.sm_pdb_to_blocks import sm_pdb_to_blocks



def parse():
    parser = argparse.ArgumentParser(description='Process BioLiP benchmark of protein-ligand interaction for pre-training')
    parser.add_argument('--benchmark_dir', type=str, required=True,
                        help='Directory of the benchmark containing metadata and pdb_files')
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--fragment', default=None, choices=['PS_300', 'PS_500'], help='Use fragment-based representation of small molecules')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def residue_to_pd_rows(chain: str, residue: Residue):
    rows = []
    res_id, insertion_code = residue.get_id()
    resname = residue.real_abrv if hasattr(residue, 'real_abrv') else VOCAB.symbol_to_abrv(residue.get_symbol())
    for atom_name in residue.get_atom_names():
        atom = residue.get_atom(atom_name)
        if atom.element == 'H':  # skip hydrogen
            continue
        rows.append((
            chain, insertion_code, res_id, resname,
            atom.coordinate[0], atom.coordinate[1], atom.coordinate[2],
            atom.element, atom.name
        ))
    return rows


def process_one(data_idx, protein_file_name, ligand_file_name, benchmark_dir, interface_dist_th, fragment):

    item = {}
    item['id'] = data_idx  # pdb code, e.g. 1fc2
    item['affinity'] = { 'neglog_aff': -1.0 }

    prot_fname = os.path.join(benchmark_dir, 'receptor', protein_file_name)
    sm_fname = os.path.join(benchmark_dir, 'ligand', ligand_file_name)

    try:
        list_blocks1 = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{protein_file_name} protein parsing failed: {e}', level='ERROR')
        return None
    ligand_id = ligand_file_name.split("_")[1]
    if ligand_id in {"rna", "dna", "peptide"}:
        try:
            list_of_blocks2 = pdb_to_list_blocks(sm_fname)
            blocks2 = []
            for b in list_of_blocks2:
                blocks2.extend(b)
        except Exception as e:
            print_log(f'{ligand_file_name} ligand parsing failed: {e}', level='ERROR')
            return None
    else:
        try:
            blocks2 = sm_pdb_to_blocks(sm_fname, fragment=fragment)
        except Exception as e:
            print_log(f'{ligand_file_name} ligand parsing failed: {e}', level='ERROR')
            return None
    blocks1 = []
    for b in list_blocks1:
        blocks1.extend(b)

    # construct pockets
    blocks1, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
    if len(blocks1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{protein_file_name} and {ligand_file_name} has no interface', level='ERROR')
        return None

    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    item['data'] = data

    return item


def main(args):

    # TODO: 1. preprocess PDBbind into json summaries and complex pdbs
    with open(args.index_path, 'rb') as f:
        protein_file_names, ligand_file_names = pickle.load(f)
    data_idxs = list(range(len(protein_file_names)))

    print_log('Preprocessing')
    processed_data = {}
    cnt = 0
    for data_idx, protein_file_name, ligand_file_name in zip(data_idxs, protein_file_names, ligand_file_names):
        item = process_one(data_idx, protein_file_name, ligand_file_name, args.benchmark_dir, args.interface_dist_th, args.fragment is not None)

        if item == '':  # annotation
            continue
        cnt += 1
        if item is None:
            continue
        processed_data[data_idx] = item

        print_log(f'{item["id"]} succeeded, valid/processed={len(data_idxs)}/{cnt}')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    processed_data = [processed_data[key] for key in processed_data.keys()]

    from random import shuffle
    shuffle(processed_data)
    train_dataset = processed_data[:-10000]
    valid_dataset = processed_data[-10000:]

    database_out_train = os.path.join(args.out_dir, 'BioLiP_train.pkl')
    print_log(f'Obtained {len(processed_data)} data after filtering, saving to {database_out_train}...')
    with open(database_out_train, 'wb') as f:
        pickle.dump(train_dataset, f)

    database_out_valid = os.path.join(args.out_dir, 'BioLiP_valid.pkl')
    with open(database_out_valid, 'wb') as f:
        pickle.dump(valid_dataset, f)

    print_log('Finished!')

if __name__ == '__main__':
    main(parse())
