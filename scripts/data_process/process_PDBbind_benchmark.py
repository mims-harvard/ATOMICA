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



def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind benchmark of protein-ligand interaction')
    parser.add_argument('--benchmark_dir', type=str, required=True,
                        help='Directory of the benchmark containing metadata and pdb_files')
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


def process_one(pdb_id, label, benchmark_dir, interface_dist_th, fragment):

    item = {}
    item['id'] = pdb_id  # pdb code, e.g. 1fc2
    item['affinity'] = { 'neglog_aff': label }
    pdb_dir = os.path.join(benchmark_dir, 'pdb_files')

    prot_fname = os.path.join(pdb_dir, pdb_id, pdb_id + '.pdb')
    sm_fname = os.path.join(pdb_dir, pdb_id, f'{pdb_id}_ligand.mol2')

    try:
        list_blocks1 = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{pdb_id} protein parsing failed: {e}', level='ERROR')
        return None
    try:
        blocks2 = mol2_to_blocks(sm_fname, fragment=fragment)
    except Exception as e:
        print_log(f'{pdb_id} ligand parsing failed: {e}', level='ERROR')
        return None
    blocks1 = []
    for b in list_blocks1:
        blocks1.extend(b)

    # construct pockets
    blocks1, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
    if len(blocks1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{pdb_id} has no interface', level='ERROR')
        return None

    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    item['data'] = data

    return item


def main(args):

    # TODO: 1. preprocess PDBbind into json summaries and complex pdbs
    labels = json.load(open(os.path.join(args.benchmark_dir, 'metadata', 'affinities.json'), 'r'))
    print_log('Preprocessing')
    processed_pdbbind = {}
    cnt = 0
    for pdb_id in labels:
        item = process_one(pdb_id, labels[pdb_id], args.benchmark_dir, args.interface_dist_th, args.fragment is not None)
        if item == '':  # annotation
            continue
        cnt += 1
        if item is None:
            continue
        processed_pdbbind[pdb_id] = item
        print_log(f'{item["id"]} succeeded, valid/processed={len(processed_pdbbind)}/{cnt}')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for split in ['identity30', 'identity60', 'scaffold']:
        split_info = json.load(open(os.path.join(args.benchmark_dir, 'metadata', f'{split}_split.json'), 'r'))
        out_dir = os.path.join(args.out_dir, split)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for name in ['train', 'valid', 'test']:
            data_out_path = os.path.join(out_dir, name + '.pkl')
            data_out = []
            miss_cnt = 0
            for pdb_id in split_info[name]:
                if pdb_id in processed_pdbbind:
                    data_out.append(processed_pdbbind[pdb_id])
                else:
                    miss_cnt += 1
            print_log(f'Obtained {len(data_out)}, missing {miss_cnt}, saving to {data_out_path}...')
            with open(data_out_path, 'wb') as fout:
                pickle.dump(data_out, fout)

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
