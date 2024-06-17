#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import sys
import math
import pickle
import argparse
from Bio.PDB import PDBParser
import pandas as pd

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.convert import kd_to_dg
from utils.logger import print_log
from data.pdb_utils import Complex, Residue, VOCAB, Protein, Peptide
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.pdb_utils import Atom, VOCAB
from data.dataset import Block, blocks_interface, blocks_to_data
from data.converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks


def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to the index file: INDEX_general_PL.2020 / INDEX_refined_set.2020')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdbs: PL')
    parser.add_argument('--lig_dir', type=str, required=True,
                        help='Directory of pdbs: PL')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=6.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    parser.add_argument('--fragment', type=str, default=None, choices=['PS_300', 'PS_500'], help='Fragmentation of ligand')
    return parser.parse_args()

def process_line(line, pdb_dir, lig_dir, interface_dist_th, fragmentation_method):
    if line.startswith('#'):  # annotation
        return ''

    item = {}
    line_split = re.split(r'\s+', line)
    pdb, kd = line_split[0], line_split[3]
    item['id'] = pdb  # pdb code, e.g. 1fc2
    item['resolution'] = line_split[1]  # resolution of the pdb structure, e.g. 2.80, another kind of value is NMR
    item['year'] = int(line_split[2])
    item['ligand'] = ccd_code = line_split[6][1:-1]

    if (not kd.startswith('Kd')) and (not kd.startswith('Ki')):  # IC50 is very different from Kd and Ki, therefore discarded
        print_log(f'{pdb} not measured by Kd or Ki, dropped.', level='ERROR')
        return None
    
    if '=' not in kd:  # some data only provide a threshold, e.g. Kd<1nM, discarded
        print_log(f'{pdb} Kd only has threshold: {kd}', level='ERROR')
        return None

    kd = kd.split('=')[-1].strip()
    aff, unit = float(kd[:-2]), kd[-2:]
    if unit == 'mM':
        aff *= 1e-3
    elif unit == 'nM':
        aff *= 1e-9
    elif unit == 'uM':
        aff *= 1e-6
    elif unit == 'pM':
        aff *= 1e-12
    elif unit == 'fM':
        aff *= 1e-15
    else:
        return None   # unrecognizable unit
    
    # affinity data
    item['affinity'] = {
        'Kd': aff,
        'dG': kd_to_dg(aff, 25.0),   # regard as measured under the standard condition
        'neglog_aff': -math.log(aff, 10)  # pK = -log_10 (Kd)
    }
    
    pdb_file = os.path.join(pdb_dir, pdb + '_protein.pdb')
    ligand_file = os.path.join(lig_dir, pdb, pdb + '_ligand.mol2')

    prot_blocks, pdb_indexes = pdb_to_list_blocks(pdb_file, use_model=0, return_indexes=True)
    prot_blocks = sum(prot_blocks, [])
    pdb_indexes = sum(pdb_indexes, [])
    try:
        lig_blocks = mol2_to_blocks(ligand_file, fragment=fragmentation_method, molecule_type='protein' if '-mer' in ccd_code else 'small')
    except:
        if fragmentation_method != None:
            print_log(f'{pdb} ligand failed to process with fragment {fragmentation_method}', level='ERROR')
            lig_blocks = mol2_to_blocks(ligand_file, molecule_type='protein' if '-mer' in ccd_code else 'small')
        else:
            print_log(f'{pdb} ligand failed to process', level='ERROR')
            return None

    interface_blocks, _, interface_indexes, _ = blocks_interface(prot_blocks, lig_blocks, dist_th=interface_dist_th, return_indexes=True)
    data = blocks_to_data(interface_blocks, lig_blocks)

    pdb_indexes_map = dict(zip(range(1,len(interface_blocks)+1), [pdb_indexes[i] for i in interface_indexes])) # map block index to pdb index, +1 for global block)

    item['data'] = data
    item["block_to_pdb_indexes"] = pdb_indexes_map
    return item


def main(args):

    # TODO: 1. preprocess PDBbind into json summaries and complex pdbs
    print_log('Preprocessing')
    with open(args.index_file, 'r') as fin:
        lines = fin.readlines()
    processed_pdbbind = []
    cnt = 0
    for i, line in enumerate(lines):
        item = process_line(line, args.pdb_dir, args.lig_dir, args.interface_dist_th, args.fragment)
        if item == '':  # annotation
            continue
        cnt += 1
        if item is None:
            continue
        processed_pdbbind.append(item)
        print_log(f'{item["id"]} succeeded, valid/processed={len(processed_pdbbind)}/{cnt}')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    database_out = os.path.join(args.out_dir, 'PDBbind.pkl')
    print_log(f'Obtained {len(processed_pdbbind)} data after filtering, saving to {database_out}...')
    with open(database_out, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())