#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import sys
import math
import pickle
import argparse
import numpy as np

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.convert import kd_to_dg
from utils.network import url_get
from utils.logger import print_log
from data.pdb_utils import Complex, Residue, VOCAB
from data.split import main as split
from data.dataset import BlockGeoAffDataset
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks_and_atom_array
from data.dataset import blocks_interface, blocks_to_data

NUCLEOTIDES = set(x[1] for x in VOCAB.bases)
AAS = set(x[1] for x in VOCAB.aas)

def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to the index file: INDEX_general_PN.2020')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdbs: PN')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=6.0,
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


def process_line(line, pdb_dir, interface_dist_th):

    if line.startswith('#'):  # annotation
        return ''

    item = {}
    line_split = re.split(r'\s+', line)
    pdb, kd = line_split[0], line_split[3]
    item['id'] = pdb  # pdb code, e.g. 1fc2
    item['resolution'] = line_split[1]  # resolution of the pdb structure, e.g. 2.80, another kind of value is NMR
    item['year'] = int(line_split[2])

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
    
    pdb_file = os.path.join(pdb_dir, pdb + '.ent.pdb')
    
    prot_blocks, prot_array, prot_residues = pdb_to_list_blocks_and_atom_array(pdb_file)
    prot_blocks = sum(prot_blocks, [])
    prot_residues = sum(prot_residues, [])
    lig_blocks, lig_array, lig_residues = pdb_to_list_blocks_and_atom_array(pdb_file, is_rna=True, is_dna=True)
    lig_blocks = sum(lig_blocks, [])
    lig_residues = sum(lig_residues, [])

    if len(prot_blocks) == 0 or len(lig_blocks) == 0:
        print_log(f'{pdb} has no protein or ligand', level='ERROR')
        return None

    blocks1, blocks2, indexes1, indexes2 = blocks_interface(prot_blocks, lig_blocks, interface_dist_th, return_indexes=True)
    if len(blocks1) == 0 or len(blocks2) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{pdb} has no interface', level='ERROR')
        return None
    
    pdb_indexes1 = [prot_residues[idx] for idx in indexes1]
    pdb_indexes2 = [lig_residues[idx] for idx in indexes2]
    data = blocks_to_data(blocks1, blocks2)
    item['data'] = data

    pdb_indexes_map = {}
    pdb_indexes_map.update(dict(zip(range(1,len(blocks1)+1), pdb_indexes1)))# map block index to pdb index, +1 for global block)
    pdb_indexes_map.update(dict(zip(range(len(blocks1)+2,len(blocks1)+len(blocks2)+2), pdb_indexes2)))# map block index to pdb index, +1 for global block)
    
    item["block_to_pdb_indexes"] = pdb_indexes_map
    item['atom_array1'] = prot_array
    item['atom_array2'] = lig_array
    return item


def main(args):

    # TODO: 1. preprocess PDBbind into json summaries and complex pdbs
    print_log('Preprocessing')
    with open(args.index_file, 'r') as fin:
        lines = fin.readlines()
    processed_pdbbind = []
    cnt = 0
    for i, line in enumerate(lines):
        item = process_line(line, args.pdb_dir, args.interface_dist_th)
        if item == '':  # annotation
            continue
        cnt += 1
        if item is None:
            continue
        processed_pdbbind.append(item)
        print_log(f'{item["id"]} succeeded, valid/processed={len(processed_pdbbind)}/{cnt}')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    database_out = os.path.join(args.out_dir, 'PDBbind_PN.pkl')
    print_log(f'Obtained {len(processed_pdbbind)} data after filtering, saving to {database_out}...')
    with open(database_out, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)

    # idx = list(range(len(processed_pdbbind)))
    # np.random.seed(0)
    # np.random.shuffle(idx)

    # train_len = int(len(idx) * 0.9)
    # train = [processed_pdbbind[i] for i in idx[:train_len]]
    # val = [processed_pdbbind[i] for i in idx[train_len:]]

    # pickle.dump(train, open(os.path.join(args.out_dir, 'train.pkl'), 'wb'))
    # pickle.dump(val, open(os.path.join(args.out_dir, 'valid.pkl'), 'wb'))
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())