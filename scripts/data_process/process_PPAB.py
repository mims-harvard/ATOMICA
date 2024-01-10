#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import math
import pickle
import argparse

import pandas as pd

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from utils.convert import kd_to_dg
from utils.network import fetch_from_pdb
from data.pdb_utils import Complex, Protein
from scripts.data_process.process_PDBbind_PP import residue_to_pd_rows


def parse():
    parser = argparse.ArgumentParser(description='Process protein-protein binding affinity data from the structural benchmark')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to the index file: PPAB_V2.csv')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdbs: benchmark5.5')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=6.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()
    

def process_line(line, pdb_dir, interface_dist_th):
    line = line.split(',')
    assert len(line) == 14
    pdb, (lig_chains, rec_chains) = line[0][:4], line[0][5:].split(':')  # e.g. 1A2K_C:AB
    try:
        Kd, dG = float(line[7]), float(line[8])
    except ValueError as e:
        print_log(f'{pdb} Kd not correct: {e}.', level='ERROR')
        return None
    if dG > 0:
        dG = kd_to_dg(Kd)

    lig_chains, rec_chains = list(lig_chains), list(rec_chains)
    item = {
            'id': pdb.lower() + line[0][4:],
            'class': line[1],
            'I_rmsd': float(line[9])
    }
    item['affinity'] = {
            'Kd': Kd,
            'dG': dG,
            'neglog_aff': -math.log(Kd, 10),
    }
    assert item['I_rmsd'] > 0

    lig_path = os.path.join(pdb_dir, 'structures', f'{pdb}_l_b.pdb')
    rec_path = os.path.join(pdb_dir, 'structures', f'{pdb}_r_b.pdb')
    if os.path.exists(lig_path) and os.path.exists(rec_path):
        lig_prot = Protein.from_pdb(lig_path)
        rec_prot = Protein.from_pdb(rec_path)
        for c in rec_chains:
            all_hit = True
            if c not in rec_prot.peptides:
                all_hit = False
                break
        if not all_hit:
            lig_chains, rec_chains = rec_chains, lig_chains
        for c in rec_chains:
            if c not in rec_prot.peptides:
                print_log(f'Chain {c} not in {pdb} receptor: {rec_prot.get_chain_names()}', level='ERROR')
                return None
        for c in lig_chains:
            if c not in lig_prot.peptides:
                print_log(f'Chain {c} not in {pdb} ligand: {lig_prot.get_chain_names()}', level='ERROR')
                return None
        peptides = lig_prot.peptides
        peptides.update(rec_prot.peptides)
        cplx = Complex(item['id'], peptides, rec_chains, lig_chains)
    else:
        print_log(f'{pdb} not found in the local files, fetching it from remote server.', level='WARN')
        cplx_path = os.path.join(pdb_dir, 'structures', f'{pdb}_cplx.pdb')
        fetch_from_pdb(pdb.upper(), cplx_path)
        cplx = Complex.from_pdb(cplx_path, rec_chains, lig_chains)


    # Protein1 is receptor, protein2 is ligand
    item['seq_protein1'] = ''.join([cplx.get_chain(c).get_seq() for c in rec_chains])
    item['chains_protein1'] = rec_chains
    item['seq_protein2'] = ''.join([cplx.get_chain(c).get_seq() for c in lig_chains])
    item['chains_protein2'] = lig_chains

    # construct pockets
    interface1, interface2 = cplx.get_interacting_residues(dist_th=interface_dist_th)
    if len(interface1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{pdb} has no interface', level='ERROR')
        return None
    columns = ['chain', 'insertion_code', 'residue', 'resname', 'x', 'y', 'z', 'element', 'name']
    for i, interface in enumerate([interface1, interface2]):
        data = []
        for chain, residue in interface:
            data.extend(residue_to_pd_rows(chain, residue))
        item[f'atoms_interface{i + 1}'] = pd.DataFrame(data, columns=columns)
            
    # construct DataFrame of coordinates
    for i, chains in enumerate([rec_chains, lig_chains]):
        data = []
        for chain in chains:
            chain_obj = cplx.get_chain(chain)
            if chain_obj is None:
                print_log(f'{chain} not in {pdb}: {cplx.get_chain_names()}. Skip this chain.', level='WARN')
                continue
            for residue in chain_obj:
                data.extend(residue_to_pd_rows(chain, residue))                
        item[f'atoms_protein{i + 1}'] = pd.DataFrame(data, columns=columns)

    return item



def main(args):
    with open(args.index_file, 'r') as fin:
        lines = fin.readlines()
    lines = lines[1:]  # the first one is head

    print_log('Preprocessing')
    processed_ppab = []
    cnt = 0
    for i, line in enumerate(lines):
        item = process_line(line, args.pdb_dir, args.interface_dist_th)
        cnt += 1
        if item is None:
            continue
        processed_ppab.append(item)
        print_log(f'{item["id"]} succeeded, valid/processed={len(processed_ppab)}/{cnt}')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    database_out = os.path.join(args.out_dir, 'PPAB_V2.pkl')
    print_log(f'Obtained {len(processed_ppab)} data after filtering, saving to {database_out}...')
    with open(database_out, 'wb') as fout:
        pickle.dump(processed_ppab, fout)
    print_log('Binary file saved.')

    th, name = [(-1, 1e6), (-1, 0.5), (0.5, 1.5), (1.5, 1e6)], ['PPAB_V2_all', 'PPAB_V2_rigid', 'PPAB_V2_medium', 'PPAB_V2_flexible']

    # these are brief summaries for evaluations
    for (l, h), n in zip(th, name):
        items = [ {'id': item['id'], 'affinity': item['affinity'] } for item in processed_ppab if item['I_rmsd'] >= l and item['I_rmsd'] < h]
        out_path = os.path.join(args.out_dir, n + '.jsonl')
        with open(out_path, 'w') as fout:
            for item in items:
                s = json.dumps(item)
                fout.write(s + '\n')
        print_log(f'Saved {len(items)} entries as summary to {out_path}.')

    print_log('Finished!')

if __name__ == '__main__':
    main(parse())