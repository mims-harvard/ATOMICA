#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import sys
import math
import pickle
import argparse
from argparse import Namespace

import pandas as pd

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.convert import kd_to_dg
from utils.network import url_get
from utils.logger import print_log
from data.pdb_utils import Complex, Residue, VOCAB, Protein, Peptide
from data.split import main as split
from data.dataset import BlockGeoAffDataset



def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to the index file: INDEX_general_PP.2020')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdbs: PP')
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
    prot = Protein.from_pdb(pdb_file, include_all=True)
    # assert len(prot.get_chain_names()) == 1, pdb
    peptides = {}
    bases = ['DA', 'DG', 'DC', 'DT', 'RA', 'RG', 'RC', 'RU']

    rec_chains, lig_chains = [], []
    rec_seqs, lig_seqs = [], []
    for chain_name, chain in prot:
        split_point = None
        for i in range(len(chain.residues)):
            residue = chain.residues[len(chain.residues) - i - 1]
            if residue.symbol in bases:
                split_point = len(chain.residues) - i
                break
        peptides[chain_name] = Peptide(chain_name, chain.residues[:split_point])
        rec_chains.append(chain_name)
        rec_seqs.append(peptides[chain_name].get_seq())
        # assert split_point == len(chain) - 1, f'{split_point} {len(chain)}'
        mol_recs = []
        for res in chain.residues[split_point:]:
            if res.symbol != VOCAB.UNK:
                mol_recs.append(res)
                continue
            for atom_name, atom in res:
                mol_recs.append(Residue(
                    symbol=VOCAB.abrv_to_symbol(atom.element),
                    atom_map={ atom_name: atom },
                    _id=(len(mol_recs) + 1, ' '),
                    sidechain=[atom_name]
                ))
        if len(mol_recs) == 0:
            continue
        peptides[chain_name.lower()] = Peptide(chain_name.lower(), mol_recs)
        lig_chains.append(chain_name.lower())
        lig_seqs.append(peptides[chain_name.lower()].get_seq())

    cplx = Complex(pdb, peptides, rec_chains, lig_chains)       

    # write sequence
    item['seq_protein1'] = rec_seqs
    item['chains_protein1'] = rec_chains
    item['seq_protein2'] = lig_seqs
    item['chains_protein2'] = lig_chains

    # construct pockets
    interface1, _ = cplx.get_interacting_residues(dist_th=interface_dist_th)
    interface2 = []
    for c in lig_chains:
        for res in peptides[c]:
            interface2.append((c, res))
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
    for i, details in enumerate([rec_chains, lig_chains]):
        data = []
        for chain in details:
            chain_obj = cplx.get_chain(chain)
            if chain_obj is None:
                print_log(f'{chain} not in {pdb}: {cplx.get_chain_names()}. Skip this chain.', level='WARN')
                continue
            for residue in chain_obj:
                data.extend(residue_to_pd_rows(chain, residue))                
        item[f'atoms_protein{i + 1}'] = pd.DataFrame(data, columns=columns)

    # if pdb == '1ytf':
    #     aa
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
    database_out = os.path.join(args.out_dir, 'PDBbind.pkl')
    print_log(f'Obtained {len(processed_pdbbind)} data after filtering, saving to {database_out}...')
    with open(database_out, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())