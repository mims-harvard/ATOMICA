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
from data.pdb_utils import Complex, Residue, VOCAB
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


def _group_strategy1(proteins):
    # group by protein type
    protein_map = {}
    for _, prot_type, chains, chains2, seq in proteins:
        if prot_type not in protein_map:
            protein_map[prot_type] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_type]['chains'].extend(chains)
        protein_map[prot_type]['seq'] += seq
        protein_map[prot_type]['chains2'].extend(chains2)
    proteins = []
    for prot_type in protein_map:
        proteins.append(protein_map[prot_type])
    return proteins


def _group_strategy2(proteins):
    '''
    group by keywords in the protein name
    e.g.
    >1I51_1|Chains A, C|CASPASE-7 SUBUNIT P20|Homo sapiens (9606)
    >1I51_2|Chains B, D|CASPASE-7 SUBUNIT P11|Homo sapiens (9606)
    >1I51_3|Chains E, F|X-LINKED INHIBITOR OF APOPTOSIS PROTEIN|Homo sapiens (9606)
    1 and 2 should be grouped together
    '''
    keywords = ['SUBUNIT', 'RECEPTOR']
    protein_map = {}
    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        for keyword in keywords:
            if keyword in prot_name:
                prot_name = keyword
                break
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])
    return proteins

def _group_strategy3(proteins):
    '''
    group by antibody (the rest is antigen)
    e.g.
    >1YYM_1|Chains A[auth G], E[auth P]|Exterior membrane glycoprotein(GP120),Exterior membrane glycoprotein(GP120),Exterior membrane glycoprotein(GP120)|Human immunodeficiency virus 1 (11676)
    >1YYM_2|Chains B[auth L], F[auth Q]|antibody 17b light chain|Homo sapiens (9606)
    >1YYM_3|Chains C[auth H], G[auth R]|antibody 17b heavy chain|Homo sapiens (9606)
    >1YYM_4|Chains D[auth M], H[auth S]|F23, scorpion-toxin mimic of CD4|synthetic construct (32630)
    2 and 3 should be grouped, while 1 and 4 compose the antigen
    '''
    protein_map = {}
    antibody_detected = False
    for prot_name, _, _, _, _ in proteins:
        if 'ANTIBODY' in prot_name.upper():
            antibody_detected = True
            break
    if not antibody_detected:
        return proteins

    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        prot_name = 'ANTIBODY' if 'ANTIBODY' in prot_name else 'ANTIGEN'
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])

    return proteins


def _group_strategy4(proteins):
    '''
    group by modified name (get rid of pairing difference like alpha/beta, heavy/light
    e.g.
    >4CNI_1|Chains A, E[auth H]|OLOKIZUMAB HEAVY CHAIN, FAB PORTION|HOMO SAPIENS (9606)
    >4CNI_2|Chains B, F[auth L]|OLOKIZUMAB LIGHT CHAIN, FAB PORTION|HOMO SAPIENS (9606)
    >4CNI_3|Chains C, D|INTERLEUKIN-6|HOMO SAPIENS (9606)
    1 and 2 should be grouped
    '''
    protein_map = {}
    pair_keywords = ['HEAVY', 'LIGHT', 'ALPHA', 'BETA', 'VH', 'VL']
    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        for keyword in pair_keywords:
            prot_name = prot_name.replace(keyword, '')
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])

    return proteins


def _group_strategy5(proteins):
    '''
    special strategy for TCR-related cases
    e.g.
    >6MSS_1|Chain A|A11B8.2 NKT TCR alpha-chain|Mus musculus (10090)
    >6MSS_2|Chain B|A11B8.2 NKT TCR beta-chain|Mus musculus (10090)
    >6MSS_3|Chain C|Antigen-presenting glycoprotein CD1d1|Mus musculus (10090)
    >6MSS_4|Chain D|Beta-2-microglobulin|Mus musculus (10090)
    1 and 2 are TCR components, 3 and 4 are MHC components (3 is the presenting peptide, 4 is a infrastructure of MHC molecule)
    '''
    mhc_keywords = ['MHC', 'HLA', 'ANTIGEN-PRESENTING', 'BETA-2-MICROGLOBULIN', 'GLYCOPROTEIN']
    protein_map = {}
    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        if ('TCR' in prot_name) or ('RECEPTOR' in prot_name and (('T-CELL' in prot_name) or ('T CELL' in prot_name))):
            prot_name = 'TCR'
        else:
            for keyword in mhc_keywords:
                if keyword in prot_name:
                    prot_name = 'MHC'
                    break
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])

    return proteins


def parse_fasta(lines):
    assert len(lines) % 2 == 0, 'Number of fasta lines is not an even number!'
    proteins = []
    for i in range(0, len(lines), 2):
        details = lines[i].split('|')
        assert len(details) == 4
        prot_name = details[2]
        prot_type = details[3]
        chain_strs = details[1]
        if chain_strs.startswith('Chains '):
            chain_strs = chain_strs.replace('Chains ', '')
        elif chain_strs.startswith('Chain'):
            chain_strs = chain_strs.replace('Chain ', '')
        else:
            raise ValueError(f'Chain details has wrong format: {chain_strs}')
        chain_strs = chain_strs.split(', ')  # e.g. Chains B[auth H], F[auth K], or Chain A
        chains = [s[0] for s in chain_strs]
        chains2 = [s[-2] if len(s) > 1 else s[0] for s in chain_strs]  # multiple models
        seq = lines[i + 1]
        proteins.append((prot_name, prot_type, chains, chains2, seq))
    if len(proteins) > 2:  # receptor or ligand has been splitted into different sets of chains
        for strategy in [_group_strategy1, _group_strategy2, _group_strategy3, _group_strategy4, _group_strategy5]:
            grouped_proteins = strategy(proteins)
            if len(grouped_proteins) == 2:
                proteins = grouped_proteins
                break
    else:
        proteins = [{ 'chains': chains, 'chains2': chains2, 'seq': seq } \
                    for _, _, chains, chains2, seq in proteins]
    return proteins


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


    fasta = url_get(f'http://www.pdbbind.org.cn/FASTA/{pdb}.txt')
    #fasta = url_get(f'https://www.rcsb.org/fasta/entry/{pdb.upper()}/download')

    if fasta is None:
        print_log(f'Failed to fetch fasta for {pdb}!', level='ERROR')
        return None
    fasta = fasta.text.strip().split('\n')
    proteins = parse_fasta(fasta)
    if len(proteins) != 2:  # irregular fasta, cannot distinguish which chains composes one protein
        print_log(f'{pdb} has {len(proteins)} chain sets!', level='ERROR')
        return None
    
    pdb_file = os.path.join(pdb_dir, pdb + '.ent.pdb')
    cplx = Complex.from_pdb(pdb_file, proteins[0]['chains'], proteins[1]['chains'])
    
    # specify model
    chain_set_id = None
    for chain_set_name in ['chains', 'chains2']:
        all_in = True
        for c in proteins[0][chain_set_name] + proteins[1][chain_set_name]:
            if c not in cplx.peptides:
                all_in = False
                break
        if all_in:
            chain_set_id = chain_set_name
            break
    if chain_set_id is None:
        print_log(f'Chains {proteins[0]["chains"] + proteins[1]["chains"]} have at least one missing in {pdb}: {cplx.get_chain_names()}', level='ERROR')
        return None
    proteins[0]['chains'] = proteins[0][chain_set_id]
    proteins[1]['chains'] = proteins[1][chain_set_id]
    cplx.receptor_chains = proteins[0]['chains']
    cplx.ligand_chains = proteins[1]['chains']

    # write sequence
    item['seq_protein1'] = proteins[0]['seq']
    item['chains_protein1'] = proteins[0]['chains']
    item['seq_protein2'] = proteins[1]['seq']
    item['chains_protein2'] = proteins[1]['chains']

    for chain in proteins[0]['chains'] + proteins[1]['chains']:
        if chain not in cplx.peptides:
            print_log(f'Chain {chain} missing in {pdb}: {cplx.get_chain_names()}', level='ERROR')
            return None
    
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
    for i, details in enumerate(proteins):
        data = []
        for chain in details['chains']:
            chain_obj = cplx.get_chain(chain)
            if chain_obj is None:
                print_log(f'{chain} not in {pdb}: {cplx.get_chain_names()}. Skip this chain.', level='WARN')
                continue
            for residue in chain_obj:
                data.extend(residue_to_pd_rows(chain, residue))                
        item[f'atoms_protein{i + 1}'] = pd.DataFrame(data, columns=columns)

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