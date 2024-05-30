import numpy as np
import pandas as pd
import os
import sys
import pickle
from tqdm import tqdm

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
sys.path.append(PROJ_DIR)

from data.pdb_utils import VOCAB, Atom
from data.dataset import Block
from data.dataset import blocks_interface, blocks_to_data
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks

def group_chains(list_chain_blocks, list_chain_pdb_indexes, group1, group2):
    group1_chains = []
    group2_chains = []
    group1_indexes = []
    group2_indexes = []
    for chain_blocks, chain_pdb_indexes in zip(list_chain_blocks, list_chain_pdb_indexes):
        if chain_pdb_indexes[0].split("_")[0] in group1:
            group1_chains.extend(chain_blocks)
            group1_indexes.extend(chain_pdb_indexes)
        elif chain_pdb_indexes[0].split("_")[0] in group2:
            group2_chains.extend(chain_blocks)
            group2_indexes.extend(chain_pdb_indexes)
    return [group1_chains, group2_chains], [group1_indexes, group2_indexes]

def process_wt(pdb_file, dist_th, group1_chains, group2_chains):
    wt_blocks, pdb_indexes = pdb_to_list_blocks(pdb_file, return_indexes=True)
    if len(wt_blocks) != 2:
        wt_blocks, pdb_indexes = group_chains(wt_blocks, pdb_indexes, group1_chains, group2_chains)
    wt_blocks = blocks_interface(wt_blocks[0], wt_blocks[1], dist_th)
    wt_data = blocks_to_data(*wt_blocks)
    return wt_data

def process_mut(pdb_file, dist_th, group1_chains, group2_chains, mut_sites):
    # mut_sites [(mut_ch, mut_site, wt_aa, mut_aa)]
    mut_blocks, pdb_indexes = pdb_to_list_blocks(pdb_file, return_indexes=True)
    if len(mut_blocks) != 2:
        mut_blocks, pdb_indexes = group_chains(mut_blocks, pdb_indexes, group1_chains, group2_chains)
    
    mut_locs = []
    for mut_ch, mut_site, wt_aa, mut_aa in mut_sites:
        mut_ch_i, mut_site_i = None, None
        for ch_i, ch in enumerate(pdb_indexes):
            for res_i, res in enumerate(ch):
                if res == f"{mut_ch}_{mut_site}":
                    assert mut_blocks[ch_i][res_i].symbol == wt_aa, f"Invalid mutation site, wild type mismatch. pdb={pdb_file}"
                    mut_ch_i, mut_site_i = ch_i, res_i
                    mut_locs.append((mut_ch_i, mut_site_i, mut_aa))
                    break
    assert len(mut_locs) == len(mut_sites), "Not all mutation sites found"
    
    if mut_ch_i is None or mut_site_i is None:
        raise ValueError(f"Invalid mutation site, mutation chain and site not found. pdb={pdb_file}")
    
    # Mask the mutation site, keep only the backbone atoms
    for mut_ch_i, mut_site_i, mut_aa in mut_locs:
        backbone_atoms = []
        for atom in mut_blocks[mut_ch_i][mut_site_i].units:
            if (atom.element, atom.pos_code) in [("N", ""), ("C", "A"), ("C", ""), ("O", "")]:
                backbone_atoms.append(atom)
        if len(backbone_atoms) != 4:
            raise ValueError(f"Invalid mutation site, backbone atoms not found. pdb={pdb_file}")
        mut_blocks[mut_ch_i][mut_site_i] = Block(symbol=mut_aa, units=backbone_atoms)

    mut_blocks1, mut_blocks2, mut_indexes1, mut_indexes2 = blocks_interface(mut_blocks[0], mut_blocks[1], dist_th, return_indexes=True)

    # Check if mutation site is in the interface
    mut_block_indexes = []
    for mut_ch_i, mut_site_i, _ in mut_locs:
        if mut_ch_i == 0:
            if mut_site_i not in mut_indexes1:
                raise ValueError(f"Invalid mutation site, mutated residue not found in the interface. pdb={pdb_file}")
            else:
                mut_block_indexes.append(np.where(mut_indexes1 == mut_site_i)[0][0] + 1) # +1 for global block
        elif mut_ch_i == 1:
            if mut_site_i not in mut_indexes2:
                raise ValueError(f"Invalid mutation site, mutated residue not found in the interface. pdb={pdb_file}")
            else:
                mut_block_indexes.append(np.where(mut_indexes2 == mut_site_i)[0][0] + 2 + len(mut_blocks1)) # +2 for global block of interface 1 and interface 2
        
    mut_data = blocks_to_data(mut_blocks1, mut_blocks2)
    return mut_data, mut_block_indexes

def process_data(skempi_csv, pdb_dir, dist_th=8.0):
    skempi_df = pd.read_csv(skempi_csv, sep=';')
    data = []
    for _, row in tqdm(skempi_df.iterrows(), total=len(skempi_df), desc="Processing SKEMPI data"):
        try:
            ddg = -np.log10(float(row['Affinity_mut (M)'])) + np.log10(float(row['Affinity_wt (M)']))
            wt_ba = -np.log10(float(row['Affinity_wt (M)']))
            mt_ba = -np.log10(float(row['Affinity_mut (M)']))
        except ValueError:
            print(f"WARNING: Invalid ddG value. pdb={row['#Pdb']}, ddG={row['Affinity_mut (M)']}, {row['Affinity_wt (M)']}")
            continue
        pdb_id, group1_chains, group2_chains = row['#Pdb'].split('_')
        pdb_file = f"{pdb_dir}/{pdb_id}.pdb"
        if not os.path.exists(pdb_file):
            raise ValueError(f"Invalid PDB file. File not found. pdb={pdb_file}")
        
        mutation_site_list = row['Mutation(s)_cleaned'].split(",")
        mutation_sites = []
        for mutation_site in mutation_site_list:
            mutation_sites.append((mutation_site[1], mutation_site[2:-1], mutation_site[0], mutation_site[-1]))

        wt_data = process_wt(pdb_file, dist_th, group1_chains, group2_chains)
        try:
            mt_data, mut_block_indexes = process_mut(pdb_file, dist_th, group1_chains, group2_chains, mutation_sites)
        except Exception as e:
            if "Invalid mutation site, mutated residue not found in the interface." in str(e):
                print(f"WARNING: Invalid mutation site, mutated residue not found in the interface. pdb={pdb_file}")
                continue
            else:
                raise e
        item = {
            "id": f'{row["#Pdb"]}_{row["Mutation(s)_cleaned"]}',
            "wt": wt_data,
            "mt": mt_data,
            "ddG": ddg,
            "wt_binding_affinity": wt_ba,
            "mt_binding_affinity": mt_ba,
            "mt_block_indexes": mut_block_indexes,
        }
        data.append(item)
    return data

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process SKEMPI data')
    parser.add_argument('--skempi_csv', type=str, required=True,
                        help='Path to the SKEMPI CSV file')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of PDB files')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Output path')
    parser.add_argument('--dist_th', type=float, default=8.0,
                        help='Distance threshold to consider residues in the interface')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = process_data(args.skempi_csv, args.pdb_dir, args.dist_th)
    with open(args.out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {args.out_path}, n={len(data)}")
