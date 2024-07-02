import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
from process_SKEMPI import process_mut
from get_esm_embeddings import get_esm_block_embeddings

one_to_three_letter = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'E': 'GLU',
    'Q': 'GLN',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'
}


def process_data(ddg_csv, pdb_dir, dist_th=8.0):
    df = pd.read_csv(ddg_csv)
    df = df.drop_duplicates(subset=['pdb_chains', 'chain.mut'], keep='first') # discard reverse mutations, we will handle this in the model
    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing DDG data"):
        ddg = float(row['ddG'])
        if "renum_" in row['pdb_chains']:
            row['pdb_chains'] = row['pdb_chains'].replace("renum_", "")
        pdb_id, chains = row['pdb_chains'].split('_')
        pdb_file = f"{pdb_dir}/{pdb_id}.pdb"
        if not os.path.exists(pdb_file):
            raise ValueError(f"Invalid PDB file. File not found. pdb={pdb_file}")
        
        mut_chain, mutation_site = row['chain.mut'].split(".")
        wt_aa, mut_idx, mut_aa = mutation_site[0], int(mutation_site[1:-1]), mutation_site[-1]

        cleaned_pdb_indexes_mapping = {}
        with open(f"{pdb_dir}/{pdb_id}.mapping", 'r') as f:
            for line in f.readlines():
                aa, chain = line[:3], line[4]
                pdb_pos, cleaned_pos = [x for x in line[5:].strip().split(" ") if x != ""][-2:]
                if pdb_pos.isdigit() and cleaned_pos.isdigit():
                    cleaned_pdb_indexes_mapping[chain, aa, int(pdb_pos)] = int(cleaned_pos)

        mutation_sites = []
        # [(mut_ch, mut_site, wt_aa, mut_aa)]
        mutation_sites.append((mut_chain, cleaned_pdb_indexes_mapping.get((mut_chain, one_to_three_letter[wt_aa], mut_idx), mut_idx), wt_aa, mut_aa))
        try:
            wt_data, mt_data, mut_block_indexes, pdb_indexes_map = process_mut(pdb_file, dist_th, [mut_chain], list(chains.replace(mut_chain, '')), mutation_sites)
            esm_mutations = {f"{mut_ch}_{mut_site}": (wt_aa, mut_aa) for mut_ch, mut_site, wt_aa, mut_aa in mutation_sites}
            wt_esm_block_embeddings = get_esm_block_embeddings(pdb_file, pdb_indexes_map)
            if wt_esm_block_embeddings is None:
                print(f"WARNING: No sequences found in PDB file. PDB={pdb_file} positions={pdb_indexes_map} Skipping...")
                continue
            mt_esm_block_embeddings = get_esm_block_embeddings(pdb_file, pdb_indexes_map, esm_mutations)
            if mt_esm_block_embeddings is None:
                print(f"WARNING: No sequences found in PDB file. PDB={pdb_file} positions={pdb_indexes_map} Skipping...")
                continue
        except Exception as e:
            if "Invalid mutation site, mutated residue not found in the interface." in str(e):
                print(f"WARNING: Invalid mutation site, mutated residue not found in the interface. row={row}")
                continue
            else:
                raise e
        item = {
            "id": f'{pdb_id}_{chains}_{row["chain.mut"]}',
            "wt": wt_data,
            "mt": mt_data,
            "ddG": ddg,
            "mt_block_indexes": mut_block_indexes,
            "pdb_indexes_map": pdb_indexes_map,
            "wt_esm_block_embeddings": wt_esm_block_embeddings,
            "mt_esm_block_embeddings": mt_esm_block_embeddings,
        }

        assert len(item['wt']['B']) == len(item["wt_esm_block_embeddings"])
        assert len(item['mt']['B']) == len(item["mt_esm_block_embeddings"])

        data.append(item)
    return data

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process DDMut data')
    parser.add_argument('--ddg_csv', type=str, required=True,
                        help='Path to the DDG CSV file')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of PDB files')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Output path')
    parser.add_argument('--dist_th', type=float, default=8.0,
                        help='Distance threshold to consider residues in the interface')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = process_data(args.ddg_csv, args.pdb_dir, args.dist_th)
    with open(args.out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {args.out_path}, n={len(data)}")
