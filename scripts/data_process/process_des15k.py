import os
import sys
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
from ase.io import read
from tqdm import tqdm

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.dataset import blocks_to_data, Block
from data.pdb_utils import VOCAB, Atom

def find_last_H_in_first_subsequence(atom_list):
    in_first_subsequence = False
    last_H_index = -1
    
    for index, atom in enumerate(atom_list):
        if atom == 'H':
            if not in_first_subsequence:
                in_first_subsequence = True
            last_H_index = index
        else:
            if in_first_subsequence:
                break
    
    return last_H_index

def process_one(xyz_file):
    atoms = read(xyz_file)
    atom_list = atoms.get_chemical_symbols()
    last_H_index = find_last_H_in_first_subsequence(atom_list)
    coords = atoms.get_positions()
    blocks1, blocks2 = [], []
    for idx, (atom, coord) in enumerate(zip(atom_list, coords)):
        block = Block(atom.lower(), [Atom(atom, coord.tolist(), atom)])
        if idx <= last_H_index:
            blocks1.append(block)
        else:
            blocks2.append(block)
    return blocks_to_data(blocks1, blocks2)

def main(data_dir, label_csv, output_path):
    output = []

    df = pd.read_csv(label_csv)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mol_pair = f"{row['smiles0']}_{row['smiles1']}"
        group_orig = row['group_orig']
        group = row['group_id']
        k_index = row['k_index']
        label = row['cc_CCSD(T)_all']
        xyz_file = os.path.join(data_dir, mol_pair, str(group_orig), str(group), f"{k_index}.xyz")
        data = process_one(xyz_file)
        output.append({
            'data': data,
            'label': label,
            'id': f"{mol_pair}_{group_orig}_{group}_{k_index}",
        })

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"Saved n={len(output)} data to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--label_csv', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.label_csv, args.output_path)

                    