import biotite.structure.io.pdb as pdb
from collections import defaultdict
import pandas as pd
import biotite.structure as bs
import numpy as np
from tqdm import tqdm
import argparse
import pickle
import os
import pandas as pd
from .dataset import blocks_to_data
from .converter.pdb_to_list_blocks import atoms_array_to_blocks, get_residues

def parse_args():
    parser = argparse.ArgumentParser(description="Process protein structures based on B-factor cutoff.")
    parser.add_argument('--b_factor_cutoff', type=float, required=True, help='B-factor cutoff value')
    parser.add_argument('--plddt_cutoff', type=float, default=None, required=False, help='pLDDT cutoff value')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the protein data files processed by PESTO')
    parser.add_argument('--raw_data_dir', type=str, default=None, required=False, help='Directory containing the AF2 protein data files')
    parser.add_argument('--prot_list', type=str, required=True, help='File containing the list of protein names separated by newline character')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed output files')
    return parser.parse_args()

def process_one(atom_array: bs.AtomArray):
    list_blocks = []
    list_residues = []
    for chain_id in np.unique(atom_array.chain_id):
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        _, residues = get_residues(chain_atom_array)
        blocks = atoms_array_to_blocks(chain_atom_array)
        list_blocks.extend(blocks)
        list_residues.extend(residues)
    
    data = blocks_to_data(list_blocks)
    pdb_indexes_map = {i+1: list_residues[i] for i in range(len(list_residues))} # +1 for global residue index

    item = {
        "data": data,
        "block_to_pdb_indexes": pdb_indexes_map,
    }
    return item

if __name__ == "__main__":
    args = parse_args()

    with open(args.prot_list, "r") as f:
        prot_names = f.read().splitlines()

    processed_data = defaultdict(list)
    output = []
    binders = ['protein', 'nucleic_acid', 'ion', 'ligand', 'lipid']
    for prot_name in tqdm(prot_names, total=len(prot_names)):

        if args.plddt_cutoff and args.raw_data_dir:
            raw_pdb_file_path = os.path.join(args.raw_data_dir, f'{prot_name}.pdb')
            if not os.path.exists(raw_pdb_file_path):
                print(f"Raw PDB file not found: {raw_pdb_file_path}")
                continue
            raw_pdb_file = pdb.PDBFile()
            with open(raw_pdb_file_path, 'r') as file:
                raw_pdb_file.read(file)
            raw_atom_array = pdb.get_structure(raw_pdb_file)[0]
            plddt = pdb.PDBFile.get_b_factor(raw_pdb_file)[0]

        for i in range(5):
            pdb_file_path = os.path.join(args.data_dir, f'{prot_name}_i{i}.pdb')
            if not os.path.exists(pdb_file_path):
                print(f"PESTO processed PDB file not found: {pdb_file_path}")
                continue

            pdb_file = pdb.PDBFile()
            with open(pdb_file_path, 'r') as file:
                pdb_file.read(file)
            atom_array = pdb.get_structure(pdb_file)[0]
            b_factor = pdb.PDBFile.get_b_factor(pdb_file)[0]

            if args.plddt_cutoff and args.raw_data_dir:
                if len(raw_atom_array) != len(atom_array):
                    print(f"Atom array length mismatch between PESTO file and AF2 file: {prot_name}")
                    continue
                atom_array = atom_array[plddt > args.plddt_cutoff]
                b_factor = b_factor[plddt > args.plddt_cutoff]
            
            atom_array_filtered = atom_array[b_factor > args.b_factor_cutoff]
            if len(atom_array_filtered) == 0:
                print(f"{prot_name} - {binders[i]}: 0 residues")
                residue_starts = []
                residues = []
            else:
                residue_starts, residues = get_residues(atom_array_filtered)
                print(f"{prot_name} - {binders[i]}: {len(residues)} residues")
                if len(residues) > 5:
                    atom_array.res_tuples = list(zip(atom_array.chain_id, atom_array.res_id, atom_array.res_name, atom_array.ins_code))
                    atom_array_binding_residues = atom_array[[res_tuple in residues for res_tuple in atom_array.res_tuples]]
                    atom_array_binding_residues = atom_array_binding_residues[bs.filter_amino_acids(atom_array_binding_residues)]
                    item = process_one(atom_array_binding_residues)
                    item['id'] = prot_name
                    item['binder'] = binders[i]
                    processed_data[binders[i]].append(item)
            output.append([prot_name, pdb_file_path, binders[i], residue_starts, residues])

    output_df = pd.DataFrame(output, columns=['protein', 'pdb_file_path', 'binder_type', 'residue_starts', 'residues'])

    os.makedirs(args.output_dir, exist_ok=True)
    if args.plddt_cutoff:
        fname = f"pesto_residues_{int(args.b_factor_cutoff*100)}_plddt_{int(args.plddt_cutoff)}.csv"
    else:
        fname = f"pesto_residues_{int(args.b_factor_cutoff*100)}.csv"
    output_df.to_csv(os.path.join(args.output_dir, fname), index=False)

    if args.plddt_cutoff:
        fname = f"pesto_{int(args.b_factor_cutoff*100)}_plddt_{int(args.plddt_cutoff)}"
    else:
        fname = f"pesto_{int(args.b_factor_cutoff*100)}"

    for binder, items in processed_data.items():
        print(f"{binder}: {len(items)} items")
        with open(os.path.join(args.output_dir, f"{fname}_{binder}.pkl"), "wb") as f:
            pickle.dump(items, f)