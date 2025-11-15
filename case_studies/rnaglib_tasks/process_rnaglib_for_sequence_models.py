from rnaglib.tasks import (
    ChemicalModification, 
    RNAGo,
    InverseFolding,
    LigandIdentification,
    ProteinBindingSite,
    BenchmarkBindingSite,
    BindingSite,
)

import numpy as np
import os
import pandas as pd
import json
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from tqdm import tqdm

DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"

def pdb_to_seq(pdb_path, chains, keep_residues):
    output = []
    parser = MMCIFParser()
    structure = parser.get_structure('pdb', pdb_path)
    pdb_id = os.path.basename(pdb_path).split(".")[0]
    for chain in chains:
        list_residues = []
        seq = ""

        chain = structure[0][chain]
        for residue in chain.get_residues():
            hetero_flag, res_number, insert_code = residue.get_id()
            residue_id = f"{pdb_id}.{chain.id}.{res_number}"
            list_residues.append(residue_id)
            res_name = residue.get_resname().strip()
            if res_name not in ['A', 'C', 'G', 'U']:
                seq += "-"
            else:
                seq += res_name
            assert len(seq) == len(list_residues), f"Sequence length {len(seq)} does not match number of residues {len(list_residues)}"
        
        # left strip the non canonical residues at the end of the sequence
        # usually these are water molecules
        if len(seq) == 0:
            print(f"Sequence is empty for pdb {pdb_id} chain {chain.id}")
            return None
        while seq[-1] == "-":
            if list_residues[-1] in keep_residues:
                break
            seq = seq[:-1]
            list_residues = list_residues[:-1]

        assert len(seq) == len(list_residues), f"Sequence length {len(seq)} does not match number of residues {len(list_residues)}. Seq {seq} residues {list_residues}. Pdb {pdb_id} chain {chain.id}."

        output.append([pdb_id, chain.id, seq, list_residues])
    return output



def process_rnaglib_for_sequence_models(train, val, test, task_name):
    # write sequences out in fasta format for RNA-FM
    sequences = []
    processed_pdbs = set()
    
    all_ranglib_residues = set()
    for split_name, split in zip(["train", "val", "test"], [train, val, test]):
        for entry in split:
            with open(entry['graph_path'], 'r') as f:
                data = json.load(f)
            for node in data['nodes']:
                all_ranglib_residues.add(node['id'])

    for split_name, split in zip(["train", "val", "test"], [train, val, test]):
        for entry in tqdm(split, desc=f"Processing {split_name} tasks", total=len(split)):
            with open(entry['graph_path'], 'r') as f:
                data = json.load(f)
            pdb_id = data['graph']['pdbid']
            chains = set(node['chain_id'] for node in data['nodes'])
            unprocessed_chains = set()
            for chain in chains:
                pdb_chain = f"{pdb_id}.{chain}"
                if pdb_chain in processed_pdbs:
                    continue
                processed_pdbs.add(pdb_chain)
                unprocessed_chains.add(chain)
            
            if unprocessed_chains:
                entries = pdb_to_seq(f'{DATA_DIR}/pdbs/{pdb_id}.cif', unprocessed_chains, all_ranglib_residues)
                if entries is None:
                    continue
                sequences.extend(entries)
            
            for node in data['nodes']:
                node['id']

    sequences_df = pd.DataFrame(sequences, columns=['pdb_id', 'chain_id', 'seq', 'residues'])
    sequences_df.to_parquet(f'{DATA_DIR}/{task_name}/{task_name}_sequences.parquet')

    print(f"{task_name}: Checking for missing residues...")
    all_residues = set(sum(sequences_df['residues'], []))
    missing_residues = all_ranglib_residues - all_residues
    print("\n".join(missing_residues))


def main():
    ta = RNAGo(
        root=f"{DATA_DIR}/RNAGo",
        recompute=False,
        debug=False,
    )
    train, val, test = ta.get_split_datasets(recompute=False)
    process_rnaglib_for_sequence_models(train, val, test, "RNAGo")

    ta = ProteinBindingSite(
        root=f"{DATA_DIR}/RNA_Protein",
        recompute=False,
        debug=False,
    )
    train, val, test = ta.get_split_datasets(recompute=False)
    process_rnaglib_for_sequence_models(train, val, test, "RNA_Protein")

    ta = BindingSite(
        root=f"{DATA_DIR}/RNA_Site",
        recompute=False,
        debug=False,
    )
    train, val, test = ta.get_split_datasets(recompute=False)
    process_rnaglib_for_sequence_models(train, val, test, "RNA_Site")

    ta = LigandIdentification(
        root=f"{DATA_DIR}/RNA_Ligand",
        recompute=False,
        debug=False,
    )
    train, val, test = ta.get_split_datasets(recompute=False)
    process_rnaglib_for_sequence_models(train, val, test, "RNA_Ligand")

if __name__ == "__main__":
    main()