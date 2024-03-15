import os 
import sys
import pandas as pd
import argparse
import pickle
from tqdm import tqdm

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from Bio.PDB import PDBParser
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.pdb_utils import Atom, VOCAB
from data.dataset import Block, blocks_interface, blocks_to_data
from data.converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks

with open("./data/converter/excluded_qbiolip_ligands.txt", "r") as f:
    EXCLUDE_LIGANDS = f.read().splitlines()

CCD = pd.read_csv('./data/converter/pdb_chemical_components_smiles.txt', sep='\t', names=['smiles', 'id', 'name']).set_index('id')
AA_SYMBOLS = [x[0] for x in VOCAB.aas]

def extract_ligand_blocks(pdb_filename, remove_Hs=True):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB_structure', pdb_filename)
    
    ligand_info = {}
    
    for chain in structure[0]:
        for residue in chain:
            if residue.id[0] != ' ':  # HETATM records are prefixed with 'H_' in Bio.PDB
                resname = f"{chain.id}_{residue.id[1]}_{residue.get_resname()}"
                if residue.get_resname() not in ['HOH', 'WAT'] + EXCLUDE_LIGANDS:  # Exclude water molecules
                    coords = []
                    atoms = []
                    num_carbons = 0
                    for atom in residue:
                        if atom.element == 'H' and remove_Hs:
                            continue
                        coords.append(list(atom.get_vector()))
                        atoms.append(atom.element)
                        if atom.element == 'C':
                            num_carbons += 1
                    if num_carbons > 5:
                        blocks = convert_ligand_to_blocks(atoms, coords)
                        if residue.get_resname() in CCD.index:
                            smiles = CCD.loc[residue.get_resname()]['smiles']
                            try:
                                blocks = atom_blocks_to_frag_blocks(blocks, smiles=smiles, fragmentation_method='PS_300')
                            except Exception as e:
                                print(f"Error tokenizing {resname}: {e}")
                        ligand_info[resname] = blocks
    return ligand_info

def convert_ligand_to_blocks(atoms, coords):
    blocks = []
    for atom, coord in zip(atoms, coords):
        atom_obj = Atom(
            atom_name=atom,
            # e.g. C1, C2, ..., these position code will be a unified encoding such as <sm> (small molecule) in our framework
            coordinate=coord,
            element=atom,
            pos_code=VOCAB.atom_pos_sm
        )
        blocks.append(Block(
            symbol=atom.lower(),
            units=[atom_obj]
        ))
    return blocks

def process_one(pdb_file, select_chain, dist_th=8.0):
    list_of_blocks, list_of_indexes = pdb_to_list_blocks(pdb_file, return_indexes=True, use_model=0)
    # cleaned_list_of_blocks = []
    # for blocks in list_of_blocks:
    #     clean_blocks = [b for b in blocks if b.symbol in AA_SYMBOLS]
    #     cleaned_list_of_blocks.append(clean_blocks)
    # list_of_blocks = cleaned_list_of_blocks

    chain_ids = [indexes[0].split("_")[0] for indexes in list_of_indexes]
    ligand_blocks = extract_ligand_blocks(pdb_file)
    select_idx = chain_ids.index(select_chain)

    interface_interactions = []

    # get all interactions of the selected chain with neighbouring chains
    for ligand in ligand_blocks:
        block1, _ = blocks_interface(list_of_blocks[select_idx], ligand_blocks[ligand], dist_th=dist_th)
        if len(block1) > 0:
            data = blocks_to_data(block1, ligand_blocks[ligand])
            pdb_id = pdb_file.split("/")[-1].split(".")[0]
            item = {
                "id": f"{pdb_id}_{select_chain}_{ligand}",
                "data": data,
            }
            interface_interactions.append(item)

    # get all interactions of the selected chain with ligands
    for idx, other_chain in enumerate(chain_ids):
        if other_chain == select_chain:
            continue
        block1, block2 = blocks_interface(list_of_blocks[select_idx], list_of_blocks[idx], dist_th=dist_th)
        if len(block1) > 0:
            data = blocks_to_data(block1, block2)
            pdb_id = pdb_file.split("/")[-1].split(".")[0]
            item = {
                "id": f"{pdb_id}_{select_chain}_{other_chain}",
                "data": data,
            }
            interface_interactions.append(item)
    return interface_interactions

def main(args):
    df = pd.read_csv(args.pdb_similarity)
    df = df[df['TM_score_reference']>args.pdb_similarity_cutoff]
    
    output = []
    for pdb_id, chain in tqdm(zip(df['pdb_id'], df['chain']), desc="Processing PDB files", total=len(df)):
        pdb_file = f"{args.pdb_dir}/{pdb_id}.pdb"
        interface_interactions = process_one(pdb_file, chain, dist_th=args.dist_th)
        output.extend(interface_interactions)
    
    with open(args.output_file, "wb") as f:
        pickle.dump(output, f)
    print(f"Saving processed data to {args.output_file}. Total of {len(output)} items.")


def parse():
    parser = argparse.ArgumentParser(description='Extract interface interactions from PDB files')
    parser.add_argument('--pdb_similarity', type=str, required=True,
                        help='path to pdb_similarity.csv containing TM scores')
    parser.add_argument('--pdb_similarity_cutoff', type=float, default=0.8,
                        help='TM score cutoff for selecting PDB files')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='path to directory containing PDB files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='path to output file')
    parser.add_argument('--dist_th', type=float, default=8.0,
                        help='distance threshold for interface interactions')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    main(args)