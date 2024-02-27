import os 
import sys
from tqdm import tqdm
import pickle
import argparse
import json

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks
from data.pdb_utils import VOCAB, Atom
from data.dataset import Block, blocks_to_data, data_to_blocks


def process_one(item, smiles, fragmentation_method):
    try:
        blocks = data_to_blocks(item["data"])
        lig_blocks = atom_blocks_to_frag_blocks(blocks[1], smiles=smiles, fragmentation_method=fragmentation_method)
        new_data = blocks_to_data(blocks[0], lig_blocks)
        new_item = {"id": item["id"], "data": new_data, "affinity": item["affinity"]}
    except Exception as e:
        print(f'Error: {e}')
        new_item = None
    return new_item

def main(args):
    with open(args.ligand_dict, 'rb') as f:
        ligand_json = json.load(f)
    ligand_dict = {item["ligid"]: item["smiles"] for item in ligand_json}
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    new_data = []
    num_success = 0
    num_fail = 0
    for item in tqdm(data):
        ligand_id = item["id"].split("_")[-2]
        if ligand_id not in ligand_dict:
            print(f'Error: ligand_id {ligand_id} not found')
            num_fail += 1
            continue
        smiles = ligand_dict[ligand_id]
        new_item = process_one(item, smiles, args.fragmentation_method)
        if new_item is not None:
            new_data.append(new_item)
            num_success += 1
        else:
            num_fail += 1
    print(f'num_success: {num_success}, num_fail: {num_fail}')
    with open(args.output_path, 'wb') as f:
        pickle.dump(new_data, f)


def parse():
    parser = argparse.ArgumentParser(description='Tokenize QBioLiP PL data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to QBioLiP data')
    parser.add_argument('--ligand_dict', type=str, required=True, help='Path to ligand dict')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output file')
    parser.add_argument('--fragmentation_method', type=str, default='PS_300', choices=['PS_300', 'PS_500'], help='Fragmentation method')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    main(args)