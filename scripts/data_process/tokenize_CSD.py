import os 
import sys
from tqdm import tqdm
import pickle
import argparse

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.dataset import PDBBindBenchmark
from data.converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks
from utils.multiprocess_pmap import pmap_multi
from data.pdb_utils import VOCAB, Atom
from data.dataset import Block, blocks_to_data


def data_to_blocks(data):
    curr_atom_idx = 0
    list_of_blocks = []
    curr_segment_id = 0
    curr_blocks = []
    for block_idx, block in enumerate(data['B']):
        symbol = VOCAB.idx_to_symbol(block)
        if symbol == VOCAB.GLB:
            curr_atom_idx += data['block_lengths'][block_idx]
            continue
        atom_coords = data['X'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]
        atom_positions = data['atom_positions'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]
        atoms = []
        for i, atom in enumerate(data['A'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]):
            atom_name=VOCAB.idx_to_atom(atom)
            if atom_name == VOCAB.atom_global:
                continue
            element=VOCAB.idx_to_atom(atom)
            coordinate=atom_coords[i]
            pos_code=VOCAB.idx_to_atom_pos(atom_positions[i])
            atoms.append(Atom(atom_name=atom_name, element=element, coordinate=coordinate, pos_code=pos_code))
        curr_atom_idx += data['block_lengths'][block_idx]
        if data['segment_ids'][block_idx] != curr_segment_id:
            list_of_blocks.append(curr_blocks)
            curr_blocks = []
            curr_segment_id = data['segment_ids'][block_idx]
        curr_blocks.append(Block(symbol, atoms))
    list_of_blocks.append(curr_blocks)
    return list_of_blocks


def process_one(data, smiles, fragmentation_method):
    try:
        list_of_blocks = data_to_blocks(data)
        new_list_of_blocks = []
        for blocks, smi in zip(list_of_blocks, smiles):
            new_blocks = atom_blocks_to_frag_blocks(blocks, smiles=smi, fragmentation_method=fragmentation_method)
            new_list_of_blocks.append(new_blocks)
        new_data = blocks_to_data(*new_list_of_blocks)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None
    return new_data


def main(args):
    dataset = PDBBindBenchmark(args.data_file)
    smiles = [(d["id"].split("_")[1], d["id"].split("_")[3]) for d in dataset.indexes]
    result_list = pmap_multi(process_one, zip(dataset, smiles), 
                              fragmentation_method=args.fragmentation_method, n_jobs=args.num_workers)

    processed_data = []
    for item in tqdm(result_list, desc="Processing complexes"):
        if item is None:
            continue
        processed_data.append(item)

    print(f"Saving processed data to {args.output}. Total of {len(processed_data)} items.")
    with open(args.output, "wb") as f:
        pickle.dump(processed_data, f)


def parse():
    parser = argparse.ArgumentParser(description='Tokenize processed CSD data')
    parser.add_argument('--data_file', type=str, required=True,
                        help='path to processed CSD data with no tokenization')
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--fragmentation_method", type=str, default="PS_300")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse())