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
from data.dataset import Block, blocks_to_data, data_to_blocks


def process_one(item, fragmentation_method):
    smiles = (item["id"].split("_")[1], item["id"].split("_")[3])
    try:
        list_of_blocks = data_to_blocks(item["data"])
        new_list_of_blocks = []
        for blocks, smi in zip(list_of_blocks, smiles):
            new_blocks = atom_blocks_to_frag_blocks(blocks, smiles=smi, fragmentation_method=fragmentation_method)
            new_list_of_blocks.append(new_blocks)
        new_data = blocks_to_data(*new_list_of_blocks)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None
    new_item = {"id": item["id"], "data": new_data, "affinity": item["affinity"]}
    return new_item


def main(args):
    dataset = PDBBindBenchmark(args.data_file)
    result_list = pmap_multi(process_one, zip(dataset.data), 
                            fragmentation_method=args.fragmentation_method, 
                            n_jobs=args.num_workers)

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