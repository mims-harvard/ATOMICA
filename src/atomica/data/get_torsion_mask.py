# Adapted from https://github.com/gcorso/DiffDock/blob/main/utils/torsion.py

import argparse
import pickle
from tqdm import tqdm
import multiprocessing

from .dataset import data_to_blocks
from .pdb_utils import VOCAB
from .torsion import get_side_chain_torsion_mask_block, get_segment_torsion_mask

RESIDUES = set([x[0] for x in VOCAB.aas] + [x[0] for x in VOCAB.bases])

def process_one(item):
    """
    Get torsion mask for each complex
    sidechain = torsion mask is applied to the side chains - this is for proteins/DNA/RNA to not disturb the backbone
    segment = torsion mask is applied to all atoms this is for the ligands

    Returns:
    {
        'id': original id,
        ...
        'data': original data,
        'torsion_mask': [
            {
                'type': 0=sidechain, 1=segment,
                'edges': [n_rotatable_edges, 2],
                'mask_rotate': [n_rotatable_edges, n_atoms]
            } for each segment id
        ]
    }
    """
    list_of_masks = []
    list_of_blocks = data_to_blocks(item["data"])
    for blocks in list_of_blocks:
        block_ids = set([block.symbol for block in blocks])
        rot_sidechains = len(block_ids.intersection(RESIDUES)) > 0
        if rot_sidechains:
            # protein or nucleic acid
            edges, mask_rotate = get_side_chain_torsion_mask_block(blocks)
        else:
            edges, mask_rotate = get_segment_torsion_mask(blocks)
        list_of_masks.append({
            "type": 0 if rot_sidechains else 1,
            "edges": edges,
            "mask_rotate": mask_rotate,
        })
    item["data"]["torsion_mask"] = list_of_masks
    return item


def main(args):
    VOCAB.load_tokenizer(args.fragmentation_method)
    with open(args.data_file, "rb") as f:
        dataset = pickle.load(f)
    
    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_one, dataset), total=len(dataset), desc="Processing torsion edges"))

    with open(args.output_file, "wb") as f:
        pickle.dump(results, f)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--fragmentation_method", type=str, default=None, choices=["PS_300", "PS_500"])
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    main(args)