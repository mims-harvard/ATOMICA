import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np

from atomica.data.dataset import blocks_to_data
from atomica.data.converter.pdb_to_list_blocks import pdb_to_list_blocks

DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"

def process_pdb(row):
    try:
        # For one of the pdbs the whole chain is HETATM entries, so no residues are returned
        # This raises an error, but we can skip this.
        blocks, pdb_indexes = pdb_to_list_blocks(row['pdb_path'], row['chain1'].split(','), return_indexes=True, is_rna=True, is_dna=True)
    except Exception as e:
        print(f"Error processing {row['pdb_id']}: {e}")
        return None
    blocks = sum(blocks, [])
    pdb_indexes = sum(pdb_indexes, [])

    # Only keep residues that are in pdb_indexes
    to_keep_indexes = [i for i, pdb_index in enumerate(pdb_indexes) if pdb_index in row['pdb_indexes'].tolist()]
    missing_indexes = [pdb_index for pdb_index in row['pdb_indexes'].tolist() if pdb_index not in pdb_indexes]
    blocks = [blocks[i] for i in to_keep_indexes]
    pdb_indexes = [pdb_indexes[i] for i in to_keep_indexes]
    data = blocks_to_data(blocks)
    data['block_to_pdb_indexes'] = dict(zip(range(1, len(blocks)+1), pdb_indexes))
    data['id'] = row['pdb_id']

    # Check that the data is not empty
    if len(data['X']) == 0:
        print(row['pdb_id'], "is empty. Skipping...")
        return None
    
    # Check that all requested residues are present
    if missing_indexes:
        print("Missing residues in ", row['pdb_id'], missing_indexes)
    
    # Fill out the label field
    if isinstance(row['label'], list) or isinstance(row['label'], np.ndarray):
        kept_pdb_indexes = [pdb_indexes.index(pdb_index) for pdb_index in row['pdb_indexes'].tolist() if pdb_index in pdb_indexes]
        label = [row['label'][i] for i in kept_pdb_indexes]
        assert len(kept_pdb_indexes) == len(blocks)
        if len(label) != len(blocks):
            print("Label length mismatch in ", row['pdb_id'], len(label), len(blocks))
            return None
    else:
        label = row['label']
    data['label'] = label
    return data


def main(df_path: str, out_path: str):
    df = pd.read_parquet(df_path)
    items = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        data = process_pdb(row)
        if data is None:
            continue
        items.append(data)
    items = pd.DataFrame(items)
    items['block_to_pdb_indexes'] = items['block_to_pdb_indexes'].apply(json.dumps)
    print(f"Processed {len(items)} items")
    items.to_parquet(out_path)
    print(f"Saved to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.df_path, args.out_path)