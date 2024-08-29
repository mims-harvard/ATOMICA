import pickle
import pandas as pd
import os
from tqdm import tqdm
import argparse
import sys

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.pdb_utils import VOCAB

def get_block_uniprot_pos(pdb_id, block_to_pdb_indexes, block_types, uniprot_df):
    pdb_uniprot_df = uniprot_df[uniprot_df['PDB'] == pdb_id]
    # Pre-filter DataFrame based on chains present in block_to_pdb_indexes
    chains = set(pdb_index.split("_")[0] for pdb_index in block_to_pdb_indexes.values())
    filtered_df = pdb_uniprot_df[pdb_uniprot_df['CHAIN'].isin(chains)]

    # Convert filtered DataFrame to a dictionary for faster lookup
    uniprot_dict = {}
    for chain, group in filtered_df.groupby('CHAIN'):
        uniprot_dict[chain] = group.to_dict('records')

    block_uniprot_pos = {}

    for block_idx, pdb_index in block_to_pdb_indexes.items():
        chain, idx = pdb_index.split("_")
        if not idx.isdigit():
            continue
        idx = int(idx)
        if chain in uniprot_dict:
            for row in uniprot_dict[chain]:
                if isinstance(row['PDB_BEG'], str):
                    if row['PDB_BEG'].isdigit():
                        row['PDB_BEG'] = int(row['PDB_BEG'])
                    else:
                        continue
                if isinstance(row['PDB_END'], str):
                    if row['PDB_END'].isdigit():
                        row['PDB_END'] = int(row['PDB_END'])
                    else:
                        continue
                if row['PDB_BEG'] <= idx <= row['PDB_END']:
                    block_uniprot_pos[block_idx] = (row['SP_PRIMARY'], row['SP_BEG'] + idx - row['PDB_BEG'], VOCAB.idx_to_symbol(block_types[block_idx]))
    return block_uniprot_pos


def process_dataset(dataset, uniprot_df):
    output = []
    for item in tqdm(dataset, total=len(dataset), desc=f"Processing dataset"):
        pdb_id = item['id'].split("_")[0]
        result = {
            "id": item['id'],
            "uniprot_map": get_block_uniprot_pos(pdb_id, item['block_to_pdb_indexes'], item['data']['B'], uniprot_df),
        }
        output.append(result)
    return output

def main(dataset_path, uniprot_segments_path, out_path, shard=None, num_shards=None):
    uniprot_df = pd.read_csv(uniprot_segments_path, sep="\t", skiprows=1)
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    if shard is not None and num_shards is not None:
        total_size = len(dataset)
        shard_size = total_size // num_shards
        start_index = shard * shard_size
        end_index = start_index + shard_size
        end_index = min(end_index, total_size)
        dataset = dataset[start_index:end_index]
    output = process_dataset(dataset, uniprot_df)
    with open(out_path, "wb") as f:
        pickle.dump(output, f)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--uniprot_segments_path", type=str, required=True) # from https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_segments_observed.tsv.gz
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    main(args.data_file, args.uniprot_segments_path, args.out_path, args.shard, args.num_shards)

