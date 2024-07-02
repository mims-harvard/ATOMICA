import pickle
import pandas as pd
import os
from tqdm import tqdm
import argparse

# wget http://dunbrack2.fccc.edu/ProtCiD/pfam/PDBfam.txt.gz
def get_block_pfam_domain(pdb_id, block_to_pdb_indexes, pfam_df):
    pdb_pfam_df = pfam_df[pfam_df['PdbID'] == pdb_id]
    # Pre-filter DataFrame based on chains present in block_to_pdb_indexes
    chains = set(pdb_index.split("_")[0] for pdb_index in block_to_pdb_indexes.values())
    filtered_df = pdb_pfam_df[pdb_pfam_df['AuthChain'].isin(chains)]

    # Convert filtered DataFrame to a dictionary for faster lookup
    pfam_dict = {}
    for chain, group in filtered_df.groupby('AuthChain'):
        pfam_dict[chain] = group.to_dict('records')

    block_pfam_domains = set()

    for pdb_index in block_to_pdb_indexes.values():
        chain, idx = pdb_index.split("_")
        if not idx.isdigit():
            continue
        idx = int(idx)
        if chain in pfam_dict:
            for row in pfam_dict[chain]:
                if isinstance(row['PdbSeqStart'], str):
                    if row['PdbSeqStart'].isdigit():
                        row['PdbSeqStart'] = int(row['PdbSeqStart'])
                    else:
                        continue
                if isinstance(row['PdbSeqEnd'], str):
                    if row['PdbSeqEnd'].isdigit():
                        row['PdbSeqEnd'] = int(row['PdbSeqEnd'])
                    else:
                        continue
                if row['PdbSeqStart'] <= idx <= row['PdbSeqEnd']:
                    block_pfam_domains.add(row['Pfam_Acc'])
    return block_pfam_domains

# wget https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_go.tsv.gz 
def get_block_go_labels(pdb_id, block_to_pdb_indexes, go_df):
    # Create the set of chains once
    chains = {pdb_index.split("_")[0] for pdb_index in block_to_pdb_indexes.values()}

    # Filter the DataFrame once
    pdb_go_df = go_df.loc[(go_df['PDB'] == pdb_id) & (go_df['CHAIN'].isin(chains)), 'GO_ID']

    # Use set comprehension to create the block_go_labels set
    block_go_labels = set(pdb_go_df)
    return block_go_labels

def process_dataset(dataset, modality, go_df, pfam_df):
    output = []
    for item in tqdm(dataset, total=len(dataset), desc=f"Processing {modality}"):
        pdb_id = item['id'].split("_")[0]
        result = (
            item['id'],
            modality,
            get_block_pfam_domain(pdb_id, item['block_to_pdb_indexes'], pfam_df),
            get_block_go_labels(pdb_id, item['block_to_pdb_indexes'], go_df),
        )
        output.append(result)
    return output

def main(dataset_path, go_df_path, pfam_df_path, out_path, shard=None, num_shards=None):
    go_df = pd.read_csv(go_df_path, sep="\t", skiprows=1)
    pfam_df = pd.read_csv(pfam_df_path, sep="\t")
    pfam_df = pfam_df[pfam_df['Evalue'] < 1e-5]
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    if shard is not None and num_shards is not None:
        total_size = len(dataset)
        shard_size = total_size // num_shards
        start_index = shard * shard_size
        end_index = start_index + shard_size
        end_index = min(end_index, total_size)
        dataset = dataset[start_index:end_index]
    modality = os.path.basename(dataset_path).split("_")[0]
    output = process_dataset(dataset, modality, go_df, pfam_df)
    with open(out_path, "wb") as f:
        pickle.dump(output, f)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--go_df_path", type=str, required=True)
    parser.add_argument("--pfam_df_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    main(args.data_file, args.go_df_path, args.pfam_df_path, args.out_path, args.shard, args.num_shards)

