import numpy as np
import os
import pandas as pd
import json
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from tqdm import tqdm

DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"


# Prepare RNAFM data
# The RNAErnie data is in scratch, needs to be moved to DATA_DIR
embeddings_dir = DATA_DIR + "/RNA_Protein"
model_name = "rinalmo"

for split in ["train", "val", "test"]:
    split_df = pd.read_parquet(f'{DATA_DIR}/RNA_Protein/RNA_Protein_{split}_input.parquet')
    split_df.head()

    # Load embeddings
    embeddings_df = pd.read_parquet(f'{DATA_DIR}/RNA_Protein/RNA_Protein_sequences.parquet')
    embeddings_path = f"{embeddings_dir}/RNA_Protein_embeddings_{model_name}.npy"
    tokens_path = f"{embeddings_dir}/RNA_Protein_tokens_{model_name}.npy"
    with open(embeddings_path, "rb") as f:
        embeddings = np.load(f)
    with open(tokens_path, "rb") as f:
        tokens = np.load(f)
    assert tokens.shape == embeddings.shape[:-1]
    assert len(embeddings) == len(embeddings_df)

    embeddings_id_to_index = {(row['pdb_id'], row['chain_id']): i for i, row in embeddings_df.iterrows()}

    # Extract embeddings for each split
    split_embeddings = []
    labels = []
    for i, row in tqdm(split_df.iterrows(), total=len(split_df)):
        pdb_id = row['pdb_id'].split("_")[0]
        rnafm_embeddings = {}
        for chain_id in row['chain1'].split(","):
            index = embeddings_id_to_index[(pdb_id, chain_id)]
            embedding_ = embeddings[index]
            tokens_ = tokens[index]
            # start = np.where(tokens_ == 0)[0][0]
            end = np.where(tokens_ == 2)[0][0]

            embedding_ = embedding_[:end]
            residues = embeddings_df.iloc[index]['residues']
            assert len(embedding_) == len(residues) or len(embedding_) == len(residues) + 1
            for residue, embedding in zip(residues, embedding_):
                residue_id = residue[5:].replace(".", "_")
                rnafm_embeddings[residue_id] = embedding

        assert len(row['pdb_indexes']) == len(row['label'])
        for pdb_index, label in zip(row['pdb_indexes'], row['label']):
            try:
                split_embeddings.append(rnafm_embeddings[pdb_index])
                labels.append(label)
            except KeyError:
                print("Missing", pdb_id, pdb_index)
        
    split_embeddings = np.array(split_embeddings)
    labels = np.array(labels)
    assert len(split_embeddings) == len(labels)

    np.save(f'{DATA_DIR}/RNA_Protein/RNA_Protein_{split}_embeddings_{model_name}.npy', split_embeddings)
    np.save(f'{DATA_DIR}/RNA_Protein/RNA_Protein_{split}_labels_{model_name}.npy', labels)



# Prepare RNAFM data
# only keep the pocket residues
model_name = "rinalmo"

for split in ["train", "val", "test"]:
    split_df = pd.read_parquet(f'{DATA_DIR}/RNA_Site/RNA_Site_{split}_input.parquet')
    split_df.head()

    # Load embeddings
    embeddings_df = pd.read_parquet(f'{DATA_DIR}/RNA_Site/RNA_Site_sequences.parquet')
    embeddings_path = f"{DATA_DIR}/RNA_Site/RNA_Site_embeddings_{model_name}.npy"
    tokens_path = f"{DATA_DIR}/RNA_Site/RNA_Site_tokens_{model_name}.npy"
    with open(embeddings_path, "rb") as f:
        embeddings = np.load(f)
    with open(tokens_path, "rb") as f:
        tokens = np.load(f)
    assert tokens.shape == embeddings.shape[:-1]
    assert len(embeddings) == len(embeddings_df)

    embeddings_id_to_index = {(row['pdb_id'], row['chain_id']): i for i, row in embeddings_df.iterrows()}

    # Extract embeddings for each split
    split_embeddings = []
    labels = []
    for i, row in tqdm(split_df.iterrows(), total=len(split_df)):
        pdb_id = row['pdb_id'].split("_")[0]
        rnafm_embeddings = {}
        for chain_id in row['chain1'].split(","):
            index = embeddings_id_to_index[(pdb_id, chain_id)]
            embedding_ = embeddings[index]
            tokens_ = tokens[index]
            # start = np.where(tokens_ == 0)[0][0]
            end = np.where(tokens_ == 2)[0][0]
            embedding_ = embedding_[:end]
            residues = embeddings_df.iloc[index]['residues']
            # For some we include the CLS token, for others we don't
            assert len(embedding_) == len(residues) + 1 or len(embedding_) == len(residues)
            for residue, embedding in zip(residues, embedding_):
                residue_id = residue[5:].replace(".", "_")
                rnafm_embeddings[residue_id] = embedding

        assert len(row['pdb_indexes']) == len(row['label'])
        for pdb_index, label in zip(row['pdb_indexes'], row['label']):
            try:
                split_embeddings.append(rnafm_embeddings[pdb_index])
                labels.append(label)
            except KeyError:
                print("Missing", pdb_id, pdb_index)
        
    split_embeddings = np.array(split_embeddings)
    labels = np.array(labels)
    assert len(split_embeddings) == len(labels)

    np.save(f'{DATA_DIR}/RNA_Site/RNA_Site_{split}_embeddings_{model_name}.npy', split_embeddings)
    np.save(f'{DATA_DIR}/RNA_Site/RNA_Site_{split}_labels_{model_name}.npy', labels)



# Prepare RNAFM data
embeddings_dir = DATA_DIR + "/RNA_Ligand"
model_name = "rinalmo"

for split in ["train", "val", "test"]:
    split_df = pd.read_parquet(f'{DATA_DIR}/RNA_Ligand/RNA_Ligand_{split}_input.parquet')
    split_df.head()

    # Load embeddings
    embeddings_df = pd.read_parquet(f'{DATA_DIR}/RNA_Ligand/RNA_Ligand_sequences.parquet')
    embeddings_path = f"{embeddings_dir}/RNA_Ligand_embeddings_{model_name}.npy"
    tokens_path = f"{embeddings_dir}/RNA_Ligand_tokens_{model_name}.npy"
    with open(embeddings_path, "rb") as f:
        embeddings = np.load(f)
    with open(tokens_path, "rb") as f:
        tokens = np.load(f)
    assert tokens.shape == embeddings.shape[:-1]
    assert len(embeddings) == len(embeddings_df)

    embeddings_id_to_index = {(row['pdb_id'], row['chain_id']): i for i, row in embeddings_df.iterrows()}

    # Extract embeddings for each split
    split_embeddings = []
    for i, row in split_df.iterrows():
        pdb_id = row['pdb_id'].split("_")[0]
        curr_embeddings = []
        for chain_id in row['chain1'].split(","):
            index = embeddings_id_to_index[(pdb_id, chain_id)]
            embedding_ = embeddings[index]
            tokens_ = tokens[index]
            # start = np.where(tokens_ == 0)[0][0]
            end = np.where(tokens_ == 2)[0][0]
            curr_embeddings.append(np.mean(embedding_[:end], axis=0))
        split_embeddings.append(np.mean(curr_embeddings, axis=0))
    split_embeddings = np.array(split_embeddings)
    assert len(split_embeddings) == len(split_df)
    labels = np.stack(split_df['label'].tolist())

    np.save(f'{DATA_DIR}/RNA_Ligand/RNA_Ligand_{split}_embeddings_{model_name}.npy', split_embeddings)
    np.save(f'{DATA_DIR}/RNA_Ligand/RNA_Ligand_{split}_labels_{model_name}.npy', labels)