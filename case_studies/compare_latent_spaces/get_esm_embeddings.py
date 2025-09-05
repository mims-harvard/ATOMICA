import pickle
import random
import torch
import pickle
import biotite.structure as bs
import biotite.structure.io.pdb as bs_pdb
import numpy as np
import os 
from typing import List, Tuple
from tqdm import tqdm

import esm
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
        'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 
        'UNK': '?', 'MASK': '#'}
nucleotideto1 = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T', 
                 'RA': 'A', 'RC': 'C', 'RG': 'G', 'RU': 'T', 
                 'A': 'A', 'C': 'C', 'G': 'G', 'U': 'T', 'T': 'T',
                 'UNK': '<unk>'} # FIXME RNA is just mapped to DNA in NucleotideTransformer

def get_chunk_idxs(
    seq_len: int,
    max_len: int,
) -> List[int]:
    num_pieces = (seq_len + max_len - 1) // max_len
    lo_size = seq_len // num_pieces
    hi_size = lo_size + 1
    num_hi = seq_len % num_pieces

    chunk_lens = [hi_size for _ in range(num_hi)] + [lo_size for _ in range(num_pieces - num_hi)]

    chunk_idxs = []
    curr = 0
    for chunk_len in chunk_lens:
        chunk_idxs.append((curr, curr+chunk_len))
        curr += chunk_len
    return chunk_idxs


def process_one(item, pdb_file):
    atom_array = bs_pdb.PDBFile.read(pdb_file).get_structure()[0]
    aa_filter = bs.filter_canonical_amino_acids(atom_array)
    atom_array = atom_array[aa_filter]
    sequence = ""

    pdb_indexes_to_block = {v: k for k, v in item['block_to_pdb_indexes'].items()}

    sequence_indexes = []
    for residue_start in bs.get_residue_starts(atom_array):
        chain = atom_array.chain_id[residue_start]
        res_id = atom_array.res_id[residue_start]
        residue_index = atom_array.res_name[residue_start]
        if f"{chain}_{res_id}" in pdb_indexes_to_block:
            sequence_indexes.append(len(sequence))
        sequence += d3to1.get(residue_index, '?')
    
    interface_embedding = encode_one_chain(sequence, np.array(sequence_indexes))
    return interface_embedding

def encode_one_chain(sequence, block_idx, max_len:int = 1022) -> Tuple[torch.Tensor, str]:
    batch = []
    idxs = get_chunk_idxs(len(sequence), max_len=max_len)

    for idx, (start, end) in enumerate(idxs):
        batch.append((idx, sequence[start:end].replace('?', '<unk>').replace('#', '<mask>')))
    
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to("cuda")
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()
    outputs = []
    for idx, tokens_len in enumerate(batch_lens):
        outputs.append(token_representations[idx, 1:tokens_len-1])
    outputs = torch.cat(outputs).numpy()
    interface_residues = outputs[block_idx]
    interface_embedding = np.mean(interface_residues, axis=0)
    return interface_embedding


if __name__ == '__main__':
    pdb_dir = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/interact_score/test_set_pdbs/"
    data_path = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PL_test.pkl"
    out_path = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/latent_space/ESM/PL_test_ESM2_embeddings.pkl"

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    esm_embeddings = {}
    for item in tqdm(data, total=len(data), desc="Processing ESM scores"):
        item_id = item['id']
        pdb_file = os.path.join(pdb_dir, f"{item_id}")
        esm_embeddings[item_id] = process_one(item, pdb_file)
    
    with open(out_path, 'wb') as f:
        pickle.dump(esm_embeddings, f)