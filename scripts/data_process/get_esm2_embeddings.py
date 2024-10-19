from typing import Tuple, List, Optional, Dict
from biotite.structure import AtomArray
import numpy as np
import esm
import torch
import pickle
from copy import deepcopy
from tqdm import tqdm
import sys
import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print("Adding path", PATH)
sys.path.append(PATH)

os.environ['HF_HOME'] = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/huggingface_cache"
from transformers import AutoTokenizer, AutoModelForMaskedLM

from data.converter.pdb_to_list_blocks import get_residues

ESM2_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
NT_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
NT_model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
        'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 
        'UNK': '?'}
nucleotideto1 = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T', 
                 'RA': 'A', 'RC': 'C', 'RG': 'G', 'RU': 'T', 
                 'A': 'A', 'C': 'C', 'G': 'G', 'U': 'T', 'T': 'T',
                 'UNK': '<unk>'} # FIXME RNA is just mapped to DNA

def get_embedding_size(segment_type: str) -> int:
    if segment_type == 'protein':
        return 2560
    elif segment_type == 'nucleotide':
        return 2560
    else:
        raise ValueError(f"Unknown segment type {segment_type}")


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

def encode_one_protein(bs_atoms_array: AtomArray, max_len:int = 1022) -> Tuple[torch.Tensor, str]:
    _, residues = get_residues(bs_atoms_array)
    sequence = ''.join([d3to1.get(res[2], '?') for res in residues])
    batch = []
    idxs = get_chunk_idxs(len(sequence), max_len=max_len)

    for idx, (start, end) in enumerate(idxs):
        batch.append((idx, sequence[start:end].replace('?', '<unk>')))
    
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to("cuda")
    model = ESM2_model.to("cuda")
    model.eval()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()

    outputs = []
    for idx, tokens_len in enumerate(batch_lens):
        outputs.append(token_representations[idx, 1:tokens_len-1])
    outputs = torch.cat(outputs)
    assert outputs.shape[0] == len(sequence), f"outputs shape {outputs.shape} != {len(sequence)}"
    return outputs, sequence


def encode_one_nucleotide_sequence(bs_atoms_array: AtomArray) -> Tuple[torch.Tensor, str]:
    _, residues = get_residues(bs_atoms_array)
    sequence_list = [nucleotideto1.get(res[2], '<unk>') for res in residues]
    sequence = ''.join(sequence_list)
    
    max_length = NT_tokenizer.model_max_length
    tokens_ids = NT_tokenizer.encode(sequence, return_tensors="pt", truncation=False)
    if tokens_ids.shape[1] > max_length:
        token_chunks = tokens_ids[0].split(max_length)
        token_chunks = [
            torch.cat([chunk, torch.full((max_length - chunk.size(0),), NT_tokenizer.pad_token_id, dtype=torch.long)])
            if chunk.size(0) < max_length else chunk
            for chunk in token_chunks
        ]
        tokens_ids = torch.stack(token_chunks)
    attention_mask = tokens_ids != NT_tokenizer.pad_token_id
    model = NT_model.to("cuda")
    tokens_ids = tokens_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    model.eval()
    with torch.no_grad():
        output = model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embeddings = output['hidden_states'][-1].detach().cpu()
        # attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        # mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
    
    nt_embeddings = torch.zeros((len(residues), embeddings.shape[-1]))
    for chain in range(tokens_ids.shape[0]):
        curr_seq_idx = 0
        for token_idx, token_id in enumerate(tokens_ids[chain][attention_mask[chain]]):
            token = NT_tokenizer.convert_ids_to_tokens(token_id.item())
            if token in {'<pad>', '<cls>', '<mask>'}:
                continue
            elif token == '<unk>':
                nt_embeddings[curr_seq_idx] = embeddings[chain][token_idx]
                curr_seq_idx += 1
            else:
                for nt in token: # NucleotideTransformer uses 6-mer tokens, expand to 1-mer
                    nt_embeddings[curr_seq_idx] = embeddings[chain][token_idx]
                    curr_seq_idx += 1
            if curr_seq_idx == len(residues):
                break
    return nt_embeddings, sequence


def encode_segment(
        bs_atoms_array: AtomArray, interface_residues: Dict[int, List[Tuple[str, int, str, str]]], global_block_idx: int, segment_type: str
    ) -> Dict[int, torch.Tensor]:
    block_embeddings = {}
    all_outputs = []
    for chain in np.unique(bs_atoms_array.chain_id):
        chain_atoms_array = bs_atoms_array[bs_atoms_array.chain_id == chain]
        _, residues = get_residues(chain_atoms_array)
        interface_indexes = {block_idx: residues.index(res) for block_idx, res in interface_residues.items() if res in residues}
        if segment_type == 'protein':
            outputs, _ = encode_one_protein(chain_atoms_array)
        elif segment_type == 'nucleotide':
            outputs, _ = encode_one_nucleotide_sequence(chain_atoms_array)
        else:
            raise ValueError(f"Unknown segment type {segment_type}")
        block_embeddings.update({block_idx: outputs[res_idx] for block_idx, res_idx in interface_indexes.items()})
        all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs)
    block_embeddings[global_block_idx] = torch.mean(all_outputs, dim=0)
    return block_embeddings


def process_one_only_one_block_embedding(item, segment0_type:str, segment1_type: str):
    num_segment0_blocks = len(item['data']['segment_ids']) - sum(item['data']['segment_ids'])
    num_segment1_blocks = sum(item['data']['segment_ids'])

    if (segment0_type and segment1_type) and (segment0_type != segment1_type):
        raise ValueError(f"Segment0 type {segment0_type} and segment1 type {segment1_type} must be the same") 
    
    if segment0_type is not None:
        segment0_residues = {block_idx: res for block_idx, res in item['block_to_pdb_indexes'].items() if item['data']['segment_ids'][block_idx] == 0}
        block_embeddings_segment0 = torch.zeros((num_segment0_blocks, get_embedding_size(segment0_type)))
        block_embeddings_dict = encode_segment(item['atom_array1'], segment0_residues, 0, segment0_type)
        for block_idx, block_embedding in block_embeddings_dict.items():
            block_embeddings_segment0[block_idx] = block_embedding
        block_embeddings_segment0 = block_embeddings_segment0.tolist()
    else:
        block_embeddings_segment0 = torch.zeros((num_segment0_blocks, get_embedding_size(segment1_type))).tolist()
    
    if segment1_type is not None:
        segment1_residues = {block_idx: res for block_idx, res in item['block_to_pdb_indexes'].items() if item['data']['segment_ids'][block_idx] == 1}
        block_embeddings_segment1 = torch.zeros((num_segment1_blocks, get_embedding_size(segment1_type)))
        block_embeddings_dict = encode_segment(item['atom_array2'], segment1_residues, num_segment0_blocks, segment1_type)
        for block_idx, block_embedding in block_embeddings_dict.items():
            block_embeddings_segment1[block_idx-num_segment0_blocks] = block_embedding
        block_embeddings_segment1 = block_embeddings_segment1.tolist()
    else:
        block_embeddings_segment1 = torch.zeros((num_segment1_blocks, get_embedding_size(segment0_type))).tolist()

    item['data']['block_embeddings'] = block_embeddings_segment0 + block_embeddings_segment1
    return item


def process_one(item, segment0_type:str, segment1_type: str):
    num_segment0_blocks = len(item['data']['segment_ids']) - sum(item['data']['segment_ids'])
    num_segment1_blocks = sum(item['data']['segment_ids'])

    if segment0_type is not None:
        block_embeddings_segment0 = torch.zeros((num_segment0_blocks, get_embedding_size(segment0_type)))
        block_embeddings_dict = encode_segment(item['atom_array1'], item['block_to_pdb_indexes'], 0, segment0_type)
        for block_idx, block_embedding in block_embeddings_dict.items():
            block_embeddings_segment0[block_idx] = block_embedding
        item['data']['block_embeddings0'] = block_embeddings_segment0.tolist()
    
    if segment1_type is not None:
        block_embeddings_segment1 = torch.zeros((num_segment1_blocks, get_embedding_size(segment1_type)))
        block_embeddings_dict = encode_segment(item['atom_array2'], item['block_to_pdb_indexes'], num_segment0_blocks, segment1_type)
        for block_idx, block_embedding in block_embeddings_dict.items():
            block_embeddings_segment1[block_idx-num_segment0_blocks] = block_embedding
        item['data']['block_embeddings1'] = block_embeddings_segment1.tolist()
    return item


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process PDBbind dataset')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--segment0_type', type=str, choices=['protein', 'nucleotide'], default=None)
    parser.add_argument('--segment1_type', type=str, choices=['protein', 'nucleotide'], default=None)
    parser.add_argument('--only_one_block_embedding', action='store_true', default=False)
    parser.add_argument('--drop_atom_array', action='store_true', default=False)
    parser.add_argument('--shard_idx', type=int, default=None)
    parser.add_argument('--num_shards', type=int, default=None)
    return parser.parse_args()


def main(args):
    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if args.shard_idx is not None and args.num_shards is not None:
        assert args.shard_idx < args.num_shards
        chunk_size = len(dataset) // args.num_shards + 1
        lo = args.shard_idx * chunk_size
        hi = min(len(dataset), (args.shard_idx + 1) * chunk_size)
        dataset = dataset[lo:hi]

    new_dataset = []
    for item in tqdm(dataset, total=len(dataset)):
        if args.only_one_block_embedding:
            new_item = process_one_only_one_block_embedding(item, args.segment0_type, args.segment1_type)
        else:
            new_item = process_one(item, args.segment0_type, args.segment1_type)
        if args.drop_atom_array:
            if 'atom_array1' in new_item:
                del new_item['atom_array1']
            if 'atom_array2' in new_item:
                del new_item['atom_array2']
        new_dataset.append(new_item)
    with open(args.output_path, 'wb') as f:
        pickle.dump(new_dataset, f)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)