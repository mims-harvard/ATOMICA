import os
os.environ['HF_HOME'] = '/n/holystore01/LABS/mzitnik_lab/Lab/afang/huggingface_cache'
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinTensor, SamplingConfig, SamplingTrackConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from typing import List
from tqdm import tqdm
import torch
import pickle
import numpy as np
import biotite.structure as bs
from biotite.structure import AtomArray, get_residue_starts
from biotite.structure.io.pdb import PDBFile
import pickle
import numpy as np
from typing import Tuple, List, Dict

CLIENT = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))

# Imports for ESM3 function embeddings

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
        'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 
        'UNK': '?'}

def get_residues(atoms_array: AtomArray) -> Tuple[np.ndarray, List[Tuple[str, int, str, str]]]:
    # residues: (chain_id, res_id, res_name, ins_code)
    residue_starts = get_residue_starts(atoms_array)
    residues = []
    for res_idx in residue_starts:
        residues.append((atoms_array.chain_id[res_idx], atoms_array.res_id[res_idx], atoms_array.res_name[res_idx], atoms_array.ins_code[res_idx]))
    return residue_starts, residues

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

def encode_one_protein(bs_atoms_array, max_len:int = 1024):
    protein_chain=ProteinChain.from_atomarray(bs_atoms_array)
    unique_chain_ids = np.unique(bs_atoms_array.chain_id)
    if len(unique_chain_ids) > 1:
        print("WARNING: more than one chain, will only embed the first chain")
    protein = ESMProtein.from_protein_chain(protein_chain)
    seq_len = len(protein.sequence)
    idxs = get_chunk_idxs(seq_len+2, max_len=max_len) # + 2 for EOS and BOS
    protein_tensor = CLIENT.encode(protein).to("cpu")

    interpro_keywords_list = []
    for (start, end) in idxs:
        sub_tensor = ESMProteinTensor(
            sequence=protein_tensor.sequence[start:end],
            structure=protein_tensor.structure[start:end],
            coordinates=protein_tensor.coordinates[start:end],
        ).to("cuda")
        interpro_keywords, tfidf_embedding, lsh_embedding = function_annotation(sub_tensor)
        for interpro_keyword in interpro_keywords:
            interpro_keyword.start += start-1 # -1 for BOS token
            interpro_keyword.end += start-1 # -1 for BOS token
        interpro_keywords_list.extend(interpro_keywords)
    return interpro_keywords_list

@torch.no_grad()
def embed_one_prot_sequence(
    seq: str, 
    client, 
    max_len:int = 2048,
):
    seq_len = len(seq)
    # + 2 for EOS and BOS
    idxs = get_chunk_idxs(seq_len+2, max_len=max_len)
    protein = ESMProtein(
        sequence=seq,
    )
    full_tensor = client.encode(protein).to("cpu")

    interpro_keywords_list = []
    for (start, end) in idxs:
        sub_tensor = ESMProteinTensor(
            sequence=full_tensor.sequence[start:end]
        ).to("cuda")
        interpro_keywords, tfidf_embedding, lsh_embedding = function_annotation(sub_tensor)
        for interpro_keyword in interpro_keywords:
            interpro_keyword.start += start-1 # -1 for BOS token
            interpro_keyword.end += start-1 # -1 for BOS token
        interpro_keywords_list.extend(interpro_keywords)
    return interpro_keywords_list

def function_annotation(protein_tensor):
    sampling_config = SamplingConfig(function=SamplingTrackConfig(only_sample_masked_tokens=False))
    protein_tensor = protein_tensor.to("cuda")
    print("num residues", protein_tensor.sequence.shape)
    output = CLIENT.forward_and_sample(
        protein_tensor, sampling_config
    )
    decoded_function = CLIENT.get_function_decoder().decode(
        output.protein_tensor.function,
        tokenizer=CLIENT.tokenizers.function,
    )
    interpro_keywords = decoded_function["function_keywords"]
    tfidf_embedding = CLIENT.tokenizers.function._tfidf.encode(decoded_function["function_keywords"])
    lsh_embedding = output.protein_tensor.function.detach().cpu().numpy()
    return interpro_keywords, tfidf_embedding, lsh_embedding # interpro_keywords, tfidf_embedding, lsh_embedding

def find_interface_annotations(annotations, interface_indexes):
    annotations.sort(key=lambda x: x.end)
    interface_annotations = set()
    for point in interface_indexes:
        for annotation in annotations:
            if annotation.start <= point <= annotation.end:
                interface_annotations.add(annotation.label)
            elif point > annotation.end:
                break
    return interface_annotations


def encode_segment(
        bs_atoms_array: AtomArray, interface_residues: Dict[int, List[Tuple[str, int, str, str]]]
    ) -> Dict[int, torch.Tensor]:
    for chain in np.unique(bs_atoms_array.chain_id):
        chain_atoms_array = bs_atoms_array[bs_atoms_array.chain_id == chain]
        _, residues = get_residues(chain_atoms_array)
        interface_indexes = {block_idx: residues.index(res) for block_idx, res in interface_residues.items() if res in residues}
        functional_annotations = encode_one_protein(chain_atoms_array)

        if len(interface_indexes) == 0:
            return set(), set()

        print("Interface residues", len(interface_indexes))
        interface_annotations = find_interface_annotations(functional_annotations, interface_indexes.values())
        all_annotations = set([annotation.label for annotation in functional_annotations])
    return interface_annotations, all_annotations


def process_one(item, segment0_type:str, segment1_type: str):
    if segment0_type is not None and segment0_type == 'protein':
        segment0_residues = {block_idx: res for block_idx, res in item['block_to_pdb_indexes'].items() if item['data']['segment_ids'][block_idx] == 0}
        if 'atom_array1' in item:
            atom_array = item['atom_array1']
        else:
            pdb_path = f"/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/protein_universe/function/uniprot_cluster_proteins/AF-{item['id']}-F1-model_v4.pdb"
            atom_array = PDBFile.read(pdb_path).get_structure()[0]
        segment0_interface_functions, segment0_functions = encode_segment(atom_array, segment0_residues)
    else:
        segment0_functions = set()
        segment0_interface_functions = set()

    if segment1_type is not None and segment1_type == 'protein':
        raise NotImplementedError("Segment 1 not implemented")
    #     segment1_residues = {block_idx: res for block_idx, res in item['block_to_pdb_indexes'].items() if item['data']['segment_ids'][block_idx] == 1}
    #     segment1_interface_functions, segment1_functions = encode_segment(item['atom_array2'], segment1_residues)
    else:
        segment1_functions = set()
        segment1_interface_functions = set()
    
    return {'id': item['id'], 'segment0_interface_functions': segment0_interface_functions, 'segment1_interface_functions': segment1_interface_functions, 'segment0_functions': segment0_functions, 'segment1_functions': segment1_functions}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process PDBbind dataset')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--segment0_type', type=str, choices=['protein'], default=None)
    parser.add_argument('--segment1_type', type=str, choices=['protein'], default=None)
    parser.add_argument('--shard_idx', type=int, default=None)
    parser.add_argument('--num_shards', type=int, default=None)
    return parser.parse_args()


def main_fasta(args):
    with open(args.data_path, 'r') as f:
        lines = f.readlines()
    ids = [l.strip().replace(">", "") for l in lines[::2]]
    sequences = [l.strip() for l in lines[1::2]]
    fasta_dict = {id: seq for id, seq in zip(ids, sequences)}
    output = {}
    for id, seq in tqdm(fasta_dict.items()):
        interpro_keywords_list = embed_one_prot_sequence(seq, CLIENT)
        output[id] = interpro_keywords_list
    with open(args.output_path, 'wb') as f:
        pickle.dump(output, f)


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
        new_item = process_one(item, args.segment0_type, args.segment1_type)
        new_dataset.append(new_item)
        print(new_item)
    with open(args.output_path, 'wb') as f:
        pickle.dump(new_dataset, f)

if __name__ == "__main__":
    args = parse_args()
    if args.data_path.endswith('.fasta'):
        main_fasta(args)
    else:
        main(args)