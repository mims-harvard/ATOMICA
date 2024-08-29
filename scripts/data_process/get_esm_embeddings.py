from Bio.PDB import PDBParser
import torch 
import esm 

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
        'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 
        'UNK': '?'}

MODEL, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()

def get_chain_sequence(pdb_file, chain_id, mutations=None):
    # mutations (dictionary): key = "{mut_ch}_{mut_site}", value = (wt_aa, mut_aa)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)[0]  # use model 0

    for chain in structure.get_chains():
        if chain.id == chain_id:
            seq = []
            resnums = []
            for residue in chain:
                resnum = f"{chain.id}_{residue.get_id()[1]}"
                resname = d3to1.get(residue.resname, '?')
                if mutations is not None and resnum in mutations:
                    wt_aa, mut_aa = mutations[resnum]
                    if resname != wt_aa:
                        print(f"WARNING: Residue mismatch: {resname} != {wt_aa}. PDB={pdb_file}. Chain={chain_id}. Resnum={resnum}. Mut_aa={mut_aa}.")
                    seq.append(mut_aa)
                else:
                    seq.append(resname)
                resnums.append(resnum)
            return ''.join(seq), resnums
    return None


def get_chain_sequences(pdb_file, mutations):
    # mutations (dictionary): key = "{mut_ch}_{mut_site}", value = (wt_aa, mut_aa)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)[0]  # use model 0
    output = {}
    for chain in structure.get_chains():
        seq = []
        resnums = []
        for residue in chain:
            resnum = f"{chain.id}_{residue.get_id()[1]}"
            resname = d3to1.get(residue.resname, '?')
            if mutations is not None and resnum in mutations:
                wt_aa, mut_aa = mutations[resnum]
                if resname != wt_aa:
                    print(f"WARNING: Residue mismatch: {resname} != {wt_aa}. PDB={pdb_file}. Chain={chain.id}. Resnum={resnum}. Mut_aa={mut_aa}.")
                seq.append(mut_aa)
            else:
                seq.append(resname)
            resnums.append(resnum)
        chain_seq = ''.join(seq)
        chain_resnums = resnums
        output[chain.id] = (chain_seq, chain_resnums)
    return output


def chunk_string(input_string, chunk_size=1022):
    return [(i, input_string[i:min(i + chunk_size, len(input_string))]) for i in range(0, len(input_string), chunk_size)]


def get_esm_embeddings(pdb_file, chains, mutations=None):
    if chains == None:
        chain_sequences = get_chain_sequences(pdb_file, mutations)
        chains = list(chain_sequences.keys())
    else:
        chain_sequences = {chain: get_chain_sequence(pdb_file, chain, mutations) for chain in chains}
    chain_sequences = {k: (v[0].rstrip("?"), v[1][:len(v[0].rstrip("?"))]) for k, v in chain_sequences.items()} # strip trailing HETATM from sequences
    data = [(chain, chain_sequences[chain][0]) for chain in chains]

    batch = []
    for chain_id, chain in data:
        if len(chain) > 1022:
            batch.extend([(f"{chain_id}_{start_idx}", subch.replace('?', '<unk>')) for start_idx, subch in chunk_string(chain)])
        else:
            batch.append((chain_id, chain.replace('?', '<unk>')))
    
    if len(batch) == 0:
        print(f"WARNING: No sequences found in PDB file. PDB={pdb_file} Chain={chains} Skipping...")
        return None

    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to("cuda")
    model = MODEL.to("cuda")
    model.eval()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()
    
    # group up the split chains
    grouped_results = {}
    for j, tokens_len in enumerate(batch_lens):
        chain_id = batch_labels[j] 
        if "_" in chain_id:
            chain_id = chain_id.split("_")[0]
        if chain_id in grouped_results:
            grouped_results[chain_id] = torch.cat((grouped_results[chain_id], token_representations[j, 1:tokens_len-1]), dim=0)
        else:
            grouped_results[chain_id] = token_representations[j, 1:tokens_len-1]
    
    return grouped_results, chain_sequences

def get_chain_mean_esm_embedding(pdb_file, mutations=None):
    grouped_results, _ = get_esm_embeddings(pdb_file, chains=None, mutations=mutations)
    # return mean of each chain
    return {chain_id: torch.mean(grouped_results[chain_id], dim=0) for chain_id in grouped_results}

def get_block_esm_embedding(pdb_file, pdb_indexes_map, mutations=None):
    chains = set([x.split("_")[0] for x in pdb_indexes_map.values()])
    grouped_results, chain_sequences = get_esm_embeddings(pdb_file, chains, mutations)
    # return pocket block embeddings according to pdb_indexes_map
    esm_embeddings = {}
    global_blocks = {}
    prev_global_block = None
    for i in range(max(pdb_indexes_map.keys())+1):
        if i not in pdb_indexes_map: 
            global_blocks[i] = set()
            prev_global_block = i
        else:
            chain_id = pdb_indexes_map[i].split("_")[0]
            global_blocks[prev_global_block].add(chain_id)
            resnums = chain_sequences[chain_id][1]
            esm_embeddings[i] = grouped_results[chain_id][resnums.index(pdb_indexes_map[i])]
    
    # global block ESM embeddings are the mean of the whole chain of all chains in the segment
    for global_block, chains in global_blocks.items():
        esm_global_embedding = torch.mean(torch.stack([torch.mean(grouped_results[chain_id], dim=0) for chain_id in chains]), dim=0)
        esm_embeddings[global_block] = esm_global_embedding
    
    output = []
    for i in range(max(pdb_indexes_map.keys())+1):
        assert i in esm_embeddings, f"Missing ESM embedding for block {i}"
        output.append(esm_embeddings[i])
    output = torch.stack(output)
    return output


if __name__ == "__main__":
    import os 
    from tqdm import tqdm
    import pickle

    def parse():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--pdb_dir", type=str, required=True)
        parser.add_argument("--out_path", type=str, required=True)
        return parser.parse_args()

    output = {}
    args = parse()
    for pdb_file in tqdm(os.listdir(args.pdb_dir), total=len(os.listdir(args.pdb_dir))):
        if not pdb_file.endswith(".pdb"):
            continue
        pdb_file = os.path.join(args.pdb_dir, pdb_file)
        pdb_id = os.path.basename(pdb_file).split(".")[0]
        output[pdb_id] = get_chain_mean_esm_embedding(pdb_file)
    
    with open(args.out_path, "wb") as f:
        pickle.dump(output, f)