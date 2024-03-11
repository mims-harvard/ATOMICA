
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
from tqdm import tqdm
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool

def tm_align_pdbs(pdb1, chain1_idx, pdb2, chain2_idx, out_path="tm_align_results"):
    s1 = get_structure(pdb1)
    chain1 = list(s1.get_chains())[chain1_idx]
    coords1, seq1 = get_residue_data(chain1)
    s2 = get_structure(pdb2)
    chain2 = list(s2.get_chains())[chain2_idx]
    try:
        coords2, seq2 = get_residue_data(chain2)
    except Exception as e:
        print(f"Error with pdb {pdb2}, chain {chain2_idx}: {e}")
        return None
    res = tm_align(coords1, coords2, seq1, seq2)
    pdb2_name = os.path.basename(pdb2).split(".")[0]
    with open(f"{out_path}/{pdb2_name}_{chain2.id}.pdb", "w") as f:
        f.write(f"{res.tm_norm_chain1}, {res.tm_norm_chain2}")
    return res.tm_norm_chain1, res.tm_norm_chain2

def process_one(params):
    pdb_file, canonical_pdb = params
    chains = list(get_structure(pdb_file).get_chains())
    output = []
    for ch_idx, ch in enumerate(chains):
        if len(list(ch.get_residues())) < 50:
            continue
        res = tm_align_pdbs(canonical_pdb, 0, pdb_file, ch_idx)
        if res is None:
            continue
        similarity1, similarity2 = res
        output.append((pdb_file, ch.id, similarity1, similarity2))
        print((pdb_file, ch.id, similarity1, similarity2))
    return output


def main(pdb_files, canonical_pdb, num_workers):
    os.makedirs("tm_align_results", exist_ok=True)
    output = []
    with Pool(num_workers) as p:
        params = [(x, canonical_pdb) for x in pdb_files]
        for res in tqdm(p.imap_unordered(process_one, params), total=len(pdb_files)):
            output.extend(res)
    
    df = pd.DataFrame(output, columns=["pdb_file", "chain_id", "similarity1", "similarity2"])
    df.to_csv("tm_align_results.csv", index=False)


if __name__ == '__main__':
    num_workers=16
    pdb_files = [f"pdb_files/{x}" for x in os.listdir('pdb_files')]
    canonical_pdb = 'pdb_files/4QL1.pdb'
    main(pdb_files, canonical_pdb, num_workers)
