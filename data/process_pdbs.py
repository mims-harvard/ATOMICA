import argparse
import pandas as pd
import pickle
from tqdm import tqdm
from .converter.pdb_lig_to_blocks import extract_pdb_ligand
from .converter.pdb_to_list_blocks import pdb_to_list_blocks
from .dataset import blocks_interface, blocks_to_data

def parse_args():
    parser = argparse.ArgumentParser(description='Process Q-BioLiP PP data of protein-ligand interaction for pre-training')
    parser.add_argument('--data_index_file', type=str, required=True, help="""CSV file containing the following headers [ pdb_id | pdb_path | chain1 | chain2 | lig_code | lig_smiles ]
                            pdb_id: 4 letter pdb code, 
                            pdb_path: path to the pdb file, 
                            chain1: chain of the protein delimited with '_', 
                            chain2: chain of the ligand delimited with '_', 
                            lig_code: ligand code if ligand, leave empty/None if the interface is chain 2. If lig, then chain2 must refer to the chain the ligand is on.
                            lig_smiles: smiles for ligand, used for fragmentation of ligand into common chemical motifs.
                            label (optional): binding affinity label for the interaction, leave empty/None if not available.
                        """)
    parser.add_argument('--out_path', type=str, required=True, help='Output path')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    parser.add_argument('--fragmentation_method', type=str, default=None, choices=['PS_300'], help='fragmentation method for small molecule ligands')
    return parser.parse_args()

def process_PL_pdb(pdb_file, pdb_id, rec_chain, lig_code, lig_chain, smiles, dist_th, fragmentation_method=None):
    items = []
    list_lig_blocks, list_lig_indexes = extract_pdb_ligand(pdb_file, lig_code, lig_chain, smiles, use_model=0, fragmentation_method=fragmentation_method)
    rec_blocks, rec_indexes = pdb_to_list_blocks(pdb_file, selected_chains=rec_chain, return_indexes=True)
    rec_blocks = sum(rec_blocks, [])
    rec_indexes = sum(rec_indexes, [])
    for idx, (lig_blocks, lig_indexes) in enumerate(zip(list_lig_blocks, list_lig_indexes)):
        interface_rec_blocks, interface_lig_blocks, interface_rec_indexes, interface_lig_indexes = blocks_interface(rec_blocks, lig_blocks, dist_th, return_indexes=True)
        if len(interface_rec_blocks) == 0 or len(interface_lig_blocks) == 0:
            continue
        data = blocks_to_data(interface_rec_blocks, interface_lig_blocks)
        rec_pdb_indexes = [rec_indexes[i] for i in interface_rec_indexes]
        lig_pdb_indexes = [lig_indexes[i] for i in interface_lig_indexes]
        id = f"{pdb_id}_{''.join(rec_chain)}_{lig_chain}_{lig_code}"
        if len(list_lig_blocks) > 1:
            id = f"{id}_{idx}"
        pdb_indexes_map = {}
        pdb_indexes_map.update(dict(zip(range(1,len(interface_rec_blocks)+1), rec_pdb_indexes))) # map block index to pdb index, +1 for global block)
        pdb_indexes_map.update(dict(zip(range(len(interface_rec_blocks)+2,len(interface_rec_blocks)+len(interface_lig_indexes)+2), lig_pdb_indexes))) # map block index to pdb index, +2 for global blocks
        items.append({
            'data': data,
            'block_to_pdb_indexes': pdb_indexes_map,
            'id': id,
        })
    return items

def group_chains(list_chain_blocks, list_chain_pdb_indexes, group1, group2):
    group1_chains = []
    group2_chains = []
    group1_indexes = []
    group2_indexes = []
    for chain_blocks, chain_pdb_indexes in zip(list_chain_blocks, list_chain_pdb_indexes):
        if chain_pdb_indexes[0].split("_")[0] in group1:
            group1_chains.extend(chain_blocks)
            group1_indexes.extend(chain_pdb_indexes)
        elif chain_pdb_indexes[0].split("_")[0] in group2:
            group2_chains.extend(chain_blocks)
            group2_indexes.extend(chain_pdb_indexes)
    return [group1_chains, group2_chains], [group1_indexes, group2_indexes]

def process_pdb(pdb_file, pdb_id, group1_chains, group2_chains, dist_th):
    blocks, pdb_indexes = pdb_to_list_blocks(pdb_file, selected_chains=group1_chains+group2_chains, return_indexes=True, use_model=0)
    if len(blocks) != 2:
        blocks, pdb_indexes = group_chains(blocks, pdb_indexes, group1_chains, group2_chains)
    blocks1, blocks2, block1_indexes, block2_indexes = blocks_interface(blocks[0], blocks[1], dist_th, return_indexes=True)
    if len(blocks1) == 0 or len(blocks2) == 0:
        return None
    pdb_indexes_map = {}
    pdb_indexes_map.update(dict(zip(range(1,len(blocks1)+1), [pdb_indexes[0][i] for i in block1_indexes])))# map block index to pdb index, +1 for global block)
    pdb_indexes_map.update(dict(zip(range(len(blocks1)+2,len(blocks1)+len(blocks2)+2), [pdb_indexes[1][i] for i in block2_indexes])))# map block index to pdb index, +1 for global block)
    data = blocks_to_data(blocks1, blocks2)
    return {
        "data": data,
        "id": f"{pdb_id}_{''.join(group1_chains)}_{''.join(group2_chains)}",
        "block_to_pdb_indexes": pdb_indexes_map,
    }

def main(args):
    data_index_file = pd.read_csv(args.data_index_file)
    items = []
    for _, row in tqdm(data_index_file.iterrows(), total=len(data_index_file)):
        pdb_file = row['pdb_path']
        pdb_id = row['pdb_id']
        chain1 = row['chain1']
        chain2 = row['chain2']
        lig_code = row['lig_code']
        smiles = row['lig_smiles']
        chain1 = chain1.split("_")
        chain2 = chain2.split("_")
        if 'label' in row:
            label = row['label']
        else:
            label = None
        if lig_code is None or lig_code == '' or pd.isna(lig_code):
            # For PP, PDNA, PRNA, Ppeptide interactions
            item = process_pdb(pdb_file, pdb_id, chain1, chain2, args.interface_dist_th) 
            if item is not None:
                if label is not None:
                    item['label'] = label
                items.append(item)
            else:
                print(f"WARNING: Invalid interface, no interface found. pdb={pdb_id}, chain1={''.join(chain1)}, chain2={''.join(chain2)}")
        else:
            if len(chain2) > 1:
                raise ValueError(f"Invalid chain2, ligand chain must be a single chain. pdb={pdb_file}")
            chain2 = chain2[0]
            pl_items = process_PL_pdb(pdb_file, pdb_id, chain1, lig_code, chain2, smiles, args.interface_dist_th, fragmentation_method=args.fragmentation_method)
            if len(pl_items) == 0:
                print(f"WARNING: Invalid interface, no interface found. pdb={pdb_id}, chain1={''.join(chain1)}, chain2={chain2}, lig_code={lig_code}")
            elif len(pl_items) > 1:
                print(f"WARNING: Multiple ligands {len(pl_items)} that match the description, adding all of them. pdb={pdb_id}, chain1={''.join(chain1)}, chain2={chain2}, lig_code={lig_code}")
            for item in pl_items:
                if label is not None:
                    item['label'] = label
            items.extend(pl_items)
    
    with open(args.out_path, 'wb') as f:
        pickle.dump(items, f)
    
    print(f"Finished processing. Total items={len(items)}. Saved to {args.out_path}")


if __name__ == "__main__":
    main(parse_args())
