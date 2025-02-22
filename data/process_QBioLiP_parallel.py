import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import itertools
import multiprocessing
from joblib import Parallel, delayed, cpu_count

from .converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks
from .converter.pdb_to_list_blocks import pdb_to_list_blocks_and_atom_array
from .converter.sm_pdb_to_blocks import sm_pdb_to_blocks
from .pdb_utils import Residue, VOCAB
from .dataset import blocks_interface, blocks_to_data


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  """

  Parallel map using joblib.

  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.

  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def parse():
    parser = argparse.ArgumentParser(description='Process Q-BioLiP PP data of protein-ligand interaction for pre-training')
    parser.add_argument('--data_dir_rec', type=str, required=True,
                        help='Directory containing receptor pdb_files')
    parser.add_argument('--data_dir_lig', type=str, default=None,
                    help='Directory containing ligand pdb_files')
    parser.add_argument('--task', required=True, type=str, choices=['PP', 'PL', 'PRNA', 'PDNA', 'Ppeptide', 'Pion', 'RNAL'], 
                        help='PP=protein-protein, PL=protein-small molecule ligand, PRNA=protein-RNA, PDNA=protein-DNA,\
                              Ppeptide=protein-peptide, Pion=protein-ion, RNAL=RNA-small molecule ligand')
    parser.add_argument('--index_path', type=str, required=True, help='Path to Q-BioLiP annotation file')
    parser.add_argument('--exclude_path', type=str, default=None, help='Path to file with PDB ids to be excluded from the dataset')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--fragment', type=str, default=None, choices=['PS_300'], help='Fragment small molecules into common chemical motifs')
    parser.add_argument('--ccd_dictionary', type=str, default=None, help='Path to SMILES for ligand CCD codes. Required for fragmentation of small molecules.')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--shard', type=int, default=0)
    return parser.parse_args()


def residue_to_pd_rows(chain: str, residue: Residue):
    rows = []
    res_id, insertion_code = residue.get_id()
    resname = residue.real_abrv if hasattr(residue, 'real_abrv') else VOCAB.symbol_to_abrv(residue.get_symbol())
    for atom_name in residue.get_atom_names():
        atom = residue.get_atom(atom_name)
        if atom.element == 'H':  # skip hydrogen
            continue
        rows.append((
            chain, insertion_code, res_id, resname,
            atom.coordinate[0], atom.coordinate[1], atom.coordinate[2],
            atom.element, atom.name
        ))
    return rows


def process_one_PP(protein_file_name, data_dir_rec, data_dir_lig, interface_dist_th):
    items = []
    prot_fname = os.path.join(data_dir_rec, protein_file_name)
    try:
        list_blocks, atom_array, pdb_indexes = pdb_to_list_blocks_and_atom_array(prot_fname)
    except Exception as e:
        print(f'{protein_file_name} protein parsing failed: {e}')
        return None

    if len(list_blocks) < 2:
        print(f'{protein_file_name} does not have at least 2 protein chains')
        return None
    
    pairs = list(itertools.combinations(range(len(list_blocks)), 2))
    for i, j in pairs:
        blocks1, blocks2, indexes1, indexes2 = blocks_interface(list_blocks[i], list_blocks[j], interface_dist_th, return_indexes=True)
        if len(blocks1) >= 4 and len(blocks2) >= 4: # Minimum interface size
            chain1 = pdb_indexes[i][indexes1[0]][0]
            chain2 = pdb_indexes[j][indexes2[0]][0]
            data = blocks_to_data(blocks1, blocks2)
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].tolist()
            item = {}
            item['id'] = protein_file_name[:-len(".pdb")] + "_" + chain1 + "_" + chain2
            item['affinity'] = { 'neglog_aff': -1.0 }
            item['data'] = data

            pdb_indexes_map = {}
            pdb_indexes_map.update(dict(zip(range(1,len(blocks1)+1), [pdb_indexes[i][idx] for idx in indexes1])))# map block index to pdb index, +1 for global block)
            pdb_indexes_map.update(dict(zip(range(len(blocks1)+2,len(blocks1)+len(blocks2)+2), [pdb_indexes[j][idx] for idx in indexes2])))# map block index to pdb index, +1 for global block)
            item["block_to_pdb_indexes"] = pdb_indexes_map

            # item['atom_array1'] = atom_array[atom_array.chain_id == chain1]
            # item['atom_array2'] = atom_array[atom_array.chain_id == chain2]

            item['dist_th'] = interface_dist_th
            items.append(item)
    return items


def process_one_complex(complex_file_name, data_dir_rec, data_dir_lig, interface_dist_th):
    lig = os.path.join(data_dir_lig, complex_file_name[1])
    rec = os.path.join(data_dir_rec, complex_file_name[0])

    item = {}
    item['id'] = complex_file_name[0] + "_" + complex_file_name[1]
    item['affinity'] = { 'neglog_aff': -1.0 }

    try:
        is_rna = "_RNA_" in rec # for RNAL
        list_blocks1, atom_array1, list_pdb_indexes1 = pdb_to_list_blocks_and_atom_array(rec, is_rna=is_rna)
    except Exception as e:
        print(f'{rec} protein parsing failed: {e}')
        return None

    lig_type = complex_file_name[1].split("_")[2]
    if len(lig_type) != 3:
        if "RNA" in lig_type: # for PRNA some ligands are RNA|DNA etc
            lig_type = "RNA"
    if lig_type in {"RNA", "DNA", "III"}:
        try:
            list_of_blocks2, atom_array2, list_pdb_indexes2 = pdb_to_list_blocks_and_atom_array(lig, is_rna=lig_type=="RNA", is_dna=lig_type=="DNA")
            blocks2 = sum(list_of_blocks2, [])
            pdb_indexes2 = sum(list_pdb_indexes2, [])
        except Exception as e:
            print(f'{lig} ligand parsing failed: {e}')
            return None
    else:
        try:
            atom_array2 = None
            blocks2 = sm_pdb_to_blocks(lig, fragment=None)
            smiles, fragment = complex_file_name[2], complex_file_name[3]
            if smiles is not None and fragment is not None:
                try:
                    blocks2 = atom_blocks_to_frag_blocks(blocks2, smiles=smiles, fragmentation_method=fragment)
                except Exception as e:
                    print(f'{lig} ligand fragmentation failed: {e}')
                    # use original ligand if fragmentation fails
        except Exception as e:
            print(f'{lig} ligand parsing failed: {e}')
            return None
    blocks1 = sum(list_blocks1, [])
    pdb_indexes1 = sum(list_pdb_indexes1, [])

    # construct pockets
    blocks1, interface_blocks2, indexes1, indexes2 = blocks_interface(blocks1, blocks2, interface_dist_th, return_indexes=True)
    if len(blocks1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print(f'{complex_file_name} has no interface')
        return None
    
    # Crop large RNA/DNA/III ligands
    if lig_type in {"RNA", "DNA", "III"} and len(blocks2) > 100:
        print(f'{lig} ligand is too big cropping it to interface')
        blocks2 = interface_blocks2
        pdb_indexes2 = [pdb_indexes2[idx] for idx in indexes2]
    
    if lig_type in {"RNA", "DNA"}:
        blocks2_symbols = set([block.symbol for block in blocks2])
        invalid_blocks = blocks2_symbols.difference({"DA", "DT", "DC", "DG", "RU", "RA", "RG", "RC", VOCAB.UNK})
        if len(invalid_blocks) > 0:
            print(f'{lig} ligand has invalid symbols: {invalid_blocks}')
            return None

    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    item['data'] = data
    pdb_indexes_map = {}
    pdb_indexes_map.update(dict(zip(range(1,len(blocks1)+1), [pdb_indexes1[idx] for idx in indexes1])))# map block index to pdb index, +1 for global block)
    if lig_type in {"RNA", "DNA", "III"}:
        assert len(blocks2) == len(pdb_indexes2), "Number of blocks and pdb indexes must match"
        pdb_indexes_map.update(dict(zip(range(len(blocks1)+2,len(blocks1)+len(blocks2)+2), pdb_indexes2)))# map block index to pdb index, +1 for global block)
    item["block_to_pdb_indexes"] = pdb_indexes_map
    item['dist_th'] = interface_dist_th
    # item['atom_array1'] = atom_array1
    # item['atom_array2'] = atom_array2

    return item



def filter_PP_indexes(args, start, end):
    protein_indexes = pd.read_csv(args.index_path, sep=',')
    raw_protein_file_names = set(f[:-len(".pdb")] for f in os.listdir(args.data_dir_rec))
    if args.exclude_path is not None:
        with open(args.exclude_path, "r") as f:
            exclude_protein_file_names = f.readlines()
            exclude_protein_file_names = [x.strip() for x in exclude_protein_file_names]
    else:
        exclude_protein_file_names = []
    protein_file_names = []
    for _, row in protein_indexes[start:end].iterrows():
        file_name = row[0]
        if file_name not in raw_protein_file_names:
            print(f"Missing file: {file_name}.pdb", level="ERROR")
            continue
        pdb_id = file_name.split("_")[0]
        assert len(pdb_id) == 4, "PDB ID must be 4 characters long"
        if pdb_id in exclude_protein_file_names:
            print(f"Excluding file: {file_name}.pdb", level="ERROR")
            continue
        protein_file_names.append(f"{file_name}.pdb")
    return protein_file_names


def filter_complex_indexes(args, start, end):
    if args.task == 'PL' and args.fragment is not None:
        ccd_df = pd.read_csv(args.ccd_dictionary, sep='\t', names=['smiles', 'ccd', 'name'])

    complex_indexes = pd.read_csv(args.index_path, sep=',')
    raw_protein_file_names = set(f[:-len(".pdb")] for f in os.listdir(args.data_dir_rec))
    raw_ligand_file_names = set(f[:-len(".pdb")] for f in os.listdir(args.data_dir_lig))
    if args.exclude_path is not None:
        with open(args.exclude_path, "r") as f:
            exclude_protein_file_names = f.readlines()
            exclude_protein_file_names = [x.strip() for x in exclude_protein_file_names]
    else:
        exclude_protein_file_names = []
    complex_file_names = []
    for _, row in tqdm(complex_indexes[start:end].iterrows(), total=end-start, desc="Filtering complexes"):
        rec_file_name, ligand_file_name = row[0], row[1]
        if rec_file_name not in raw_protein_file_names:
            print(f"Missing file: {rec_file_name}.pdb", level="ERROR")
            continue
        if ligand_file_name not in raw_ligand_file_names:
            print(f"Missing file: {ligand_file_name}.pdb", level="ERROR")
            continue
        pdb_id = rec_file_name.split("_")[0]
        assert len(pdb_id) == 4, "PDB ID must be 4 characters long"
        if pdb_id in exclude_protein_file_names:
            print(f"Excluding file: {rec_file_name}.pdb", level="INFO")
            continue
        with open(os.path.join(args.data_dir_lig, f"{ligand_file_name}.pdb"), "r") as f:
            ligand_data = f.readlines()
        if len(ligand_data) > 5000: # some very large ribosomal RNA ligands are excluded
            print(f"Skipping ligand file: {ligand_file_name}.pdb because it is too large {len(ligand_data)}", level="INFO")
            continue
        
        ligand_id = ligand_file_name.split("_")[2]
        if args.task == 'PL' and args.fragment is not None and ligand_id in ccd_df['ccd'].values:
            smiles = ccd_df[ccd_df['ccd'] == ligand_id]['smiles'].iloc[0]
            fragment = args.fragment
        else:
            smiles = None
            fragment = None
        complex_file_names.append((f"{rec_file_name}.pdb", f"{ligand_file_name}.pdb", smiles, fragment))
    return complex_file_names


def process_shard(params):
    args, start, end, shard_idx = params
    print(f'Shard {shard_idx}: Filtering start={start}, end={end} indexes...')
    if args.task == "PP":
        complex_file_names = filter_PP_indexes(args, start, end)
    else:
        complex_file_names = filter_complex_indexes(args, start, end)

    print(f'Shard {shard_idx}: Preprocessing {len(complex_file_names)} protein files...')
    processed_data = []
    cnt = 0

    if args.task == "PP":
        process_one = process_one_PP
    else:
        process_one = process_one_complex

    for complex_file_name in tqdm(complex_file_names, desc=f"Processing complexes, shard {shard_idx}"):
        cnt += 1
        item = process_one(complex_file_name, args.data_dir_rec, args.data_dir_lig, args.interface_dist_th)
        if item is None:
            continue
        if isinstance(item, list):
            processed_data.extend(item)
        else:
            processed_data.append(item)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    with open(os.path.join(args.out_dir, f'{args.task}_{shard_idx}.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f'Finished shard={shard_idx}! Processed {len(processed_data)} items. Saved to {args.out_dir}')



def main(args):
    complex_indexes = pd.read_csv(args.index_path, sep=',')
    num_shards = args.num_workers
    shard_size = len(complex_indexes) // num_shards
    shard_start = [i * shard_size for i in range(num_shards)]
    shard_end = shard_start[1:] + [len(complex_indexes)]

    if num_shards > 1:
        with multiprocessing.Pool(num_shards) as pool:
            params = [
                (
                    args,
                    shard_start[worker_id],
                    shard_end[worker_id],
                    worker_id,
                )
                for worker_id in range(num_shards)
            ]
            list(pool.imap_unordered(process_shard, params))
    else:
        if args.end is None:
            args.end = len(complex_indexes)
        args.end = min(args.end, len(complex_indexes))
        process_shard((args, args.start, args.end, args.shard))


if __name__ == '__main__':
    main(parse())
