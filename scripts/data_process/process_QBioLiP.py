#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import itertools

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from data.pdb_utils import Residue, VOCAB
from data.dataset import blocks_interface, blocks_to_data
from data.converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.converter.sm_pdb_to_blocks import sm_pdb_to_blocks
from joblib import Parallel, delayed, cpu_count

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
    parser.add_argument('--fragment', type=str, default=None, choices=['PS_300', 'PS_500'], help='Fragment small molecules into common chemical motifs')
    parser.add_argument('--ccd_dictionary', type=str, default=None, help='Path to SMILES for ligand CCD codes. Required for fragmentation of small molecules.')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    parser.add_argument('--num_workers', type=int, default=16)
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
        list_blocks, pdb_indexes = pdb_to_list_blocks(prot_fname, return_indexes=True)
    except Exception as e:
        print_log(f'{protein_file_name} protein parsing failed: {e}', level='ERROR')
        return None

    if len(list_blocks) < 2:
        print_log(f'{protein_file_name} does not have at least 2 protein chains', level='ERROR')
        return None
    
    pairs = list(itertools.combinations(range(len(list_blocks)), 2))
    for i, j in pairs:
        blocks1, blocks2, indexes1, indexes2 = blocks_interface(list_blocks[i], list_blocks[j], interface_dist_th, return_indexes=True)
        if len(blocks1) >= 4 and len(blocks2) >= 4: # Minimum interface size
            data = blocks_to_data(blocks1, blocks2)
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].tolist()
            item = {}
            item['id'] = protein_file_name[:-len(".pdb")] + "_" + "_".join(set(x.split("_")[0] for x in pdb_indexes_map.values()))
            item['affinity'] = { 'neglog_aff': -1.0 }
            item['data'] = data

            pdb_indexes_map = {}
            pdb_indexes_map.update(dict(zip(range(1,len(blocks1)+1), [pdb_indexes[i][idx] for idx in indexes1])))# map block index to pdb index, +1 for global block)
            pdb_indexes_map.update(dict(zip(range(len(blocks1)+2,len(blocks1)+len(blocks2)+2), [pdb_indexes[j][idx] for idx in indexes2])))# map block index to pdb index, +1 for global block)
            item["block_to_pdb_indexes"] = pdb_indexes_map

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
        list_blocks1, list_pdb_indexes1 = pdb_to_list_blocks(rec, is_rna=is_rna, return_indexes=True)
    except Exception as e:
        print_log(f'{rec} protein parsing failed: {e}', level='ERROR')
        return None

    lig_type = complex_file_name[1].split("_")[2]
    if len(lig_type) != 3:
        if "RNA" in lig_type: # for PRNA some ligands are RNA|DNA etc
            lig_type = "RNA"
    if lig_type in {"RNA", "DNA", "III"}:
        try:
            list_of_blocks2, list_pdb_indexes2 = pdb_to_list_blocks(lig, is_rna=lig_type=="RNA", is_dna=lig_type=="DNA", return_indexes=True)
            blocks2 = sum(list_of_blocks2, [])
            pdb_indexes2 = sum(list_pdb_indexes2, [])
        except Exception as e:
            print_log(f'{lig} ligand parsing failed: {e}', level='ERROR')
            return None
    else:
        try:
            blocks2 = sm_pdb_to_blocks(lig, fragment=None)
            smiles, fragment = complex_file_name[2], complex_file_name[3]
            if smiles is not None and fragment is not None:
                try:
                    blocks2 = atom_blocks_to_frag_blocks(blocks2, smiles=smiles, fragmentation_method=fragment)
                except Exception as e:
                    print_log(f'{lig} ligand fragmentation failed: {e}', level='ERROR')
                    # use original ligand if fragmentation fails
        except Exception as e:
            print_log(f'{lig} ligand parsing failed: {e}', level='ERROR')
            return None
    blocks1 = sum(list_blocks1, [])
    pdb_indexes1 = sum(list_pdb_indexes1, [])

    # construct pockets
    blocks1, interface_blocks2, indexes1, indexes2 = blocks_interface(blocks1, blocks2, interface_dist_th, return_indexes=True)
    if len(blocks1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{complex_file_name} has no interface', level='ERROR')
        return None
    
    # Crop large RNA/DNA/III ligands
    if lig_type in {"RNA", "DNA", "III"} and len(blocks2) > 100:
        print_log(f'{lig} ligand is too big cropping it to interface', level='ERROR')
        blocks2 = interface_blocks2
        pdb_indexes2 = [pdb_indexes2[idx] for idx in indexes2]
    
    if lig_type in {"RNA", "DNA"}:
        blocks2_symbols = set([block.symbol for block in blocks2])
        invalid_blocks = blocks2_symbols.difference({"DA", "DT", "DC", "DG", "RU", "RA", "RG", "RC", VOCAB.UNK})
        if len(invalid_blocks) > 0:
            print_log(f'{lig} ligand has invalid symbols: {invalid_blocks}', level='ERROR')
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

    return item



def filter_PP_indexes(args):
    protein_indexes = pd.read_csv(args.index_path, sep=',')
    raw_protein_file_names = set(f[:-len(".pdb")] for f in os.listdir(args.data_dir_rec))
    if args.exclude_path is not None:
        with open(args.exclude_path, "r") as f:
            exclude_protein_file_names = f.readlines()
            exclude_protein_file_names = [x.strip() for x in exclude_protein_file_names]
    else:
        exclude_protein_file_names = []
    protein_file_names = []
    for _, row in protein_indexes.iterrows():
        file_name = row[0]
        if file_name not in raw_protein_file_names:
            print_log(f"Missing file: {file_name}.pdb", level="ERROR")
            continue
        pdb_id = file_name.split("_")[0]
        assert len(pdb_id) == 4, "PDB ID must be 4 characters long"
        if pdb_id in exclude_protein_file_names:
            print_log(f"Excluding file: {file_name}.pdb", level="ERROR")
            continue
        protein_file_names.append(f"{file_name}.pdb")
    return protein_file_names


def filter_complex_indexes(args):
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
    for _, row in tqdm(complex_indexes.iterrows(), total=len(complex_indexes), desc="Filtering complexes"):
        rec_file_name, ligand_file_name = row[0], row[1]
        if rec_file_name not in raw_protein_file_names:
            print_log(f"Missing file: {rec_file_name}.pdb", level="ERROR")
            continue
        if ligand_file_name not in raw_ligand_file_names:
            print_log(f"Missing file: {ligand_file_name}.pdb", level="ERROR")
            continue
        pdb_id = rec_file_name.split("_")[0]
        assert len(pdb_id) == 4, "PDB ID must be 4 characters long"
        if pdb_id in exclude_protein_file_names:
            print_log(f"Excluding file: {rec_file_name}.pdb", level="INFO")
            continue
        with open(os.path.join(args.data_dir_lig, f"{ligand_file_name}.pdb"), "r") as f:
            ligand_data = f.readlines()
        if len(ligand_data) > 5000: # some very large ribosomal RNA ligands are excluded
            print_log(f"Skipping ligand file: {ligand_file_name}.pdb because it is too large {len(ligand_data)}", level="INFO")
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


def main(args):
    if args.task == "PP":
        complex_file_names = filter_PP_indexes(args)
    else:
        complex_file_names = filter_complex_indexes(args)

    print_log(f'Preprocessing {len(complex_file_names)} protein files...')
    processed_data = []
    cnt = 0

    if args.task == "PP":
        process_one = process_one_PP
    else:
        process_one = process_one_complex

    result_list = pmap_multi(process_one, zip(complex_file_names), 
                             data_dir_rec=args.data_dir_rec,
                             data_dir_lig=args.data_dir_lig,
                             interface_dist_th=args.interface_dist_th, 
                             n_jobs=args.num_workers, desc='check BioLiP data validity')

    for item in tqdm(result_list, desc="Processing complexes"):
        if item == '':  # annotation
            continue
        cnt += 1
        if item is None:
            continue
        if isinstance(item, list):
            processed_data.extend(item)
        else:
            processed_data.append(item)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    with open(os.path.join(args.out_dir, f'QBioLiP_{args.task}.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)

    # from random import shuffle
    # shuffle(processed_data)
    # if len(processed_data)*0.1 < 10000:
    #     num_valid = int(len(processed_data)*0.1)
    #     print_log(f'10 percent of data used for validation, num data points = {num_valid}', level='WARN')
    # else:
    #     num_valid = 10000
    # train_dataset = processed_data[:-num_valid]
    # valid_dataset = processed_data[-num_valid:]

    # database_out_train = os.path.join(args.out_dir, f'QBioLiP_{args.task}_train.pkl')
    # print_log(f'Obtained {len(processed_data)} data after filtering')
    # print_log(f'Saving {len(train_dataset)} to {database_out_train} ...')
    # with open(database_out_train, 'wb') as f:
    #     pickle.dump(train_dataset, f)

    # database_out_valid = os.path.join(args.out_dir, f'QBioLiP_{args.task}_valid.pkl')
    # print_log(f'Saving {len(valid_dataset)} to {database_out_valid} ...')
    # with open(database_out_valid, 'wb') as f:
    #     pickle.dump(valid_dataset, f)

    print_log(f'Finished! Processed {len(processed_data)} items. Saved to {args.out_dir}')

if __name__ == '__main__':
    main(parse())
