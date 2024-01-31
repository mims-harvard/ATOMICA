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
                        description='PP=protein-protein, PL=protein-small molecule ligand, PRNA=protein-RNA, PDNA=protein-DNA,\
                              Ppeptide=protein-peptide, Pion=protein-ion, RNAL=RNA-small molecule ligand')
    parser.add_argument('--index_path', type=str, required=True, help='Path to Q-BioLiP annotation file')
    parser.add_argument('--exclude_path', type=str, required=True, help='Path to file with PDB ids to be excluded from the dataset')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
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


def process_one_PP(protein_file_name, data_dir, interface_dist_th):
    items = []
    prot_fname = os.path.join(data_dir, protein_file_name)
    try:
        list_blocks = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{protein_file_name} protein parsing failed: {e}', level='ERROR')
        return None

    if len(list_blocks) < 2:
        print_log(f'{protein_file_name} does not have at least 2 protein chains', level='ERROR')
        return None
    
    pairs = list(itertools.combinations(range(len(list_blocks)), 2))
    for i, j in pairs:
        blocks1, blocks2 = blocks_interface(list_blocks[i], list_blocks[j], interface_dist_th)
        if len(blocks1) >= 4 and len(blocks2) >= 4: # Minimum interface size
            data = blocks_to_data(blocks1, blocks2)
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].tolist()
            item = {}
            item['id'] = protein_file_name[:-len(".pdb")]
            item['affinity'] = { 'neglog_aff': -1.0 }
            item['data'] = data
            items.append(item)
    return items


def main(args):
    protein_indexes = pd.read_csv(args.index_path, sep=',')
    raw_protein_file_names = set(f[:-len(".pdb")] for f in os.listdir(args.data_dir))
    with open(args.exclude_path, "r") as f:
        exclude_protein_file_names = f.readlines()
        exclude_protein_file_names = [x.strip() for x in exclude_protein_file_names]
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

    print_log('Preprocessing')
    processed_data = []
    cnt = 0

    process_one_dict = {
        "PP": process_one_PP,
    }
    process_one = process_one_dict[args.task]

    result_list = pmap_multi(process_one, zip(protein_file_names), 
                             data_dir=args.data_dir,
                             interface_dist_th=args.interface_dist_th, 
                             n_jobs=args.num_workers, desc='check BioLiP data validity')

    for item in tqdm(result_list, desc="Processing complexes"):
        if item == '':  # annotation
            continue
        cnt += 1
        if item is None:
            continue
        processed_data.extend(item)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    from random import shuffle
    shuffle(processed_data)
    train_dataset = processed_data[:-10000]
    valid_dataset = processed_data[-10000:]

    database_out_train = os.path.join(args.out_dir, 'QBioLiP_PP_train.pkl')
    print_log(f'Obtained {len(processed_data)} data after filtering, saving to {database_out_train}...')
    with open(database_out_train, 'wb') as f:
        pickle.dump(train_dataset, f)

    database_out_valid = os.path.join(args.out_dir, 'QBioLiP_PP_valid.pkl')
    with open(database_out_valid, 'wb') as f:
        pickle.dump(valid_dataset, f)

    print_log('Finished!')

if __name__ == '__main__':
    main(parse())
