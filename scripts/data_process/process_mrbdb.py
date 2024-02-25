#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from data.dataset import blocks_interface, blocks_to_data
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
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

def make_lig_file(complex_file_name):
    with open(complex_file_name, 'r') as f:
        lines = f.readlines()
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.startswith('HETATM') and start_idx is None:
            start_idx = i
        if not line.startswith('HETATM') and start_idx is not None:
            end_idx = i
            break
    new_filename = complex_file_name.replace("_complex.pdb", "_ligand.pdb")
    with open(new_filename, 'w') as f:
        f.writelines(lines[start_idx:end_idx])
    return new_filename


def process_one_complex(complex_file_name, mut_site, interface_dist_th):
    rec = os.path.join(complex_file_name)
    lig = make_lig_file(complex_file_name)

    try:
        list_blocks1, list_indexes1 = pdb_to_list_blocks(rec, return_indexes=True)
    except Exception as e:
        print_log(f'{rec} protein parsing failed: {e}', level='ERROR')
        return None
    blocks1 = []
    for b in list_blocks1:
        blocks1.extend(b)
    indexes1 = []
    for i in list_indexes1:
        indexes1.extend(i)
    try:
        blocks2 = sm_pdb_to_blocks(lig, fragment=None)
    except Exception as e:
        print_log(f'{lig} ligand parsing failed: {e}', level='ERROR')
        return None

    # construct pockets
    blocks1, _, chosen_indexes1, _ = blocks_interface(blocks1, blocks2, interface_dist_th, return_indexes=True)
    indexes1 = [indexes1[i] for i in chosen_indexes1]
    mutation_site_on_interface = False
    for idx in indexes1:
        idx = int(idx.split('_')[-1])
        if idx == mut_site:
            mutation_site_on_interface = True
            break
    if not mutation_site_on_interface:
        print_log(f'{complex_file_name} mutation site is not on interface', level='ERROR')
        return None
    if len(blocks1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{complex_file_name} has no interface', level='ERROR')
        return None
    
    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    return data


def process_wt_mt_pair(dirname, mut_site, ddG, interface_dist_th):
    item = {}
    item['id'] = os.path.basename(dirname)
    item['ddG'] = ddG
    wt_file = None
    mt_file = None
    for f in os.listdir(dirname):
        if f.endswith('_complex.pdb'):
            if f.startswith('WT'):
                wt_file = os.path.join(dirname, f)
            elif f.startswith('MT'):
                mt_file = os.path.join(dirname, f)
    if wt_file is None or mt_file is None:
        print_log(f'wt or mt file not found in {dirname}', level='ERROR')
        return None
    wt_data = process_one_complex(wt_file, mut_site, interface_dist_th)
    if wt_data is None:
        return None
    mt_data = process_one_complex(mt_file, mut_site, interface_dist_th)
    if mt_data is None:
        return None
    item['wt'] = wt_data
    item['mt'] = mt_data
    return item


def main(args):
    df = pd.read_csv(args.index_path, sep='\t')
    df.set_index("SAMPLE_ID", inplace=True)

    files = os.listdir(args.data_dir)
    complex_file_names = []
    ddG_list = []
    mut_site_list = []
    for f in files:
        if f in df.index:
            ddG = df.loc[f]["DDG.EXP"]
            dirname = os.path.join(args.data_dir, f)
            mut_site = int(df.loc[f]["MUTATION"][1:-1])
            complex_file_names.append(dirname)
            ddG_list.append(ddG)
            mut_site_list.append(mut_site)

    print_log(f'Preprocessing {len(complex_file_names)} protein files...')
    processed_data = []

    result_list = pmap_multi(process_wt_mt_pair, zip(complex_file_names, mut_site_list, ddG_list), 
                             interface_dist_th=args.interface_dist_th, 
                             n_jobs=args.num_workers, desc='Processing complexes')

    for item in tqdm(result_list, desc="Processing complexes"):
        if item is None:
            continue
        if isinstance(item, list):
            processed_data.extend(item)
        else:
            processed_data.append(item)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    from random import shuffle
    shuffle(processed_data)
    num_valid = len(processed_data) // 10
    train_dataset = processed_data[:-2*num_valid]
    valid_dataset = processed_data[-num_valid:]
    test_dataset = processed_data[-2*num_valid:-num_valid]

    database_out_train = os.path.join(args.out_dir, f'random_split_train.pkl')
    print_log(f'Obtained {len(processed_data)} data after filtering')
    print_log(f'Saving {len(train_dataset)} to {database_out_train} ...')
    with open(database_out_train, 'wb') as f:
        pickle.dump(train_dataset, f)

    database_out_valid = os.path.join(args.out_dir, f'random_split_valid.pkl')
    print_log(f'Saving {len(valid_dataset)} to {database_out_valid} ...')
    with open(database_out_valid, 'wb') as f:
        pickle.dump(valid_dataset, f)
    
    database_out_test = os.path.join(args.out_dir, f'random_split_test.pkl')
    print_log(f'Saving {len(test_dataset)} to {database_out_test} ...')
    with open(database_out_test, 'wb') as f:
        pickle.dump(test_dataset, f)

    print_log('Finished!')


def parse():
    parser = argparse.ArgumentParser(description='Process MdrDB data to create a dataset for mutation DDG prediction')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing MdrDB data folders')
    parser.add_argument('--index_path', type=str, required=True, help='Path to MdrDB index .tsv file')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    parser.add_argument('--num_workers', type=int, default=16)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())