import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
import os
import argparse
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

def read_biolip_pd(file_path):
    df = pd.read_csv(file_path, sep="\t", names=['PDB ID',
                                                 'Receptor chain',
                                                 'Resolution',
                                                 'Binding site number code',
                                                 'Ligand ID',
                                                 'Ligand chain',
                                                 'Ligand serial number',
                                                 'Binding site residues (with PDB residue numbering)',
                                                 'Binding site residues (with residue re-numbered starting from 1)',
                                                 'Catalytic site residues (different sites are separated by ;) (with PDB residue numbering)',
                                                 'Catalytic site residues (different sites are separated by ;) (with residue re-numbered starting from 1)',
                                                 'EC number',
                                                 'GO terms',
                                                 'Binding affinity by manual survey of the original literature.',
                                                 'Binding affinity provided by the Binding MOAD database.',
                                                 'Binding affinity provided by the PDBbind-CN database.',
                                                 'Binding affinity provided by the BindingDB database',
                                                 'UniProt ID',
                                                 'PubMed ID',
                                                 'Residue sequence number of the ligand (field _atom_site.auth_seq_id in PDBx/mmCIF format)',
                                                 'Receptor sequence'])

    return df


def load_biolip_data(file_path, dataset_base_dir, exclude_pdb_idx):
    df = read_biolip_pd(file_path)

    print(f'{len(df)} items in {os.path.basename(file_path)}')
    df = df[~df['PDB ID'].isin(exclude_pdb_idx)]
    print(f'{len(df)} items in {os.path.basename(file_path)} after exclude pdb id')

    # protein file: f{PDB}{Receptor chain}.pdb
    # ligand file: f{PDB}_{Ligand ID}_{Ligand chain}_{Ligand serial number}.pdb
    protein_file_names, ligand_file_names = [], []

    def check_valid(pdb, r_c, rs, l_id, l_c, l_s_n, dataset_base_dir):
        protein_file_name = f'{pdb}{r_c}.pdb'
        ligand_file_name = f'{pdb}_{l_id}_{l_c}_{l_s_n}.pdb'

        if float(rs) > 3.0 or float(rs) == -1.00:
            return None

        receptor_path = os.path.join(dataset_base_dir, 'receptor', protein_file_name)
        ligand_path = os.path.join(dataset_base_dir, 'ligand', ligand_file_name)

        if not os.path.exists(receptor_path) or not os.path.exists(ligand_path):
            return None

        pdb_mol = Chem.MolFromPDBFile(ligand_path, removeHs=False)
        if pdb_mol is not None:
            atom_count = pdb_mol.GetNumAtoms()
        else:
            atom_count = 0
        if atom_count > MAX_ATOM_COUNT or atom_count < 15:
            return None

        return protein_file_name, ligand_file_name

    result_list = pmap_multi(check_valid, zip(df['PDB ID'].values.tolist(),
                                              df['Receptor chain'].values.tolist(),
                                              df['Resolution'].values.tolist(),
                                              df['Ligand ID'].values.tolist(),
                                              df['Ligand chain'].values.tolist(),
                                              df['Ligand serial number'].values.tolist()),
                             dataset_base_dir=dataset_base_dir, n_jobs=16, desc='check BioLiP data validity')

    protein_unique_dict = defaultdict(int)
    for rs in tqdm(result_list):
        if rs is not None:
            protein_file_name, ligand_file_name = rs
            if protein_unique_dict[protein_file_name] == 0:
                protein_file_names.append(protein_file_name)
                ligand_file_names.append(ligand_file_name)
                protein_unique_dict[protein_file_name] += 1

    return df, protein_file_names, ligand_file_names


def ConstructBioLiPDataset(dataset_base_dir, file_path, exclude_pdb_idx_path):
    file_path_base = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).split('.')[0]
    index_path = os.path.join(file_path_base, f'{file_name}_selected_index_max_{MAX_ATOM_COUNT}.pkl')

    with open(exclude_pdb_idx_path, 'r') as f:
        exclude_pdb_idx = f.read().strip().split('\n')

    # if not os.path.exists(index_path):
    df, protein_file_names, ligand_file_names = load_biolip_data(file_path, dataset_base_dir, exclude_pdb_idx)
    print(f'filter out {len(protein_file_names)} items finally.')
    with open(index_path, 'wb') as f:
        pickle.dump((protein_file_names, ligand_file_names), f)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base_dir', type=str, default='./datasets/BioLiP')
    parser.add_argument('--file_path', type=str, default='./datasets/BioLiP/BioLiP.txt')
    parser.add_argument('--exclude_pdb_idx', type=str, default='./dataset/BioLiP/pdbbind.txt')
    parser.add_argument('--max_atom_count', type=int, default=100)

    args = parser.parse_args()
    MAX_ATOM_COUNT = args.max_atom_count

    # get dataset
    ConstructBioLiPDataset(args.dataset_base_dir, args.file_path, args.exclude_pdb_idx)