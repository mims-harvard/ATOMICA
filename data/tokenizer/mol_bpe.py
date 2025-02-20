# Source https://github.com/THUNLP-MT/GET

import json
from copy import copy
import argparse
import multiprocessing as mp
from tqdm import tqdm

from .molecule import Molecule
from .chem_utils import smi2mol, mol2smi, get_submol
from .chem_utils import cnt_atom, MAX_VALENCE


'''classes below are used for principal subgraph extraction'''

class MolInSubgraph:
    def __init__(self, mol, kekulize=False):
        self.mol = mol
        self.smi = mol2smi(mol)
        self.kekulize = kekulize
        self.subgraphs, self.subgraphs_smis = {}, {}  # pid is the key (init by all atom idx)
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()
            self.subgraphs[idx] = { idx: symbol }
            self.subgraphs_smis[idx] = symbol
        self.inversed_index = {} # assign atom idx to pid
        self.upid_cnt = len(self.subgraphs)
        for aid in range(mol.GetNumAtoms()):
            for key in self.subgraphs:
                subgraph = self.subgraphs[key]
                if aid in subgraph:
                    self.inversed_index[aid] = key
        self.dirty = True
        self.smi2pids = {} # private variable, record neighboring graphs and their pids

    def get_nei_subgraphs(self):
        nei_subgraphs, merge_pids = [], []
        for key in self.subgraphs:
            subgraph = self.subgraphs[key]
            local_nei_pid = []
            for aid in subgraph:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in subgraph or nei_idx > aid:   # only consider connecting to former atoms
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_subgraph = copy(subgraph)
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        return nei_subgraphs, merge_pids
    
    def get_nei_smis(self):
        if self.dirty:
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            nei_smis, self.smi2pids = [], {}
            for i, subgraph in enumerate(nei_subgraphs):
                submol = get_submol(self.mol, list(subgraph.keys()), kekulize=self.kekulize)
                smi = mol2smi(submol)
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis

    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.subgraphs and pid2 in self.subgraphs: # possibly del by former
                    self.subgraphs[pid1].update(self.subgraphs[pid2])
                    self.subgraphs[self.upid_cnt] = self.subgraphs[pid1]
                    self.subgraphs_smis[self.upid_cnt] = smi
                    # self.subgraphs_smis[pid1] = smi
                    for aid in self.subgraphs[pid2]:
                        self.inversed_index[aid] = pid1
                    for aid in self.subgraphs[pid1]:
                        self.inversed_index[aid] = self.upid_cnt
                    del self.subgraphs[pid1]
                    del self.subgraphs[pid2]
                    del self.subgraphs_smis[pid1]
                    del self.subgraphs_smis[pid2]
                    self.upid_cnt += 1
        self.dirty = True   # mark the graph as revised

    def get_smis_subgraphs(self):
        # return list of tuple(smi, idxs)
        res = []
        for pid in self.subgraphs_smis:
            smi = self.subgraphs_smis[pid]
            group_dict = self.subgraphs[pid]
            idxs = list(group_dict.keys())
            res.append((smi, idxs))
        return res


def freq_cnt(mol):
    freqs = {}
    nei_smis = mol.get_nei_smis()
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += 1
    return freqs, mol


def graph_bpe(fname, vocab_len, vocab_path, cpus, kekulize):
    # load molecules
    print(f'Loading mols from {fname} ...')
    with open(fname, 'r') as fin:
        smis = list(map(lambda x: x.strip(), fin.readlines()))
    # init to atoms
    mols = []
    for smi in tqdm(smis):
        try:
            mol = MolInSubgraph(smi2mol(smi, kekulize), kekulize)
            mols.append(mol)
        except Exception as e:
            print(f'Error: Parsing {smi} failed. Skip.')
    # loop
    selected_smis, details = list(MAX_VALENCE.keys()), {}   # details: <smi: [atom cnt, frequency]
    # calculate single atom frequency
    for atom in selected_smis:
        details[atom] = [1, 0]  # frequency of single atom is not calculated
    for smi in smis:
        cnts = cnt_atom(smi, return_dict=True)
        for atom in details:
            if atom in cnts:
                details[atom][1] += cnts[atom]
    # bpe process
    add_len = vocab_len - len(selected_smis)
    print(f'Added {len(selected_smis)} atoms, {add_len} principal subgraphs to extract')
    pbar = tqdm(total=add_len)
    pool = mp.Pool(cpus)
    while len(selected_smis) < vocab_len:
        res_list = pool.map(freq_cnt, mols)  # each element is (freq, mol) (because mol will not be synced...)
        freqs, mols = {}, []
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        # find the subgraph to merge
        max_cnt, merge_smi = 0, ''
        for smi in freqs:
            cnt = freqs[smi]
            if cnt > max_cnt:
                max_cnt = cnt
                merge_smi = smi
        # merge
        for mol in mols:
            mol.merge(merge_smi)
        if merge_smi in details:  # corner case: re-extracted from another path
            continue
        selected_smis.append(merge_smi)
        details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
        pbar.update(1)
    pbar.close()
    print('sorting vocab by atom num')
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    pool.close()
    with open(vocab_path, 'w') as fout:
        fout.write(json.dumps({'kekulize': kekulize}) + '\n')
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))
    return selected_smis, details


class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # load kekulize config
        config = json.loads(lines[0])
        self.kekulize = config['kekulize']
        lines = lines[1:]
        
        self.vocab_dict = {}
        self.idx2subgraph, self.subgraph2idx = [], {}
        self.max_num_nodes = 0
        for line in lines:
            smi, atom_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(atom_num), int(freq))
            self.max_num_nodes = max(self.max_num_nodes, int(atom_num))
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        # for fine-grained level (atom level)
        self.max_num_nodes += 2 # start, padding
    
    def tokenize(self, mol):
        if isinstance(mol, str):
            mol = smi2mol(mol, self.kekulize)
        rdkit_mol = mol
        mol = MolInSubgraph(mol, kekulize=self.kekulize)
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_subgraphs()
        # construct reversed index
        aid2pid = {}
        for pid, subgraph in enumerate(res):
            _, aids = subgraph
            for aid in aids:
                aid2pid[aid] = pid
        # construct adjacent matrix
        ad_mat = [[0 for _ in res] for _ in res]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                i, j = aid2pid[aid], aid2pid[nei_id]
                if i != j:
                    ad_mat[i][j] = ad_mat[j][i] = 1
        group_idxs = [x[1] for x in res]
        return Molecule(rdkit_mol, group_idxs, self.kekulize)

    def idx_to_subgraph(self, idx):
        return self.idx2subgraph[idx]
    
    def subgraph_to_idx(self, subgraph):
        return self.subgraph2idx[subgraph]
    
    def atom_vocab(self):
        return copy(self.atom_level_vocab)

    def num_subgraph_type(self):
        return len(self.idx2subgraph)
    
    def atom_pos_pad_idx(self):
        return self.max_num_nodes - 1
    
    def atom_pos_start_idx(self):
        return self.max_num_nodes - 2

    def __call__(self, mol):
        return self.tokenize(mol)
    
    def __len__(self):
        return len(self.idx2subgraph)

def parse():
    parser = argparse.ArgumentParser(description='Principal subgraph extraction motivated by bpe')
    parser.add_argument('--smiles', type=str, default='COc1cc(C=NNC(=O)c2ccc(O)cc2O)ccc1OCc1ccc(Cl)cc1',
                        help='The molecule to tokenize (example)')
    parser.add_argument('--data', type=str, required=True, help='Path to molecule corpus')
    parser.add_argument('--vocab_size', type=int, default=500, help='Length of vocab')
    parser.add_argument('--output', type=str, required=True, help='Path to save vocab')
    parser.add_argument('--workers', type=int, default=16, help='Number of cpus to use')
    parser.add_argument('--kekulize', action='store_true', help='Whether to kekulize the molecules (i.e. replace aromatic bonds with alternating single and double bonds)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    graph_bpe(args.data, vocab_len=args.vocab_size, vocab_path=args.output,
              cpus=args.workers, kekulize=args.kekulize)
    tokenizer = Tokenizer(args.output)
    print(f'Example: {args.smiles}')
    mol = tokenizer.tokenize(args.smiles)
    print('Tokenized mol: ')
    print(mol)
    print('Reconstruct smiles to make sure it is right: ')
    smi = mol.to_smiles()
    print(smi)
    assert smi == args.smiles
    print('Assertion test passed')
    mol.to_SVG('example.svg')
