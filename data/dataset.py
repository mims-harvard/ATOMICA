#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import pickle
import argparse
from tqdm.contrib.concurrent import process_map
from os.path import basename, splitext
from typing import List

import numpy as np
import torch
from torch_scatter import scatter_mean, scatter_sum

from utils.logger import print_log

########## import your packages below ##########
from .pdb_utils import Atom, VOCAB, dist_matrix_from_coords

class Block:
    def __init__(self, symbol: str, units: List[Atom]) -> None:
        self.symbol = symbol
        self.units = units

    def __len__(self):
        return len(self.units)
    
    def __iter__(self):
        return iter(self.units)

    def to_data(self):
        b = VOCAB.symbol_to_idx(self.symbol)
        x, a, positions = [], [], []
        for atom in self.units:
            a.append(VOCAB.atom_to_idx(atom.get_element()))
            x.append(atom.get_coord())
            positions.append(VOCAB.atom_pos_to_idx(atom.get_pos_code()))
        block_len = len(self)
        return b, a, x, positions, block_len
        

def blocks_to_data(*blocks_list: List[List[Block]]):
    B, A, X, atom_positions, block_lengths, segment_ids = [], [], [], [], [], []
    for i, blocks in enumerate(blocks_list):
        if len(blocks) == 0:
            continue
        # global node
        cur_B = [VOCAB.symbol_to_idx(VOCAB.GLB)]
        cur_A = [VOCAB.get_atom_global_idx()]
        cur_X = [None]
        cur_atom_positions = [VOCAB.get_atom_pos_global_idx()]
        cur_block_lengths = [1]
        # other nodes
        for block in blocks:
            b, a, x, positions, block_len = block.to_data()
            cur_B.append(b)
            cur_A.extend(a)
            cur_X.extend(x)
            cur_atom_positions.extend(positions)
            cur_block_lengths.append(block_len)
        # update coordinates of the global node to the center
        cur_X[0] = np.mean(cur_X[1:], axis=0)
        cur_segment_ids = [i for _ in cur_B]
        
        # finish these blocks
        B.extend(cur_B)
        A.extend(cur_A)
        X.extend(cur_X)
        atom_positions.extend(cur_atom_positions)
        block_lengths.extend(cur_block_lengths)
        segment_ids.extend(cur_segment_ids)

    data = {
        'X': np.array(X),   # [Natom, 2, 3]
        'B': B,             # [Nb], block (residue) type
        'A': A,             # [Natom]
        'atom_positions': atom_positions,  # [Natom]
        'block_lengths': block_lengths,  # [Nresidue]
        'segment_ids': segment_ids,      # [Nresidue]
    }

    return data


def data_to_blocks(data, fragmentation_method=None):
    if fragmentation_method:
        VOCAB.load_tokenizer(fragmentation_method)
    curr_atom_idx = 0
    list_of_blocks = []
    curr_segment_id = 0
    curr_blocks = []
    for block_idx, block in enumerate(data['B']):
        symbol = VOCAB.idx_to_symbol(block)
        if symbol == VOCAB.GLB:
            curr_atom_idx += data['block_lengths'][block_idx]
            continue
        atom_coords = data['X'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]
        atom_positions = data['atom_positions'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]
        atoms = []
        for i, atom in enumerate(data['A'][curr_atom_idx:curr_atom_idx+data['block_lengths'][block_idx]]):
            atom_name=VOCAB.idx_to_atom(atom)
            if atom_name == VOCAB.atom_global:
                continue
            element=VOCAB.idx_to_atom(atom)
            coordinate=atom_coords[i]
            pos_code=VOCAB.idx_to_atom_pos(atom_positions[i])
            atoms.append(Atom(atom_name=atom_name, element=element, coordinate=coordinate, pos_code=pos_code))
        curr_atom_idx += data['block_lengths'][block_idx]
        if data['segment_ids'][block_idx] != curr_segment_id:
            list_of_blocks.append(curr_blocks)
            curr_blocks = []
            curr_segment_id = data['segment_ids'][block_idx]
        curr_blocks.append(Block(symbol, atoms))
    list_of_blocks.append(curr_blocks)
    return list_of_blocks


def blocks_to_coords(blocks: List[Block]):
    max_n_unit = 0
    coords, masks = [], []
    for block in blocks:
        coords.append([unit.get_coord() for unit in block.units])
        max_n_unit = max(max_n_unit, len(coords[-1]))
        masks.append([1 for _ in coords[-1]])
    
    for i in range(len(coords)):
        num_pad =  max_n_unit - len(coords[i])
        coords[i] = coords[i] + [[0, 0, 0] for _ in range(num_pad)]
        masks[i] = masks[i] + [0 for _ in range(num_pad)]
    
    return np.array(coords), np.array(masks).astype('bool')  # [N, M, 3], [N, M], M == max_n_unit, in mask 0 is for padding


def df_to_blocks(df, key_residue='residue', key_insertion_code='insertion_code', key_resname='resname',
                     key_atom_name='atom_name', key_element='element', key_x='x', key_y='y', key_z='z') -> List[Block]:
    last_res_id, last_res_symbol = None, None
    blocks, units = [], []
    for row in df.itertuples():  # each row is an atom (unit)
        residue = getattr(row, key_residue)
        if key_insertion_code is None:
            res_id = str(residue)
        else:
            insert_code = getattr(row, key_insertion_code)
            res_id = f'{residue}{insert_code}'.rstrip()
        if res_id != last_res_id:  # one block ended
            # if last_res_symbol == VOCAB.UNK:
            #     print('unk')
            #     print([str(a) for a in units])
            block = Block(last_res_symbol, units)
            blocks.append(block)
            # clear
            units = []
            last_res_id = res_id
            last_res_symbol = VOCAB.abrv_to_symbol(getattr(row, key_resname))
        atom = getattr(row, key_atom_name)
        element = getattr(row, key_element)
        if element == 'H':
            continue
        units.append(Atom(atom, [getattr(row, axis) for axis in [key_x, key_y, key_z]], element))
    blocks = blocks[1:]
    blocks.append(Block(last_res_symbol, units))
    return blocks


def blocks_interface(blocks1, blocks2, dist_th, return_indexes=False):
    blocks_coord, blocks_mask = blocks_to_coords(blocks1 + blocks2)
    blocks1_coord, blocks1_mask = blocks_coord[:len(blocks1)], blocks_mask[:len(blocks1)]
    blocks2_coord, blocks2_mask = blocks_coord[len(blocks1):], blocks_mask[len(blocks1):]
    dist = dist_matrix_from_coords(blocks1_coord, blocks1_mask, blocks2_coord, blocks2_mask)
    
    on_interface = dist < dist_th
    indexes1 = np.nonzero(on_interface.sum(axis=1) > 0)[0]
    indexes2 = np.nonzero(on_interface.sum(axis=0) > 0)[0]

    blocks1 = [blocks1[i] for i in indexes1]
    blocks2 = [blocks2[i] for i in indexes2]

    if return_indexes:
        return blocks1, blocks2, indexes1, indexes2
    else:
        return blocks1, blocks2


class BlockGeoAffDataset(torch.utils.data.Dataset):

    def __init__(self, data_file, database=None, dist_th=6, n_cpu=4, suffix=''):
        '''
        data_file: path to the dataset file, can be index file in some occasions
        database: path/directory containing the complete data
        dist_th: threshold for deciding the interacting environment (minimum distance between heavy atoms of residues)
        n_cpu: number of cpus used in parallel preprocessing
        '''
        super().__init__()
        self.dist_th = dist_th
        self.data_file = os.path.abspath(data_file)
        self.database = database
        proc_file = os.path.join(
            os.path.split(data_file)[0],
            basename(splitext(data_file)[0]) + f'.{type(self).__name__}{suffix}_processed.pkl'
        )
        self.proc_file = proc_file
        need_process = True
        if os.path.exists(proc_file):
            print_log(f'Loading preprocessed data from {proc_file}...')
            with open(proc_file, 'rb') as fin:
                th, indexes, data = pickle.load(fin)
            if th == dist_th:
                self.indexes = indexes
                self.data = data
                need_process = False
        if need_process:
            print_log('Preprocessing...')
            items = self._load_data_file()
            if isinstance(items, list):
                data = process_map(self._preprocess, items, max_workers=n_cpu, chunksize=10)
            else:  # LMDB
                print('Data not list, disable parallel processing')
                from tqdm import tqdm
                data = [self._preprocess(item) for i, item in enumerate(tqdm(items))]
            self.indexes, self.data = self._post_process(items, data)
            with open(proc_file, 'wb') as fout:
                pickle.dump((dist_th, self.indexes, self.data), fout)
            print_log(f'Preprocessed data saved to {proc_file}')
    
    def _load_data_file(self):
        with open(self.data_file, 'rb') as fin:
            items = pickle.load(fin)
        return items
    
    def _post_process(self, items, processed_data):
        indexes = [ { 'id': item['id'], 'affinity': item['affinity'] } for item, d in zip(items, processed_data) if d is not None ]
        data = [d for d in processed_data if d is not None]
        return indexes, data

    def _preprocess(self, item):
        blocks1 = df_to_blocks(item['atoms_interface1'], key_atom_name='name')
        blocks2 = df_to_blocks(item['atoms_interface2'], key_atom_name='name')

        data = blocks_to_data(blocks1, blocks2)

        data['label'] = item['affinity']['neglog_aff']

        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Nblock]
            'segment_ids': [Nblock]
            'label': [1]
        }
        '''
        item = self.data[idx]
        return item
    
    @classmethod
    def filter_for_segment(cls, data, keep_segment):
        # segment_id = 0, protein
        # segment_id = 1, ligand
        for k, v in data.items():
            if type(v) is list:
                data[k] = np.array(v)

        block_mask = data['segment_ids'] == keep_segment
        block_id = np.zeros_like(data["A"]) # [Nu]
        block_id[np.cumsum(data["block_lengths"], axis=0)[:-1]] = 1
        block_id = np.cumsum(block_id, axis=0)
        atom_mask = data["segment_ids"][block_id] == keep_segment

        new_data = {}
        new_data['X'] = data['X'][atom_mask].tolist()
        new_data['B'] = data['B'][block_mask].tolist()
        new_data['A'] = data['A'][atom_mask].tolist()
        new_data['atom_positions'] = data['atom_positions'][atom_mask].tolist()
        new_data['block_lengths'] = data['block_lengths'][block_mask].tolist()
        new_data['segment_ids'] = data['segment_ids'][block_mask].tolist()
        for key in data:
            if key in ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']:
                continue
            new_data[key] = data[key]
        return new_data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        res['label'] = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res
    

class MutationDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        self.data = pickle.load(open(data_file, 'rb'))
        self.indexes = [ {'id': item['id'], 'label': item['ddG'] } for item in self.data ]  # to satify the requirements of inference.py
        for item in self.data:
            item['wt']["esm_embeddings"] = item["wt_esm_block_embeddings"]
            item['mt']["esm_embeddings"] = item["mt_esm_block_embeddings"]
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        an example of the returned data
        [
            # wild type
            {
                'X': [Natom, 3],
                'B': [Nblock],
                'A': [Natom],
                'atom_positions': [Natom],
                'block_lengths': [Natom]
                'segment_ids': [Nblock]
            },
            # mutated
            {
                'X': [Natom, 3],
                'B': [Nblock],
                'A': [Natom],
                'atom_positions': [Natom],
                'block_lengths': [Natom]
                'segment_ids': [Nblock]
            },
            # label
            [1]
        ]
        '''
        return self.data[idx]['wt'], self.data[idx]['mt'], torch.tensor(self.data[idx]['ddG'], dtype=torch.float), torch.tensor(self.data[idx]["mt_block_indexes"], dtype=torch.long)
        #torch.tensor(self.data[idx]['wt_binding_affinity'], dtype=torch.float), torch.tensor(self.data[idx]['mt_binding_affinity'], dtype=torch.float)

    @classmethod
    def collate_fn(cls, batch):
        wt = [item[0] for item in batch]
        mt = [item[1] for item in batch]
        label = [item[2] for item in batch]
        cumsum_block_lengths = torch.cumsum(torch.tensor([0] + [len(mt_item['B']) for mt_item in mt], dtype=torch.long), dim=0)
        mt_block_indexes = torch.cat([item[3] + cumsum_block_lengths[i] for i, item in enumerate(batch)])
        batch = cls.collate_fn_(wt+mt)
        return batch, torch.tensor(label, dtype=torch.float), mt_block_indexes

    @classmethod
    def collate_fn_(cls, batch):
        keys = ['X', 'B', 'A', 'block_lengths', 'segment_ids', "esm_embeddings"]
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long, torch.float]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                if key != "esm_embeddings":
                    val.append(torch.tensor(item[key], dtype=_type))
                else:
                    val.append(item[key])
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)

        block_id = torch.zeros_like(res['A']) # [Nu]
        block_id[torch.cumsum(res['block_lengths'], dim=0)[:-1]] = 1
        block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)
        res['Z_block'] = scatter_mean(res['X'], block_id, dim=0)
        
        del res['A']
        del res['X']
        del res['block_lengths']
        return res
    
    
class PDBBindBenchmark(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        self.data = pickle.load(open(data_file, 'rb'))
        self.indexes = [ {'id': item['id'], 'label': item['affinity']['neglog_aff'] } for item in self.data ]  # to satify the requirements of inference.py

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
            'label': [1]
        }        
        '''
        item = self.data[idx]
        data = item['data']
        data['label'] = item['affinity']['neglog_aff']

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        res['label'] = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


class PDBDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        self.indexes = [ item['id'] for item in self.data ]  # to satify the requirements of inference.py

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
        }        
        '''
        item = self.data[idx]
        data = item['data']

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


class LabelledPDBDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        self.indexes = [ item['id'] for item in self.data ]  # to satify the requirements of inference.py

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
            'label': [1]
        }        
        '''
        item = self.data[idx]
        data = item['data']
        if "label" in item.keys():
            data["label"] = item["label"]
        else:
            data['label'] = item['affinity']['neglog_aff']
        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids', 'label']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long, torch.float]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            if key == 'label':
                res[key] = torch.tensor(val, dtype=_type)
            else:
                res[key] = torch.cat(val, dim=0)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


class MultiClassLabelledPDBDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        self.indexes = [ item['id'] for item in self.data ]  # to satify the requirements of inference.py

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
            'label': [1]
        }        
        '''
        item = self.data[idx]
        data = item['data']
        data["label"] = item["label"]

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids', 'label']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            if key == 'label':
                res[key] = torch.tensor(val, dtype=_type)
            else:
                res[key] = torch.cat(val, dim=0)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, *datasets) -> None:
        super().__init__()
        self.datasets = datasets
        self.cum_len = []
        self.total_len = 0
        for dataset in datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        self.collate_fn = self.datasets[0].collate_fn
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i].__getitem__(idx - last_cum_len)
            last_cum_len = cum_len
        return None


class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, max_n_vertex_per_batch, max_n_vertex_per_item=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset))]
        self.max_n_vertex_per_batch = max_n_vertex_per_batch
        if max_n_vertex_per_item is None:
            max_n_vertex_per_item = max_n_vertex_per_batch
        self.max_n_vertex_per_item = max_n_vertex_per_item
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_vertex_cnt = 0
        batch = []
        
        num_too_large = 0

        for i in self.indexes:
            data = self.dataset[i]
            item_len = len(data['B']) if 'B' in data else data['len']
            if item_len > self.max_n_vertex_per_item:
                num_too_large += 1
                continue
            cur_vertex_cnt += item_len
            if cur_vertex_cnt > self.max_n_vertex_per_batch:
                self.batch_indexes.append(batch)
                batch = []
                cur_vertex_cnt = item_len
            batch.append(i)
        self.batch_indexes.append(batch)
        print_log(f'Number of items too large: {num_too_large} out of {len(self.indexes)}. Remaining: {len(self.indexes) - num_too_large} batches. Created {len(self.batch_indexes)} batches.')

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)


def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--database', type=str, default=None, help='directory of pdb data')
    return parser.parse_args()
 

if __name__ == '__main__':
    args = parse()
    dataset = BlockGeoAffDataset(args.dataset)
    print(len(dataset))
    length = [len(item['B']) for item in dataset]
    print(f'interface length: min {min(length)}, max {max(length)}, mean {sum(length) / len(length)}')
    atom_length = [len(item['A']) for item in dataset]
    print(f'atom number: min {min(atom_length)}, max {max(atom_length)}, mean {sum(atom_length) / len(atom_length)}')

    item = dataset[0]
    print(item)