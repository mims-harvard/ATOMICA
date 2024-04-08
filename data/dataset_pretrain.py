import pickle
import torch
import numpy as np
import copy
from collections import defaultdict
from utils.noise_transforms import TorsionNoiseTransform, GaussianNoiseTransform, GlobalRotationTransform, GlobalTranslationTransform

class PretrainTorsionDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        self.data = pickle.load(open(data_file, 'rb'))
        self.indexes = [ {'id': item['id'], 'label': item['affinity']['neglog_aff'] } for item in self.data ]  # to satify the requirements of inference.py
        self.atom_noise, self.global_tr, self.global_rot = None, None, None
        # remove items with no torsion angles in either segment
        cleaned_data = []
        cleaned_indexes = []
        for item, index in zip(self.data, self.indexes):
            if self._can_apply_torsion_noise(item["data"], 0) or self._can_apply_torsion_noise(item["data"], 1):
                cleaned_data.append(item)
                cleaned_indexes.append(index)
        print(f"Removed {len(self.data) - len(cleaned_data)} items with no torsion angles. Original={len(self.data)} Cleaned={len(cleaned_data)}")
        self.data = cleaned_data
        self.indexes = cleaned_indexes
    
    @classmethod
    def _can_apply_torsion_noise(cls, data, chosen_segment):
        return not ((data['torsion_mask'][chosen_segment]['edges'] is None) or all([edges is None for edges in data['torsion_mask'][chosen_segment]['edges']]))

    def set_torsion_noise(self, noise_level):
        self.tor = TorsionNoiseTransform(noise_level)
        
    def set_translation_noise(self, noise_level):
        self.global_tr = GlobalTranslationTransform(noise_level)

    def set_rotation_noise(self, noise_level, max_theta):
        self.global_rot = GlobalRotationTransform(noise_level, max_theta)

    def __len__(self):
        return len(self.data)

    def _get_raw_item(self, idx):
        # apply no noise
        return self.data[idx]

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
            'rot_score': [3]
            'tr_score': [3]
            'tr_eps': [1]
            'tor_score': [n_edges], None if no torsion angles
            'tor_edges': [2, n_edges], [2,0] if no torsion angles
            'noisy_segment': [1]
        }        
        '''
        item = self.data[idx]
        data = copy.deepcopy(item['data'])
        data['label'] = -1  # dummy label

        choices = []

        segment_length = defaultdict(int)
        for segment_id, block_len in zip(data['segment_ids'], data['block_lengths']):
            segment_length[segment_id] += block_len

        # segment length 2 means only one atom + global node, no need to add noise
        if self._can_apply_torsion_noise(data, 0) and segment_length[0] > 2:
            choices.append(0)
        if self._can_apply_torsion_noise(data, 1) and segment_length[1] > 2:
            choices.append(1)
        chosen_segment = np.random.choice(choices)
        
        if self.global_rot is not None:
            # segment length 2 means only one atom + global node, no need to rotate
            if any([segment_length[0] <= 2, segment_length[1] <= 2]):
                rot_score = np.array([0, 0, 0])
            else:
                data, rot_score = self.global_rot(data, chosen_segment)
        else:
            rot_score = np.array([0, 0, 0])
        if self.global_tr is not None: 
            data, tr_score, tr_eps = self.global_tr(data, chosen_segment)
        else:
            tr_score = np.array([0, 0, 0])
            tr_eps = 0
        
        assert self.tor is not None, "Torsion noise transform not set"
        data, tor_score, tor_edges = self.tor(data, chosen_segment)

        data['rot_score'] = rot_score
        data['tr_score'] = tr_score
        data['tr_eps'] = tr_eps
        data['tor_score'] = tor_score
        data['tor_edges'] = tor_edges
        data['noisy_segment'] = chosen_segment

        return data

    @classmethod
    def collate_fn(cls, batch):
        """
        an example of the returned batch
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
            'rot_score': [Nbatch, 3]
            'tr_score': [Nbatch, 3]
            'tr_eps': [Nbatch]
            'tor_score': [n_edges], torch.tensor([]) if no torsion angles
            'tor_edges': [2, n_edges], [2,0] if no torsion angles
            'noisy_segment': [Nbatch]
        }        
        """
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long, torch.float]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)

        res['tor_score'] = []
        for item in batch:
            if item['tor_score'] is not None:
                res['tor_score'].append(torch.tensor(item['tor_score'], dtype=torch.float))
        if len(res['tor_score']) == 0:
            # Sometimes you get a batch with no torsion angles
            res['tor_score'] = torch.zeros(0, dtype=torch.float)
        else:
            res['tor_score'] = torch.cat(res['tor_score'], dim=0)

        keys_scalars = ['rot_score', 'tr_score', 'noisy_segment', 'tr_eps']
        types_scalars = [torch.float, torch.float, torch.long, torch.float]
        for key, _type in zip(keys_scalars, types_scalars):
            val = [item[key] for item in batch]
            val = np.array(val)
            res[key] = torch.tensor(val, dtype=_type)
        res['tor_edges'] = []
        res['tor_batch'] = []
        num_atoms = 0
        for i, item in enumerate(batch):
            res['tor_edges'].append(item['tor_edges'] + num_atoms)
            num_atoms += len(item['A'])
            res['tor_batch'].extend([i for _ in range(item['tor_edges'].shape[1])])
        res['tor_edges'] = np.concatenate(res['tor_edges'], axis=1)
        res['tor_edges'] = torch.tensor(res['tor_edges'], dtype=torch.long)
        res['tor_batch'] = torch.tensor(res['tor_batch'], dtype=torch.long)
        assert res['tor_edges'].shape[1] == res['tor_score'].shape[0] == res['tor_batch'].shape[0], "mismatch in tor score and number of tor edges"
        res['label'] = torch.tensor([-1 for _ in batch], dtype=torch.float)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        res['atom_score'], res['atom_eps'] = None, None # no atom noise
        return res


class PretrainAtomDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        self.data = pickle.load(open(data_file, 'rb'))
        self.indexes = [ {'id': item['id'], 'label': item['affinity']['neglog_aff'] } for item in self.data ]  # to satify the requirements of inference.py
        self.atom_noise, self.global_tr, self.global_rot = None, None, None
    
    def set_atom_noise(self, noise_level):
        self.atom_noise = GaussianNoiseTransform(noise_level)
    
    def set_translation_noise(self, noise_level):
        self.global_tr = GlobalTranslationTransform(noise_level)

    def set_rotation_noise(self, noise_level, max_theta):
        self.global_rot = GlobalRotationTransform(noise_level, max_theta)


    def __len__(self):
        return len(self.data)
    
    def _get_raw_item(self, idx):
        # apply no noise
        return self.data[idx]
    
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
            'rot_score': [1]
            'tr_score': [1]
            'tr_eps': [1]
            'atom_score': [noisy_atoms, 3]
            'atom_eps': [1]
            'noisy_segment': [1]
        }        
        '''
        item = self.data[idx]
        data = copy.deepcopy(item['data'])
        data['label'] = -1  # dummy label

        choices = []
        
        segment_length = defaultdict(int)
        for segment_id, block_len in zip(data['segment_ids'], data['block_lengths']):
            segment_length[segment_id] += block_len

        # segment length 2 means only one atom + global node, no need to add noise
        if segment_length[0] > 2:
            choices.append(0)
        if segment_length[1] > 2:
            choices.append(1)
        chosen_segment = np.random.choice(choices)

        # segment length 2 means only one atom + global node, no need to rotate
        if self.global_rot is not None:
            if any([segment_length[0] <= 2, segment_length[1] <= 2]):
                rot_score = np.array([0, 0, 0])
            else:
                data, rot_score = self.global_rot(data, chosen_segment)
        else:
            rot_score = np.array([0, 0, 0])
        if self.global_tr is not None: 
            data, tr_score, tr_eps = self.global_tr(data, chosen_segment)
        else:
            tr_score = np.array([0, 0, 0])
            tr_eps = 0
        if self.atom_noise is not None:
            data, atom_score, atom_eps = self.atom_noise(data, chosen_segment)
        else:
            atom_score = np.zeros_like(data['X'])
            atom_eps = 0

        data['rot_score'] = rot_score
        data['tr_score'] = tr_score
        data['tr_eps'] = tr_eps
        data['atom_score'] = atom_score
        data['atom_eps'] = atom_eps
        data['noisy_segment'] = chosen_segment

        return data
    

    @classmethod
    def collate_fn(cls, batch):
        # FIXME: what to do when tor is empty?
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids', 'atom_score']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long, torch.float]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        keys_scalars = ['rot_score', 'tr_score', 'noisy_segment', 'tr_eps', 'atom_eps']
        types_scalars = [torch.float, torch.float, torch.long, torch.float, torch.float]
        for key, _type in zip(keys_scalars, types_scalars):
            val = [item[key] for item in batch]
            val = np.array(val)
            res[key] = torch.tensor(val, dtype=_type)
        res['label'] = torch.tensor([-1 for _ in batch], dtype=torch.float)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        res['tor_edges'], res['tor_score'], res['tor_batch'] = None, None, None # no tor noise
        return res
    