import pickle
import torch
import numpy as np
import copy
from collections import defaultdict
from utils.noise_transforms import TorsionNoiseTransform, GaussianNoiseTransform, GlobalRotationTransform, GlobalTranslationTransform, CropTransform
from torch_scatter import scatter_mean
from tqdm import tqdm
from .dataset import open_data_file

class PretrainMaskedDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, mask_proportion, mask_token, atom_mask_token, vocab_to_mask):
        super().__init__()
        self.data = open_data_file(data_file)
        self.indexes = [ {'id': item['id']} for item in self.data ]
        self.mask_proportion = mask_proportion
        self.mask_token = mask_token
        self.atom_mask_token = atom_mask_token
        self.vocab_to_mask = vocab_to_mask # list of vocab indices that can be masked
        self.idx_to_mask_block = dict(zip(self.vocab_to_mask, range(len(self.vocab_to_mask))))
        self.crop = None
        self.preprocess()
    
    def set_crop(self, max_n_vertex_per_item, fragmentation_method):
        self.crop = CropTransform(max_n_vertex_per_item-2, fragmentation_method) # 2 blocks are global blocks
        for item in tqdm(self.data, desc="Preprocessing, cropping large items", total=len(self.data)):
            data = item['data']
            if len(data['B']) > self.crop.max_blocks:
                data, keep_blocks = self.crop(data)
                item['data'] = data
        self.preprocess()

    def preprocess(self):
        missing_maskable_nodes = []
        for idx, item in enumerate(self.data):
            data = item["data"]
            can_mask0 = item["data"]["can_mask"] = np.where(np.logical_and(np.isin(np.array(data['B']), np.array(self.vocab_to_mask)), 
                    np.array(data["segment_ids"])==0))[0].tolist()
            can_mask1 = item["data"]["can_mask"] = np.where(np.logical_and(np.isin(np.array(data['B']), np.array(self.vocab_to_mask)), 
                    np.array(data["segment_ids"])==1))[0].tolist()
            item["data"]["can_mask"] = [can_mask0, can_mask1]
            if len(can_mask0) == 0 or len(can_mask1) == 0:
                missing_maskable_nodes.append(idx)
        print(f"Removed {len(missing_maskable_nodes)} items with no maskable nodes. Original={len(self.data)} Cleaned={len(self.data) - len(missing_maskable_nodes)}")
        self.data = [self.data[i] for i in range(len(self.data)) if i not in missing_maskable_nodes]
        self.indexes = [self.indexes[i] for i in range(len(self.indexes)) if i not in missing_maskable_nodes]
    
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
            'masked_blocks': [Nblock]
            'label': [Nmasked_blocks]
        }        
        '''
        item = self.data[idx]
        data = copy.deepcopy(item['data'])
        B = np.array(data['B'])
        # mask blocks on the non-noisy side to not interfere with the noised torsion angles
        can_mask = item["data"]["can_mask"][0] + item["data"]["can_mask"][1]
        num_to_select = max(1, int(self.mask_proportion * len(can_mask)))
        selected_indices = np.random.choice(can_mask, size=num_to_select, replace=False)
        masked_blocks = np.zeros_like(data['B'], dtype=bool)
        masked_blocks[selected_indices] = True

        block_ids = sum([block_len*[block_id] for block_id, block_len in enumerate(data['block_lengths'])], [])
        block_centers = scatter_mean(torch.tensor(data['X'], dtype=torch.float), torch.tensor(block_ids, dtype=torch.long), dim=0).tolist()
        masked_A, masked_X = [], []
        old_A_map = {}
        curr_block = 0
        curr_A = 0
        for block_id in range(len(B)):
            block_len = data['block_lengths'][block_id]
            if masked_blocks[block_id]:
                for A_i in range(block_len):
                    old_A_map[curr_A+A_i] = len(masked_A)
                masked_A.append(self.atom_mask_token)
                masked_X.append(block_centers[block_id])
            else:
                for A_i in range(block_len):
                    old_A_map[curr_A+A_i] = len(masked_A) + A_i
                masked_A.extend(data['A'][curr_block:curr_block+block_len])
                masked_X.extend(data['X'][curr_block:curr_block+block_len])
            curr_block += block_len
            curr_A += block_len
        data['A'] = masked_A
        data['X'] = masked_X

        block_lengths = np.array(data['block_lengths'])
        block_lengths[masked_blocks] = 1
        data['block_lengths'] = block_lengths.tolist()

        data['masked_labels'] = [self.idx_to_mask_block[b] for b in B[masked_blocks].tolist()]
        new_blocks = B
        new_blocks[masked_blocks] = self.mask_token
        data['B'] = new_blocks.tolist()
        data['masked_blocks'] = masked_blocks.tolist()
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
            'masked_blocks': [Nblock]
            'label': [Nmasked_blocks]
        }        
        """
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids', 'masked_blocks', 'masked_labels']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long, torch.bool, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
                res[key] = torch.cat(val, dim=0)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res

class PretrainTorsionDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        self.data = open_data_file(data_file)
        self.indexes = [ {'id': item['id']} for item in self.data ]  # to satify the requirements of inference.py
        self.tor, self.global_tr, self.global_rot, self.crop = None, None, None, None
        # remove items with no torsion angles in either segment
        self.preprocess()
    
    def preprocess(self):
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
    
    def set_crop(self, max_n_vertex_per_item, fragmentation_method):
        # crop all items before training
        self.crop = CropTransform(max_n_vertex_per_item-2, fragmentation_method) # 2 blocks are global blocks 
        for item in tqdm(self.data, desc="Preprocessing, cropping large items", total=len(self.data)):
            data = item['data']
            if len(data['B']) > self.crop.max_blocks:
                data, keep_blocks = self.crop(data)
                item['data'] = data
        self.preprocess()

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
        if 'modality' in item.keys():
            data['modality'] = item['modality']
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
        # coord_change = np.linalg.norm(data["X"] - item["data"]["X"], axis=1)
        # print(f'Change in atomic position: mean {np.mean(coord_change):.2f}, max {np.max(coord_change):.2f}, min {np.min(coord_change):.2f}')
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

        has_block_embeddings = 'block_embeddings' in batch[0]
        has_block_embeddings_separate = 'block_embeddings0' in batch[0] and 'block_embeddings1' in batch[0]
        if has_block_embeddings:
            keys.append('block_embeddings')
            types.append(torch.float)
        elif has_block_embeddings_separate:
            keys.append('block_embeddings0')
            keys.append('block_embeddings1')
            types.append(torch.float)
            types.append(torch.float)

        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                if isinstance(item[key], np.ndarray):
                    item[key] = item[key].tolist()
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
        res['label'] = torch.tensor([x['label'] for x in batch], dtype=torch.float)
        if 'modality' in batch[0].keys():
            res['modality'] = torch.tensor([x['modality'] for x in batch], dtype=torch.long)
        else:
            res['modality'] = None
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        res['atom_score'], res['atom_eps'] = None, None # no atom noise
        return res


class PretrainMaskedTorsionDataset(PretrainTorsionDataset):
    def __init__(self, data_file, mask_proportion, mask_token, atom_mask_token, vocab_to_mask):
        self.data_file = data_file
        self.data = open_data_file(data_file)
        self.indexes = [ {'id': item['id']} for item in self.data ]  # to satify the requirements of inference.py
        self.tor, self.global_tr, self.global_rot, self.crop = None, None, None, None
        self.mask_proportion = mask_proportion
        self.mask_token = mask_token
        self.atom_mask_token = atom_mask_token
        self.vocab_to_mask = vocab_to_mask # list of vocab indices that can be masked
        self.idx_to_mask_block = dict(zip(self.vocab_to_mask, range(len(self.vocab_to_mask))))
        self.preprocess()

    def preprocess(self):
        super().preprocess()
        missing_maskable_nodes = []
        for idx, item in enumerate(self.data):
            item['data'] = self.get_mask_for_item(item['data'])
            can_mask0, can_mask1 = item["data"]["can_mask"]
            if len(can_mask0) == 0 or len(can_mask1) == 0:
                missing_maskable_nodes.append(idx)
        print(f"Removed {len(missing_maskable_nodes)} items with no maskable nodes. Original={len(self.data)} Cleaned={len(self.data) - len(missing_maskable_nodes)}")
        self.data = [self.data[i] for i in range(len(self.data)) if i not in missing_maskable_nodes]
        self.indexes = [self.indexes[i] for i in range(len(self.indexes)) if i not in missing_maskable_nodes]
    
    def get_mask_for_item(self, data):
        can_mask0 = np.where(np.logical_and(np.isin(np.array(data['B']), np.array(self.vocab_to_mask)), 
                np.array(data["segment_ids"])==0))[0].tolist()
        can_mask1 = np.where(np.logical_and(np.isin(np.array(data['B']), np.array(self.vocab_to_mask)), 
                np.array(data["segment_ids"])==1))[0].tolist()
        data["can_mask"] = [can_mask0, can_mask1]
        return data
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['X'] = data['X'].tolist()
        B = np.array(data['B'])
        # # mask blocks on the non noisy side
        # if data['noisy_segment'] == 0:
        #     can_mask = data["can_mask"][1]
        # else:
        #     can_mask = data["can_mask"][0]
        can_mask = data["can_mask"][0] + data["can_mask"][1]
        num_to_select = max(1, int(self.mask_proportion * len(can_mask)))
        selected_indices = np.random.choice(can_mask, size=num_to_select, replace=False)
        masked_blocks = np.zeros_like(data['B'], dtype=bool)
        masked_blocks[selected_indices] = True

        block_ids = sum([block_len*[block_id] for block_id, block_len in enumerate(data['block_lengths'])], [])
        block_centers = scatter_mean(torch.tensor(data['X'], dtype=torch.float), torch.tensor(block_ids, dtype=torch.long), dim=0).tolist()
        masked_A, masked_X = [], []
        old_A_map = {}
        curr_block = 0
        curr_A = 0
        for block_id in range(len(B)):
            block_len = data['block_lengths'][block_id]
            if masked_blocks[block_id]:
                for A_i in range(block_len):
                    old_A_map[curr_A+A_i] = len(masked_A)
                masked_A.append(self.atom_mask_token)
                masked_X.append(block_centers[block_id])
            else:
                for A_i in range(block_len):
                    old_A_map[curr_A+A_i] = len(masked_A) + A_i
                masked_A.extend(data['A'][curr_block:curr_block+block_len])
                masked_X.extend(data['X'][curr_block:curr_block+block_len])
            curr_block += block_len
            curr_A += block_len
        data['A'] = masked_A
        data['X'] = masked_X
        def map_atoms(x):
            return old_A_map[x]  # Return the mapped value, or the original if not found
        vectorized_map_values = np.vectorize(map_atoms)
        data['tor_edges'] = vectorized_map_values(data['tor_edges'])

        block_lengths = np.array(data['block_lengths'])
        block_lengths[masked_blocks] = 1
        data['block_lengths'] = block_lengths.tolist()

        data['masked_labels'] = [self.idx_to_mask_block[b] for b in B[masked_blocks].tolist()]
        new_blocks = B
        new_blocks[masked_blocks] = self.mask_token
        data['B'] = new_blocks.tolist()
        data['masked_blocks'] = masked_blocks.tolist()

        if 'block_embeddings' in data:
            block_embeddings = np.array(data['block_embeddings'])
            masked_block_embeddings = np.zeros_like(block_embeddings[0])
            block_embeddings[masked_blocks] = masked_block_embeddings
            data['block_embeddings'] = block_embeddings.tolist()
        elif 'block_embeddings0' in data and 'block_embeddings1' in data:
            block_embeddings0 = np.array(data['block_embeddings0'])
            block_embeddings1 = np.array(data['block_embeddings1'])
            masked_block_embeddings0 = np.zeros_like(block_embeddings0[0])
            masked_block_embeddings1 = np.zeros_like(block_embeddings1[0])
            masked_blocks0 = np.logical_and(masked_blocks, np.array(data['segment_ids']) == 0)
            masked_blocks1 = np.logical_and(masked_blocks, np.array(data['segment_ids']) == 1) - np.sum(np.array(data['segment_ids']) == 0)
            block_embeddings0[masked_blocks0] = masked_block_embeddings0
            block_embeddings1[masked_blocks1] = masked_block_embeddings1
            data['block_embeddings0'] = block_embeddings0.tolist()
            data['block_embeddings1'] = block_embeddings1.tolist()
        return data

    @classmethod
    def collate_fn(cls, batch):
        res = super().collate_fn(batch)
        keys = ['masked_blocks', 'masked_labels']
        types = [torch.bool, torch.long]
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
                res[key] = torch.cat(val, dim=0)
        return res

class PretrainAtomDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        self.data = open_data_file(data_file)
        self.indexes = [ {'id': item['id']} for item in self.data ]  # to satify the requirements of inference.py
        self.atom_noise, self.global_tr, self.global_rot = None, None, None
    
    def set_atom_noise(self, noise_level):
        self.atom_noise = GaussianNoiseTransform(noise_level)
    
    def set_translation_noise(self, noise_level):
        self.global_tr = GlobalTranslationTransform(noise_level)

    def set_rotation_noise(self, noise_level, max_theta):
        self.global_rot = GlobalRotationTransform(noise_level, max_theta)
    
    def set_crop(self, max_n_vertex_per_item, fragmentation_method):
        self.crop = CropTransform(max_n_vertex_per_item-2, fragmentation_method) # 2 blocks are global blocks
        for item in tqdm(self.data, desc="Preprocessing, cropping large items", total=len(self.data)):
            data = item['data']
            if len(data['B']) > self.crop.max_blocks:
                data, keep_blocks = self.crop(data)
                item['data'] = data

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
        res['label'] = torch.tensor([x['label'] for x in batch], dtype=torch.float)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        res['tor_edges'], res['tor_score'], res['tor_batch'] = None, None, None # no tor noise
        return res
    

class NoisyNodesTorsionDataset(PretrainTorsionDataset):
    def __init__(self, data_file):
        super().__init__(data_file)
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if 'label' in self.data[idx]:
            item['label'] = self.data[idx]['label']
        else:
            item['label'] = self.data[idx]['affinity']['neglog_aff'] # replace dummy label of PretrainTorsionDataset
        return item