#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from .mmap_dataset import MMAPDataset


class ECDataset(MMAPDataset):

    def __init__(self, mmap_dir: str) -> None:
        super().__init__(mmap_dir)
        self.lengths = [int(x[0]) for x in self._properties]
        self._properties = [torch.tensor([int(i) for i in x[1:]]) for x in self._properties]
        self.n_class = 538

    def get_item_len(self, idx: int):
        return self.lengths[idx]

    def __getitem__(self, idx: int):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Nblock]
            'segment_ids': [Nblock],
        }
        '''
        item = super().__getitem__(idx)
        label = torch.zeros(self.n_class)
        label[self._properties[idx]] = 1
        item['label'] = label
        return item
    
    @classmethod
    def collate_fn(cls, batch):
        results = {
            'X': torch.cat([torch.tensor(item['X'], dtype=torch.float) for item in batch], dim=0),
            'B': torch.cat([torch.tensor(item['B'], dtype=torch.long) for item in batch], dim=0),
            'A': torch.cat([torch.tensor(item['A'], dtype=torch.long) for item in batch], dim=0),
            'atom_positions': torch.cat([torch.tensor(item['atom_positions'], dtype=torch.long) for item in batch], dim=0),
            'block_lengths': torch.cat([torch.tensor(item['block_lengths'], dtype=torch.long) for item in batch], dim=0),
            'segment_ids': torch.cat([torch.tensor(item['segment_ids'], dtype=torch.long) for item in batch], dim=0),
            'lengths': torch.tensor([len(item['B']) for item in batch], dtype=torch.long),
            'label': torch.stack([torch.tensor(item['label'], dtype=torch.float) for item in batch], dim=0), # [batch_size, 538]
        }

        results['X'] = results['X'].unsqueeze(-2)  # number of channel is 1
        return results


if __name__ == '__main__':
    import sys

    dataset = ECDataset(sys.argv[1])
    print(len(dataset))
    length = [len(item['B']) for item in dataset]
    print(f'interface length: min {min(length)}, max {max(length)}, mean {sum(length) / len(length)}')
    atom_length = [len(item['A']) for item in dataset]
    print(f'atom number: min {min(atom_length)}, max {max(atom_length)}, mean {sum(atom_length) / len(atom_length)}')
