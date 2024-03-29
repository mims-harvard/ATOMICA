#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from trainers.abs_trainer import Trainer

from train import create_dataset
from data.pdb_utils import VOCAB


def parse():
    parser = argparse.ArgumentParser(description='inference dG')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the test set')
    parser.add_argument('--task', type=str, default=None, choices=['PPA', 'PLA', 'LEP', 'PDBBind', 'NL', 'PLA_PS', 'LEP_PS', 'PN', 'DDG'],
                        help='PPA: protein-protein affinity, ' + \
                             'PLA: protein-ligand affinity (small molecules), ' + \
                             'LEP: ligand efficacy prediction, ')
    parser.add_argument('--fragment', type=str, default=None, help='fragmentation of small molecules')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


def main(args):
    VOCAB.load_tokenizer(args.fragment)
    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    test_set = create_dataset(args.task, args.test_set, fragment=args.fragment)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=test_set.collate_fn)
    items = test_set.indexes
    
    # save path
    if args.save_path is None:
        save_path = '.'.join(args.ckpt.split('.')[:-1]) + '_results.jsonl'
    else:
        save_path = args.save_path

    fout = open(save_path, 'w')

    idx = 0
    # batch_id = 0
    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            batch = Trainer.to_device(batch, device)

            # for attention visualization
            # model.encoder.encoder.prefix = str(batch_id)
            #edges, edge_attr = model.precalculate_edges(batch)
            results = model.infer(batch)
            if type(results) == tuple:
                results = (res.tolist() for res in results)
                results = (res for res in zip(*results))
            elif torch.numel(results) == 1:
                results = [results.item()]
            else:
                results = results.tolist()
            
            for i, pred_label in enumerate(results):
                item_id = items[idx]['id']
                if args.task == 'DDG':
                    gt = batch[-1][i].item()
                else:
                    gt = items[idx]['label'] if 'label' in items[idx] else items[idx]['affinity']['neglog_aff']
                out_dict = {
                        'id': item_id,
                        'label': pred_label,
                        'task': args.task,
                        'gt': gt
                    }
            
                fout.write(json.dumps(out_dict) + '\n')
                idx += 1
            # batch_id += 1
    
    fout.close()

if __name__ == '__main__':
    main(parse())