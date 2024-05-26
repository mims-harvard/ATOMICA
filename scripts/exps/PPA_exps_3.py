#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import gc
import os
import sys
import json
import argparse
from argparse import Namespace
from multiprocessing import Process

import numpy as np
# import torch

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
# print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.split import main as split
from data.dataset import BlockGeoAffDataset

from inference import main as infer_proc
from evaluate import main as eval_proc

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind')
    parser.add_argument('--pdbbind', type=str, required=True,
                        help='Path to PDBbind data containing PDBbind.pkl')
    parser.add_argument('--ppab_dir', type=str, required=True,
                        help='Directory to the PPAB test set containing PPAB_V2.pkl and PPAB_V2_all/rigid/medium/flexible.jsonl')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory. Default the same as the pdbbind dir')
    parser.add_argument('--config', type=str, required=True,
                        help='Path of the configuration for training the model')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='gpu to use, -1 for cpu')
    parser.add_argument('--test_same_split', action='store_true', help='Add flag to run three different seeds on the same train/val split')
    return parser.parse_args()


def main(args):

    if args.out_dir is None:
        args.out_dir = os.path.split(args.pdbbind)[0]
    elif not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. Load configuration and save to temporary file
    config: dict = json.load(open(args.config, 'r'))
    print_log(f'General configuration: {config}')
    tmp_config_path = os.path.join(args.out_dir, 'tmp_config.json')
    save_topk = config.get('save_topk', config['max_epoch'])

    # 2. split the PDBbind dataset with 5 seeds
    suffix = os.path.splitext(args.pdbbind)[-1]
    test_set = os.path.join(args.ppab_dir, 'PPAB_V2.pkl')

    # seeds = [2023, 2022, 2021, 2020, 2019]
    seeds = [2023, 2022, 2021]
    split_dirs = []
    out_split_dirs = []
    for i, seed in enumerate(seeds):
        print_log(f'split {i}, seed {seed}')
        split_dir = os.path.join(os.path.split(args.pdbbind)[0], f'split{i}')
        out_split_dir = os.path.join(args.out_dir, f'split{i}')
        split_dirs.append(split_dir)
        out_split_dirs.append(out_split_dir)
        if not os.path.exists(split_dir):  # split train/valid
            namespace = Namespace(
                data=args.pdbbind,
                out_dir=out_split_dir,
                seq_id=0.3,
                valid_ratio=0.1,
                test_ratio=0,  # PPAB is used as the test set
                seed=seed,
                benchmark=test_set
            )
            split(namespace)
        # 3. preprocess files into pkl, and add data path / save directory to config
        for split_name in ['valid', 'train']:
            dataset = BlockGeoAffDataset(os.path.join(split_dir, split_name + suffix))
            print_log(f'split {i}: {split_name} lengths {len(dataset)}')
        print()

    round_metrics = {}
    for i, split_dir in enumerate(split_dirs):
        if args.test_same_split:
            split_dir = split_dirs[0] # test the same split
            print()
            print_log(f'Start experiment on split 0 with seed {seeds[i]}')
        else:
            print()
            print_log(f'Start experiment on split {i}')

        # 4. add data path / save directory to config
        for split_name in ['valid', 'train']:
            config[f'{split_name}_set'] = os.path.join(split_dir, split_name + suffix)
        config['save_dir'] = f'{out_split_dirs[i]}/models/'
        config['seed'] = seeds[i]

        # training
        print_log(f'Configuration: {config}')
        print_log('Start training')
        json.dump(config, open(tmp_config_path, 'w'))
        p = os.popen(f'GPU={",".join([str(gpu) for gpu in args.gpus])} bash {PROJ_DIR}/scripts/train/train.sh {tmp_config_path}')
        text = p.read()
        p.close()
        top_ckpts = re.findall(r'Validation: ([0-9]*\.?[0-9]+), save path: (.*?)\n', text)
        top_ckpts = sorted(top_ckpts, key=lambda tup: float(tup[0]))[:save_topk]

        # find the optimal checkpoint for inference and evaluation: the latest checkpoint in topk
        best_ckpt, best_epoch = None, -1
        # # since some dataset is small and validation loss may fluctuate a lot in the initial phase, we drop the checkpoints within 50% max epoch in the topk checkpoints
        # for _, ckpt in top_ckpts:
        #     epoch = re.findall(r'epoch(\d+)_', ckpt)[0]
        #     max_topk_epoch = max(int(epoch), max_topk_epoch)

        # epoch_th = 0.8 * max_topk_epoch
        for _, ckpt in top_ckpts:  # from best ot worst
            epoch = re.findall(r'epoch(\d+)_', ckpt)[0]
            epoch = int(epoch)
            # if epoch < epoch_th:
            #     continue
            # else:
            #     best_ckpt, best_epoch = ckpt, epoch
            #     break
            if best_ckpt is None:  # set to the best checkpoint
                best_ckpt, best_epoch = ckpt, epoch

            if epoch > best_epoch:
                # give prority to the checkpoints of latter checkpoints in the topk
                best_ckpt, best_epoch = ckpt, epoch
                # break
            
        # inference
        print_log(f'Inference with checkpoint {best_ckpt}')
        reference_splits = ['all'] # 'rigid', 'medium', 'flexible', 
        result_path = os.path.join(config['save_dir'], f'split{i}_epoch{best_epoch}_results.jsonl')
        namespace = Namespace(
            test_set=test_set,
            task=config['task'],
            pdb_dir=None,   # assume already processed
            fragment=None,
            ckpt=best_ckpt,
            save_path=result_path,
            batch_size=32,
            num_workers=4,
            gpu=args.gpus[0]
        )

        p = Process(target=infer_proc, args=(namespace,))
        p.start()
        p.join()
        p.close()
        #infer_proc(namespace)

        print_log(f'Results saved to {result_path}')
        # gc.collect()
        # with torch.no_grad():
        #     torch.cuda.empty_cache()

        # evaluation
        print_log(f'Evaluating ...')
        for name in reference_splits:
            namespace = Namespace(predictions=result_path, reference=None) #os.path.join(args.ppab_dir, f'PPAB_V2_{name}.jsonl'))
            metrics = eval_proc(namespace)
            if name not in round_metrics:
                round_metrics[name] = []
            round_metrics[name].append(metrics)

        print_log(f'Split {i} done!')
        print()

    # print results
    print_log('\nResults:')
    for name in round_metrics:
        print('=' * 10 + name + '=' * 10)
        for metric_name in round_metrics[name][0]:
            values = [metrics[metric_name] for metrics in round_metrics[name]]
            if hasattr(values[0], 'pvalue'):
                pvalues = [val.pvalue for val in values]
                values = [val.statistic for val in values]
                print(f'{metric_name}: {round(np.mean(values), 4)} \pm {round(np.std(values), 4)}')
                print(f'{metric_name}_pvalue: {np.mean(pvalues)} \pm {np.std(pvalues)}')
            else:
                print(f'{metric_name}: {round(np.mean(values), 4)} \pm {round(np.std(values), 4)}')
        print()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    print(f'Project directory: {PROJ_DIR}')
    main(parse())