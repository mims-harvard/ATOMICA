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
    parser.add_argument('--config', type=str, required=True,
                        help='Path of the configuration for training the model')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='gpu to use, -1 for cpu')
    return parser.parse_args()


def main(args):

    # 1. Load configuration
    config: dict = json.load(open(args.config, 'r'))
    print_log(f'General configuration: {config}')
    test_task = config.pop('test_task', 'ALL')

    # 2. create output directory
    out_dir = config.pop('out_dir')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tmp_config_path = os.path.join(out_dir, 'tmp_config.json')
    save_topk = config.get('save_topk', config['max_epoch'])

    # 3. split the PDBbind dataset with 5 seeds
    pdbbind, ppab_dir = config.pop('pdbbind'), config.pop('ppab_dir')
    lba_dir = config.pop('lba_dir')
    pdbbind_dir = os.path.split(pdbbind)[0]
    suffix = os.path.splitext(pdbbind)[-1]
    ppab_test = os.path.join(ppab_dir, 'PPAB_V2.pkl')
    lba_test = os.path.join(lba_dir, 'test')
    

    seeds = [2023, 2022, 2021]
    split_dirs = []
    for i, seed in enumerate(seeds):
        print_log(f'split {i}, seed {seed}')
        split_dir = os.path.join(pdbbind_dir, f'split{i}')
        split_dirs.append(split_dir)
        if not os.path.exists(split_dir):  # split train/valid
            namespace = Namespace(
                data=pdbbind,
                out_dir=split_dir,
                seq_id=0.3,
                valid_ratio=0.1,
                test_ratio=0,  # PPAB is used as the test set
                seed=seed,
                benchmark=ppab_test
            )
            split(namespace)
        # 3. preprocess files into pkl, and add data path / save directory to config
        for split_name in ['valid', 'train']:
            dataset = BlockGeoAffDataset(os.path.join(split_dir, split_name + suffix))
            print_log(f'split {i}: {split_name} lengths {len(dataset)}')
        print()

    round_metrics_ppa, round_metrics_pla = {}, []
    for i, split_dir in enumerate(split_dirs):
        print()
        print_log(f'Start experiment on split {i}')

        # 4. add data path / save directory to config
        for split_name in ['valid', 'train']:
            config[f'{split_name}_set'] = os.path.join(split_dir, split_name + suffix)
        config['train_set2'] = os.path.join(lba_dir, 'train')
        config['valid_set2'] = os.path.join(lba_dir, 'val')
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
        
        for _, ckpt in top_ckpts:  # from best ot worst
            epoch = re.findall(r'epoch(\d+)_', ckpt)[0]
            epoch = int(epoch)
            if best_ckpt is None:  # set to the best checkpoint
                best_ckpt, best_epoch = ckpt, epoch

            if epoch > best_epoch:
                # give prority to the checkpoints of latter checkpoints in the topk
                best_ckpt, best_epoch = ckpt, epoch
            
        # inference
        print_log(f'Inference with checkpoint {best_ckpt}')

        if test_task == 'ALL' or test_task == 'PPA':
            # inference for PPA
            reference_splits = ['rigid', 'medium', 'flexible', 'all']
            result_path = os.path.join(config['save_dir'], f'split{i}_epoch{best_epoch}_results_ppa.jsonl')
            namespace = Namespace(
                test_set=ppab_test,
                task='PPA',
                fragment=config.get('fragment', None),
                pdb_dir=None,   # assume already processed
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

            print_log(f'Results saved to {result_path}')

            # evaluation
            print_log(f'Evaluating PPA...')
            for name in reference_splits:
                namespace = Namespace(predictions=result_path, reference=os.path.join(ppab_dir, f'PPAB_V2_{name}.jsonl'))
                metrics = eval_proc(namespace)
                if name not in round_metrics_ppa:
                    round_metrics_ppa[name] = []
                round_metrics_ppa[name].append(metrics)

        if test_task == 'ALL' or test_task == 'PLA':
            # inference for PLA
            result_path = os.path.join(config['save_dir'], f'split{i}_epoch{best_epoch}_results_pla.jsonl')
            namespace = Namespace(
                test_set=lba_test,
                pdb_dir=None,   # assume already processed
                task='PLA',
                fragment=config.get('fragment', None),
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
            # infer_proc(namespace)
            print_log(f'Results saved to {result_path}')
            # torch.cuda.empty_cache()

            # 7. evaluation
            print_log(f'Evaluating PLA...')
            namespace = Namespace(predictions=result_path, reference=None)
            metrics = eval_proc(namespace)
            round_metrics_pla.append(metrics)

        print_log(f'Split {i} done!')
        print()

    if test_task == 'ALL' or test_task == 'PPA':
        # print results for PPA
        print_log('\nResults PPA:')
        for name in round_metrics_ppa:
            print('=' * 10 + name + '=' * 10)
            for metric_name in round_metrics_ppa[name][0]:
                values = [metrics[metric_name] for metrics in round_metrics_ppa[name]]
                if hasattr(values[0], 'pvalue'):
                    pvalues = [val.pvalue for val in values]
                    values = [val.statistic for val in values]
                    print(f'{metric_name}: {round(np.mean(values), 4)} \pm {round(np.std(values), 4)}')
                    print(f'{metric_name}_pvalue: {np.mean(pvalues)} \pm {np.std(pvalues)}')
                else:
                    print(f'{metric_name}: {round(np.mean(values), 4)} \pm {round(np.std(values), 4)}')
            print()


    if test_task == 'ALL' or test_task == 'PLA':
        # print results for PLA
        print('=' * 10 + f'{config["task"]} Results' + '=' * 10)
        for metric_name in round_metrics_pla[0]:
            values = [metrics[metric_name] for metrics in round_metrics_pla]
            if hasattr(values[0], 'pvalue'):
                pvalues = [val.pvalue for val in values]
                values = [val.statistic for val in values]
                print(f'{metric_name}: {round(np.mean(values), 4)} \pm {round(np.std(values), 4)}')
                print(f'{metric_name}_pvalue: {np.mean(pvalues)} \pm {np.std(pvalues)}')
            else:
                print(f'{metric_name}: {round(np.mean(values), 4)} \pm {round(np.std(values), 4)}')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    print(f'Project directory: {PROJ_DIR}')
    main(parse())