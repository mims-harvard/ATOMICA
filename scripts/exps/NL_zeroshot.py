#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import sys
import json
import argparse
from argparse import Namespace
from multiprocessing import Process

import numpy as np

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)

from inference import main as infer_proc
from evaluate import main as eval_proc

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='pdbbind benchmark')
    parser.add_argument('--config', type=str, required=True,
                        help='Path of the configuration for training the model')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='gpu to use, -1 for cpu')
    return parser.parse_args()


def exp(test_set, config, tmp_config_path, save_topk, gpus):
    print_log(f'Configuration: {config}')
    print_log('Start training')
    json.dump(config, open(tmp_config_path, 'w'))
    p = os.popen(f'GPU={",".join([str(gpu) for gpu in gpus])} bash {PROJ_DIR}/scripts/train/train.sh {tmp_config_path}')
    text = p.read()
    p.close()
    top_ckpts = re.findall(r'Validation: ([0-9]*\.?[0-9]+), save path: (.*?)\n', text)
    top_ckpts = sorted(top_ckpts, key=lambda tup: float(tup[0]))[:save_topk]

    # find the optimal checkpoint for inference and evaluation
    best_ckpt, best_epoch, max_topk_epoch = None, -1, 0
    # since some dataset is small and validation loss may fluctuate a lot in the initial phase, we drop the checkpoints within 50% max epoch in the topk checkpoints
    for _, ckpt in top_ckpts:
        epoch = re.findall(r'epoch(\d+)_', ckpt)[0]
        max_topk_epoch = max(int(epoch), max_topk_epoch)

    for _, ckpt in top_ckpts:  # from best ot worst
        epoch = re.findall(r'epoch(\d+)_', ckpt)[0]
        epoch = int(epoch)

        if best_ckpt is None:  # set to the best checkpoint
            best_ckpt, best_epoch = ckpt, epoch

        if epoch > best_epoch:
            # give prority to the checkpoints of latter checkpoints in the topk
            best_ckpt, best_epoch = ckpt, epoch
        
    # 6. inference
    print_log(f'Inference with checkpoint {best_ckpt}')
    result_path = os.path.join(config['save_dir'], f'epoch{best_epoch}_results.jsonl')
    namespace = Namespace(
        test_set=test_set,
        pdb_dir=None,   # assume already processed
        task=config['task'],
        fragment=config.get('fragment', None),
        ckpt=best_ckpt,
        save_path=result_path,
        batch_size=32,
        num_workers=4,
        gpu=gpus[0]
    )

    p = Process(target=infer_proc, args=(namespace,))
    p.start()
    p.join()
    p.close()
    # infer_proc(namespace)
    print_log(f'Results saved to {result_path}')
    # torch.cuda.empty_cache()

    # 7. evaluation
    print_log(f'Evaluating ...')
    namespace = Namespace(predictions=result_path, reference=None)
    metrics = eval_proc(namespace)

    return metrics


def main(args):

    # 1. Load configuration and save to temporary file
    config: dict = json.load(open(args.config, 'r'))
    print_log(f'General configuration: {config}')

    # 2. get configures and delete training-unrelated configs
    save_topk = config.get('save_topk', config['max_epoch'])
    test_set = config.pop('test_set')
    out_dir = config.pop('out_dir')
    tmp_config_path = os.path.join(out_dir, 'tmp_config.json')

    # 3. create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    round_metrics = []
    for seed in [2023, 2022, 2021]:
        config['seed'] = seed
        metrics = exp(test_set, config, tmp_config_path, save_topk, args.gpus)
        round_metrics.append(metrics)

    
    print('=' * 10 + f'{config["task"]} Results' + '=' * 10)
    for metric_name in round_metrics[0]:
        values = [metrics[metric_name] for metrics in round_metrics]
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