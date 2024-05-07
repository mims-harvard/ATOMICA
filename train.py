#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
import json
import numpy as np

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

########### Import your packages below ##########
from data.dataset import BlockGeoAffDataset, PDBBindBenchmark, MixDatasetWrapper, DynamicBatchWrapper, MutationDataset
from data.distributed_sampler import DistributedSamplerResume
from data.atom3d_dataset import LEPDataset, LBADataset
from data.dataset_ec import ECDataset
import models
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB

import wandb

def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, default=None, help='path to valid set')
    parser.add_argument('--pdb_dir', type=str, default=None, help='directory to the complex pdbs (required if not preprocessed in advance)')
    parser.add_argument('--task', type=str, default=None,
                        choices=['PPA', 'PLA', 'AffMix', 'PDBBind', 'NL', 'PN', 'DDG', 'pretrain_gaussian', 'pretrain_torsion'],
                        help='PPA: protein-protein affinity, ' + \
                             'PLA: protein-ligand affinity (small molecules), ' + \
                             'PDBBind: pdbbind benchmark, ' + \
                             'pretrain_gaussian: pretraining with gaussian atom coordinate noise, ' + \
                             'pretrain_torsion: pretraining with torsion angle noise, ')
    parser.add_argument('--train_set2', type=str, default=None, help='path to another train set if task is PretrainMix')
    parser.add_argument('--valid_set2', type=str, default=None, help='path to another valid set if task is PretrainMix')
    parser.add_argument('--train_set3', type=str, default=None, help='path to the third train set')
    parser.add_argument('--valid_set3', type=str, default=None, help='path to the third valid set')
    parser.add_argument('--fragment', type=str, default=None, choices=['PS_300', 'PS_500'], help='fragmentation on small molecules')

    # training related
    parser.add_argument('--pretrain', action='store_true', help='pretraining mode')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='final learning rate')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=None, help='batch size of validation, default set to the same as training batch size')
    parser.add_argument('--max_n_vertex_per_gpu', type=int, default=None, help='if specified, ignore batch_size and form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--max_n_vertex_per_item', type=int, default=None, help='if max_n_vertex_per_gpu is specified, determines maximum number of nodes per item when forming batches with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--valid_max_n_vertex_per_gpu', type=int, default=None, help='form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--patience', type=int, default=-1, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=-1, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--cycle_steps', type=int, default=100000, help='number of steps per cycle in lr_scheduler.CosineAnnealingWarmRestarts')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--edge_size', type=int, default=16, help='Dimension of edge embeddings')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--global_message_passing', action="store_true", default=False, help='message passing between global nodes and normal nodes')
    parser.add_argument('--fragmentation_method', type=str, default=None, choices=['PS_300', 'PS_500'], help='fragmentation method for small molecules')

    # for pretraining
    parser.add_argument('--atom_noise', type=float, default=0, help='apply noise to atom coordinates')
    parser.add_argument('--translation_noise', type=float, default=0, help='apply global translation noise')
    parser.add_argument('--rotation_noise', type=float, default=0, help='apply global rotation noise')
    parser.add_argument('--torsion_noise', type=float, default=0, help='max torsion rotation noise')
    parser.add_argument('--max_rotation', type=float, default=np.pi/4, help='max global rotation angle')
    parser.add_argument('--tr_weight', type=float, default=1.0, help='Weight of translation loss')
    parser.add_argument('--rot_weight', type=float, default=1.0, help='Weight of rotation loss')
    parser.add_argument('--tor_weight', type=float, default=1.0, help='Weight of torsional loss')
    parser.add_argument('--atom_weight', type=float, default=1.0, help='Weight of atom loss')

    # load pretrain
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='path of the pretrained ckpt to load')
    parser.add_argument('--pretrain_state', type=str, default=None, help='path of the pretrained training state to load for resuming training')
    parser.add_argument('--partial_finetune', action="store_true", default=False, help='only finetune energy head')

    # logging
    parser.add_argument('--use_wandb', action="store_true", default=False, help='log to Weights and Biases')
    parser.add_argument('--run_name', type=str, default="test", help='model run name for logging')

    return parser.parse_args()


def create_dataset(task, path, path2=None, path3=None, fragment=None):
    if task == 'PLA':
        # dataset = Atom3DLBA(path)
        dataset = LBADataset(path, fragment=fragment)
        if path2 is not None:  # add protein dataset
            dataset2 = BlockGeoAffDataset(path2)
            dataset = MixDatasetWrapper(dataset, dataset2)
    elif task == 'PPA':
        dataset = BlockGeoAffDataset(path)
        if path2 is not None:  # add small molecule dataset
            dataset2 = LBADataset(path2, fragment=fragment)
            dataset = MixDatasetWrapper(dataset, dataset2)
    elif task == 'AffMix':
        dataset1 = BlockGeoAffDataset(path)
        dataset2 = LBADataset(path2, fragment=fragment)
        dataset = MixDatasetWrapper(dataset1, dataset2)
    elif task == 'PDBBind':
        dataset = PDBBindBenchmark(path)
    elif task == 'NL':
        datasets = [BlockGeoAffDataset(path)]
        if path2 is not None:
            datasets.append(BlockGeoAffDataset(path2))
        if path3 is not None:
            datasets.append(LBADataset(path3, fragment=fragment))
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = MixDatasetWrapper(*datasets)
    elif task == 'PN':
        datasets = [BlockGeoAffDataset(path)]
        if path2 is not None:
            datasets.append(LBADataset(path2, fragment=fragment))
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = MixDatasetWrapper(*datasets)
    elif task == 'DDG':
        dataset = MutationDataset(path)
    elif task == 'pretrain_torsion':
        from data.dataset_pretrain import PretrainTorsionDataset
        dataset1 = PretrainTorsionDataset(path)
        print_log(f'Pretrain dataset {path} size: {len(dataset1)}')
        if path2 is None and path3 is None:
            return dataset1
        datasets = [dataset1]
        if path2 is not None:
            dataset2 = PretrainTorsionDataset(path2)
            datasets.append(dataset2)
            print_log(f'Pretrain dataset {path2} size: {len(dataset2)}')
        if path3 is not None:
            dataset3 = PretrainTorsionDataset(path3)
            datasets.append(dataset3)
            print_log(f'Pretrain dataset {path3} size: {len(dataset3)}')
        dataset = MixDatasetWrapper(*datasets)
        print_log(f'Mixed pretrain dataset size: {len(dataset)}')
    elif task == 'pretrain_gaussian':
        from data.dataset_pretrain import PretrainAtomDataset
        dataset1 = PretrainAtomDataset(path)
        print_log(f'Pretrain dataset {path} size: {len(dataset1)}')
        if path2 is None and path3 is None:
            return dataset1
        datasets = [dataset1]
        if path2 is not None:
            dataset2 = PretrainAtomDataset(path2)
            datasets.append(dataset2)
            print_log(f'Pretrain dataset {path2} size: {len(dataset2)}')
        if path3 is not None:
            dataset3 = PretrainAtomDataset(path3)
            datasets.append(dataset3)
            print_log(f'Pretrain dataset {path3} size: {len(dataset3)}')
        dataset = MixDatasetWrapper(*datasets)
        print_log(f'Mixed pretrain dataset size: {len(dataset)}')
    else:
        raise NotImplementedError(f'Dataset for {task} not implemented!')
    return dataset


def set_noise(dataset, args):
    from data.dataset_pretrain import PretrainAtomDataset, PretrainTorsionDataset
    if type(dataset) == PretrainAtomDataset or type(dataset) == PretrainTorsionDataset:
        if args.atom_noise != 0 and args.torsion_noise != 0:
            raise ValueError('Cannot set both atom and torsion noise at the same time')
        if type(dataset) == PretrainAtomDataset and args.atom_noise != 0:
            dataset.set_atom_noise(args.atom_noise)
        if args.translation_noise != 0:
            dataset.set_translation_noise(args.translation_noise)
        if args.rotation_noise != 0:
            dataset.set_rotation_noise(args.rotation_noise, args.max_rotation)
        if type(dataset) == PretrainTorsionDataset and args.torsion_noise != 0:
            dataset.set_torsion_noise(args.torsion_noise)
    elif type(dataset) == MixDatasetWrapper:
        for d in dataset.datasets:
            set_noise(d, args)


def create_trainer(model, train_loader, valid_loader, config, resume_state=None):
    model_type = type(model)
    if model_type == models.AffinityPredictor:
        trainer = trainers.AffinityTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.DenoisePretrainModel:
        trainer = trainers.PretrainTrainer(
            model, train_loader, valid_loader, config, 
            resume_state=resume_state,
        )
    elif model_type == models.DDGPredictor:
        trainer = trainers.DDGTrainer(model, train_loader, valid_loader, config)
    else:
        raise NotImplementedError(f'Trainer for model type {model_type} not implemented!')
    return trainer


def main(args):
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragment)
    # torch.autograd.set_detect_anomaly(True)
    model = models.create_model(args)

    ########### load your train / valid set ###########
    train_set = create_dataset(args.task, args.train_set, args.train_set2, args.train_set3, args.fragment)
    if args.task in {'pretrain_torsion', 'pretrain_gaussian'}:
        set_noise(train_set, args)
    if args.valid_set is not None:
        valid_set = create_dataset(args.task, args.valid_set, args.valid_set2, args.valid_set3, fragment=args.fragment)
        if args.task in {'pretrain_torsion', 'pretrain_gaussian'}:
            set_noise(valid_set, args)
        print_log(f'Train: {len(train_set)}, validation: {len(valid_set)}')
    else:
        valid_set = None
        print_log(f'Train: {len(train_set)}, no validation')
    if args.max_n_vertex_per_gpu is not None:
        if args.valid_max_n_vertex_per_gpu is None:
            args.valid_max_n_vertex_per_gpu = args.max_n_vertex_per_gpu
        train_set = DynamicBatchWrapper(train_set, args.max_n_vertex_per_gpu, args.max_n_vertex_per_item)
        if valid_set is not None:
            valid_set = DynamicBatchWrapper(valid_set, args.valid_max_n_vertex_per_gpu, args.max_n_vertex_per_item)
        args.batch_size, args.valid_batch_size = 1, 1
        args.num_workers = 1

    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config = trainers.TrainConfig(
        args.save_dir, args.lr, args.max_epoch,
        cycle_steps=args.cycle_steps,
        warmup=args.warmup,
        patience=args.patience,
        grad_clip=args.grad_clip,
        save_topk=args.save_topk,
    )
    config.add_parameter(step_per_epoch=step_per_epoch,
                         final_lr=args.final_lr)
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size

    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = DistributedSamplerResume(train_set, shuffle=args.shuffle, seed=args.seed)
        if args.max_n_vertex_per_gpu is None:
            args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None

    if args.local_rank <= 0:
        if args.max_n_vertex_per_gpu is not None:
            print_log(f'Dynamic batch enabled. Max number of vertex per GPU: {args.max_n_vertex_per_gpu}')
        if args.pretrain_ckpt:
            print_log(f'Loaded pretrained checkpoint from {args.pretrain_ckpt}')
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    if valid_set is not None:
        valid_loader = DataLoader(valid_set, batch_size=args.valid_batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    else:
        valid_loader = None
    trainer = create_trainer(model, train_loader, valid_loader, config, 
                             resume_state=torch.load(args.pretrain_state) if args.pretrain_state else None)
    if args.local_rank <= 0: # only log on the main process
        print_log(f"Saving model checkpoints to: {config.save_dir}")
        os.makedirs(config.save_dir, exist_ok=True)
        with open(os.path.join(config.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)
        if args.use_wandb:
            wandb.init(
                entity="ada-f",
                dir=config.save_dir,
                settings=wandb.Settings(start_method="fork"),
                project="InteractNN",
                name=args.run_name,
                config=vars(args),
            )
    trainer.train(args.gpus, args.local_rank, args.use_wandb)
    
    return trainer.topk_ckpt_map


if __name__ == '__main__':
    args = parse()
    main(args)
