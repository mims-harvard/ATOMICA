#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import wandb

from .utils.logger import print_log
from .utils.random_seed import setup_seed, SEED
from .data.dataset import (
    PDBBindBenchmark, MixDatasetWrapper, DynamicBatchWrapper,
    BalancedDynamicBatchWrapper, PretrainBalancedDynamicBatchWrapper,
    LabelledPDBDataset, MultiClassLabelledPDBDataset,
    ProtInterfaceDataset, DistillationDatasetWrapper, ResidueDistillationDatasetWrapper,
    PocketEmbeddingDatasetWrapper
)
from .data.distributed_sampler import DistributedSamplerResume
from . import models
from . import trainers
from .utils.nn_utils import count_parameters
from .data import VOCAB


def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, default=None, help='path to valid set')
    parser.add_argument('--task', type=str, required=True, default=None,
                        choices=['pretrain_torsion', 'pretrain_torsion_masking', 'pretrain_gaussian',
                                 'binary_classifier', 'multiclass_classifier', 'residue_binary_classifier', 'masking', 'multilabel_classifier',
                                 'PDBBind', 'prot_interface', 'regression'])
    parser.add_argument('--num_classifier_classes', type=int, default=None, help='number of classes for task=multiclass_classifier')
    parser.add_argument('--train_set2', type=str, default=None, help='path to another train set if task is PretrainMix')
    parser.add_argument('--valid_set2', type=str, default=None, help='path to another valid set if task is PretrainMix')
    parser.add_argument('--train_set3', type=str, default=None, help='path to the third train set')
    parser.add_argument('--valid_set3', type=str, default=None, help='path to the third valid set')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=None, help='final learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs where validation loss is not used for early stopping')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-5, help='linear learning rate warmup start lr')
    parser.add_argument('--warmup_end_lr', type=float, default=1e-3, help='linear learning rate warmup end lr')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients with too big norm')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for optimizer')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=None, help='batch size of validation, default set to the same as training batch size')
    parser.add_argument('--max_n_vertex_per_gpu', type=int, default=None, help='if specified, ignore batch_size and form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--max_n_vertex_per_item', type=int, default=None, help='if max_n_vertex_per_gpu is specified, larger items will be randomly cropped')
    parser.add_argument('--valid_max_n_vertex_per_gpu', type=int, default=None, help='form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--balanced_sampler', action='store_true', default=False, help='use balanced sampler')
    parser.add_argument('--patience', type=int, default=-1, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=-1, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--cycle_steps', type=int, default=100000, help='number of steps per cycle in lr_scheduler.CosineAnnealingWarmRestarts')
    parser.add_argument('--random_block_sampling', action='store_true', default=False, help='enable random block sampling augmentation during training')
    parser.add_argument('--block_sampling_p_keep', type=float, default=1.0, help='probability of keeping each non-global block when using random block sampling (0.0 to 1.0)')
    parser.add_argument('--block_sampling_p_none', type=float, default=0.0, help='probability of skipping augmentation (returning original data unchanged) when using random block sampling (0.0 to 1.0)')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--atom_hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--block_hidden_size', type=int, default=128, help='dimension of hidden states for blocks')
    parser.add_argument('--edge_size', type=int, default=16, help='Dimension of edge embeddings')
    parser.add_argument('--k_neighbors', type=int, default=8, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--bottom_global_message_passing', action="store_true", default=False, help='message passing between global nodes and normal nodes at the bottom level')
    parser.add_argument('--global_message_passing', action="store_true", default=False, help='message passing between global nodes and normal nodes at the top level')
    parser.add_argument('--fragmentation_method', type=str, default=None, choices=['PS_300'], help='fragmentation method for small molecules')

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
    parser.add_argument('--mask_proportion', type=float, default=0, help='block masking rate')
    parser.add_argument('--mask_weight', type=float, default=1.0, help='block masking rate')
    parser.add_argument('--noisy_nodes_weight', type=float, default=0, help='coefficient for denoising loss during finetuning')
    parser.add_argument('--modality_embedding', action="store_true", default=False, help='add embedding for each modality')

    # load pretrain
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='path of the pretrained ckpt to load')
    parser.add_argument('--pretrain_config', type=str, default=None, help='path of the pretrained config to load')
    parser.add_argument('--pretrain_weights', type=str, default=None, help='path of the pretrained weights to load')
    parser.add_argument('--pretrain_state', type=str, default=None, help='path of the pretrained training state to load for resuming training')
    parser.add_argument('--partial_finetune', action="store_true", default=False, help='only finetune energy head')

    # for prediction
    parser.add_argument('--pred_dropout', type=float, default=0.0, help='dropout rate for prediction')
    parser.add_argument('--pred_nonlinearity', type=str, default='relu', choices=['relu', 'gelu', 'elu'], help='nonlinearity for prediction')
    parser.add_argument('--num_pred_layers', type=int, default=3, help='number of layers for prediction')
    parser.add_argument('--pred_hidden_size', type=int, default=32, help='hidden size for prediction')
    parser.add_argument('--num_projector_layers', type=int, default=3, help='number of layers for projector')
    parser.add_argument('--projector_hidden_size', type=int, default=256, help='hidden size for projector')
    parser.add_argument('--projector_dropout', type=float, default=0.0, help='dropout rate for projector')
    parser.add_argument('--block_embedding_size', type=int, default=None, help='embedding size for blocks')
    parser.add_argument('--block_embedding0_size', type=int, default=None, help='embedding size for blocks in segment0, block_embedding_size1 will be used for blocks in segment1')
    parser.add_argument('--block_embedding1_size', type=int, default=None, help='embedding size for blocks in segment1, block_embedding_size0 will be used for blocks in segment0')

    # logging
    parser.add_argument('--use_wandb', action="store_true", default=False, help='log to Weights and Biases')
    parser.add_argument('--use_raytune', action="store_true", default=False, help='log to RayTune')
    parser.add_argument('--run_name', type=str, default="test", help='model run name for logging')
    parser.add_argument('--multiclass_metric', type=str, default=None, choices=['auprc', 'f1_macro'],
                       help='metric to use for classification: auprc or f1_macro. Both options are supported for multiclass and multilabel classification.')
    parser.add_argument('--weighted_loss', action='store_true', default=False,
                       help='use weighted cross entropy loss for multiclass classification. Only valid for MultiClassClassifierModel')

    # knowledge distillation
    parser.add_argument('--teacher_logits_file', type=str, default=None,
                       help='path to parquet file containing teacher logits with "id" and "teacher_logits" columns for knowledge distillation')
    parser.add_argument('--distillation_alpha', type=float, default=0.5,
                       help='weight for distillation loss vs supervised loss. Total loss = (1-alpha)*supervised + alpha*distillation. Default: 0.5')
    parser.add_argument('--distillation_temperature', type=float, default=1.0,
                       help='temperature for softening probability distributions in distillation. Higher values create softer distributions. Default: 1.0')

    # pocket embeddings
    parser.add_argument('--pocket_embeddings_train_file', type=str, default=None,
                       help='path to .npy file containing pocket embeddings for training set (e.g., from RNAFM, RNA-FM, ESM). Embedding dimension is auto-detected.')
    parser.add_argument('--pocket_embeddings_val_file', type=str, default=None,
                       help='path to .npy file containing pocket embeddings for validation set.')

    # focal loss
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                       help='use focal loss instead of cross-entropy for classification tasks. Helps with class imbalance.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='focusing parameter for focal loss. Higher values focus more on hard examples. Default: 2.0')
    parser.add_argument('--focal_alpha', type=float, nargs='*', default=None,
                       help='class weighting for focal loss. Can be a single value or per-class weights. If not specified, no weighting is applied.')

    return parser.parse_args()


def create_dataset(task, path, path2=None, path3=None, fragment=None, random_block_sampling=False, p_keep=1.0, seed=None, p_none=0.0):    
    if task == 'pretrain_torsion':
        from atomica.data.dataset_pretrain import PretrainTorsionDataset
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
    elif task == 'pretrain_torsion_masking':
        from atomica.data.dataset_pretrain import PretrainMaskedTorsionDataset
        dataset_args = {
            "mask_proportion": 0,
            "mask_token": VOCAB.symbol_to_idx(VOCAB.MASK),
            "vocab_to_mask": [VOCAB.symbol_to_idx(x[0]) for x in VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags],
            "atom_mask_token": VOCAB.get_atom_mask_idx(),
        }
        dataset1 = PretrainMaskedTorsionDataset(path, **dataset_args)
        print_log(f'Pretrain dataset {path} size: {len(dataset1)}')
        if path2 is None and path3 is None:
            return dataset1
        datasets = [dataset1]
        if path2 is not None:
            dataset2 = PretrainMaskedTorsionDataset(path2, **dataset_args)
            datasets.append(dataset2)
            print_log(f'Pretrain dataset {path2} size: {len(dataset2)}')
        if path3 is not None:
            dataset3 = PretrainMaskedTorsionDataset(path3, **dataset_args)
            datasets.append(dataset3)
            print_log(f'Pretrain dataset {path3} size: {len(dataset3)}')
        dataset = MixDatasetWrapper(*datasets)
        print_log(f'Mixed pretrain dataset size: {len(dataset)}')
    elif task == 'pretrain_gaussian':
        from atomica.data.dataset_pretrain import PretrainAtomDataset
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
    elif task == 'binary_classifier' or task == 'regression' or task == 'residue_binary_classifier':
        dataset = LabelledPDBDataset(path, random_block_sampling=random_block_sampling, p_keep=p_keep, seed=seed, p_none=p_none)
        datasets = [dataset]
        if path2 is not None:
            dataset2 = LabelledPDBDataset(path2, random_block_sampling=random_block_sampling, p_keep=p_keep, seed=seed, p_none=p_none)
            datasets.append(dataset2)
        if path3 is not None:
            dataset3 = LabelledPDBDataset(path3, random_block_sampling=random_block_sampling, p_keep=p_keep, seed=seed, p_none=p_none)
            datasets.append(dataset3)
        if len(datasets) > 1:
            dataset = MixDatasetWrapper(*datasets)
    elif task == 'multiclass_classifier' or task == 'multilabel_classifier':
        dataset = MultiClassLabelledPDBDataset(path, random_block_sampling=random_block_sampling, p_keep=p_keep, seed=seed, p_none=p_none)
        datasets = [dataset]
        if path2 is not None:
            dataset2 = MultiClassLabelledPDBDataset(path2, random_block_sampling=random_block_sampling, p_keep=p_keep, seed=seed, p_none=p_none)
            datasets.append(dataset2)
        if path3 is not None:
            dataset3 = MultiClassLabelledPDBDataset(path3, random_block_sampling=random_block_sampling, p_keep=p_keep, seed=seed, p_none=p_none)
            datasets.append(dataset3)
        if len(datasets) > 1:
            dataset = MixDatasetWrapper(*datasets)
    elif task == "masking":
        from atomica.data.dataset_pretrain import PretrainMaskedDataset
        dataset_args = {
            "mask_proportion": 0,
            "mask_token": VOCAB.symbol_to_idx(VOCAB.MASK),
            "vocab_to_mask": [VOCAB.symbol_to_idx(x[0]) for x in VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags],
            "atom_mask_token": VOCAB.get_atom_mask_idx(),
        }
        dataset = PretrainMaskedDataset(path, **dataset_args)
        datasets = [dataset]
        if path2 is not None:
            dataset2 = PretrainMaskedDataset(path2, **dataset_args)
            datasets.append(dataset2)
        if path3 is not None:
            dataset3 = PretrainMaskedDataset(path3, **dataset_args)
            datasets.append(dataset3)
        if len(datasets) > 1:
            dataset = MixDatasetWrapper(*datasets)
    elif task == 'PDBBind':
        dataset = PDBBindBenchmark(path)
        if path2 is not None or path3 is not None:
            raise NotImplementedError('ProtInterfaceDataset does not support multiple datasets')
    elif task == "prot_interface":
        dataset = ProtInterfaceDataset(path)
        if path2 is not None or path3 is not None:
            raise NotImplementedError('ProtInterfaceDataset does not support multiple datasets')
    else:
        raise NotImplementedError(f'Dataset for {task} not implemented!')
    return dataset


def set_noise(dataset, args):
    from atomica.data.dataset_pretrain import PretrainAtomDataset, PretrainTorsionDataset, PretrainMaskedDataset, PretrainMaskedTorsionDataset
    if type(dataset) in [PretrainAtomDataset, PretrainTorsionDataset, PretrainMaskedTorsionDataset]:
        if args.atom_noise != 0 and args.torsion_noise != 0:
            raise ValueError('Cannot set both atom and torsion noise at the same time')
        if type(dataset) == PretrainAtomDataset and args.atom_noise != 0:
            dataset.set_atom_noise(args.atom_noise)
        if args.translation_noise != 0:
            dataset.set_translation_noise(args.translation_noise)
        if args.rotation_noise != 0:
            dataset.set_rotation_noise(args.rotation_noise, args.max_rotation)
        if args.max_n_vertex_per_item is not None:
            dataset.set_crop(args.max_n_vertex_per_item, args.fragmentation_method)
        if type(dataset) in [PretrainTorsionDataset, PretrainMaskedTorsionDataset] and args.torsion_noise != 0:
            dataset.set_torsion_noise(args.torsion_noise)
        if type(dataset) == PretrainMaskedTorsionDataset:
            dataset.mask_proportion = args.mask_proportion
    elif type(dataset) == PretrainMaskedDataset:
        dataset.mask_proportion = args.mask_proportion
    elif type(dataset) == MixDatasetWrapper:
        new_datasets = []
        for d in dataset.datasets:
            d = set_noise(d, args)
            new_datasets.append(d)
        dataset = MixDatasetWrapper(*new_datasets) # update the mix dataset wrapper with new dataset lengths
    return dataset


def create_trainer(model, train_loader, valid_loader, config, resume_state=None):
    model_type = type(model)
    if model_type in [models.AffinityPredictor, models.RegressionPredictor]:
        trainer = trainers.AffinityTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.ClassifierModel or model_type == models.ResidueClassifierModel:
        trainer = trainers.ClassifierTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.MultiClassClassifierModel or model_type == models.MultiLabelClassifierModel:
        trainer = trainers.MultiClassClassifierTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.DenoisePretrainModel:
        if model.masking_objective:
            trainer = trainers.PretrainMaskingNoisingTrainer(
                model, train_loader, valid_loader, config, 
                resume_state=resume_state,  
            )
        else:
            trainer = trainers.PretrainTrainer(
                model, train_loader, valid_loader, config, 
                resume_state=resume_state,
            )
    elif model_type == models.DenoisePretrainModelWithBlockEmbedding:
        trainer = trainers.PretrainMaskingNoisingTrainerWithBlockEmbedding(
            model, train_loader, valid_loader, config, 
            resume_state=resume_state,
        )
    elif model_type == models.MaskedNodeModel:
        trainer = trainers.MaskingTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.ProteinInterfaceModel:
        trainer = trainers.ProtInterfaceTrainer(model, train_loader, valid_loader, config)
    else:
        raise NotImplementedError(f'Trainer for model type {model_type} not implemented!')
    return trainer


def main(args):
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragmentation_method)
    # torch.autograd.set_detect_anomaly(True)
    if args.task == "pretrain_torsion_masking" or args.task == "masking":
        args.num_nodes = len(VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags)
    else:
        args.num_nodes = None
    # Validate weighted_loss is only used for multiclass_classifier or multilabel_classifier
    if args.weighted_loss and args.task not in ['multiclass_classifier', 'multilabel_classifier']:
        raise ValueError(f"weighted_loss option can only be used for multiclass_classifier or multilabel_classifier task, but got task={args.task}")

    # Auto-detect pocket embedding size if provided (before model creation)
    if args.pocket_embeddings_train_file is not None:
        if args.task not in {'multiclass_classifier', 'multilabel_classifier'}:
            raise ValueError(f"Pocket embeddings are only supported for multiclass_classifier and multilabel_classifier tasks, but got task={args.task}")
        print_log(f'Loading pocket embeddings to detect embedding size from {args.pocket_embeddings_train_file}')
        pocket_emb = np.load(args.pocket_embeddings_train_file)
        args.pocket_embedding_size = pocket_emb.shape[1] if pocket_emb.ndim == 2 else 1
        print_log(f'Auto-detected pocket embedding size: {args.pocket_embedding_size}')
        del pocket_emb  # Free memory
    else:
        args.pocket_embedding_size = None

    model = models.create_model(args)

    ########### load your train / valid set ###########
    if args.task == 'PLA_noisy_nodes':
        train_task = 'PLA_noisy_nodes_train'
    else:
        train_task = args.task
    train_set = create_dataset(train_task, args.train_set, args.train_set2, args.train_set3, args.fragmentation_method, 
                               random_block_sampling=args.random_block_sampling, p_keep=args.block_sampling_p_keep, seed=args.seed, p_none=args.block_sampling_p_none)
    if args.task in {'pretrain_torsion', 'pretrain_gaussian', 'masking', 'PLA_noisy_nodes', 'pretrain_torsion_masking'}:
        train_set = set_noise(train_set, args)
    if args.valid_set is not None:
        valid_set = create_dataset(args.task, args.valid_set, args.valid_set2, args.valid_set3, fragment=args.fragmentation_method,
                                   random_block_sampling=False, p_keep=1.0)  # Disable augmentation for validation
        if args.task in {'pretrain_torsion', 'pretrain_gaussian', 'masking', 'pretrain_torsion_masking'}:
            valid_set = set_noise(valid_set, args)
        print_log(f'Train: {len(train_set)}, validation: {len(valid_set)}')
    else:
        valid_set = None
        print_log(f'Train: {len(train_set)}, no validation')

    # Wrap datasets with DistillationDatasetWrapper if teacher logits are provided
    if args.teacher_logits_file is not None:
        if args.task not in {'binary_classifier', 'multiclass_classifier', 'multilabel_classifier', 'residue_binary_classifier'}:
            raise ValueError(f"Knowledge distillation is only supported for classification tasks, but got task={args.task}")
        print_log(f'Enabling knowledge distillation with teacher logits from {args.teacher_logits_file}')
        print_log(f'Distillation alpha: {args.distillation_alpha}, temperature: {args.distillation_temperature}')

        # Use appropriate wrapper based on task type
        if args.task == 'residue_binary_classifier':
            # For residue-level tasks, use ResidueDistillationDatasetWrapper
            print_log('Using ResidueDistillationDatasetWrapper for residue-level task')
            train_set = ResidueDistillationDatasetWrapper(train_set, args.teacher_logits_file)
        else:
            # For graph-level tasks, use standard DistillationDatasetWrapper
            train_set = DistillationDatasetWrapper(train_set, args.teacher_logits_file)
        # Note: We don't wrap validation set with teacher logits since distillation is only applied during training

    # Wrap datasets with PocketEmbeddingDatasetWrapper if pocket embeddings are provided
    if args.pocket_embeddings_train_file is not None:
        print_log(f'Wrapping train dataset with pocket embeddings from {args.pocket_embeddings_train_file}')
        train_set = PocketEmbeddingDatasetWrapper(train_set, args.pocket_embeddings_train_file)

        # Wrap validation set with pocket embeddings if provided
        if valid_set is not None and args.pocket_embeddings_val_file is not None:
            print_log(f'Wrapping validation dataset with pocket embeddings from {args.pocket_embeddings_val_file}')
            valid_set = PocketEmbeddingDatasetWrapper(valid_set, args.pocket_embeddings_val_file)

    # Calculate class weights for weighted loss if requested
    class_weights = None
    if args.weighted_loss:
        from collections import Counter
        # Get labels from the dataset (handle both wrapped and unwrapped datasets)
        # Unwrap dataset if it's wrapped
        # Wrappers with 'dataset' attribute: DynamicBatchWrapper, BalancedDynamicBatchWrapper
        # Wrappers with 'base_dataset' attribute: DistillationDatasetWrapper, ResidueDistillationDatasetWrapper, PocketEmbeddingDatasetWrapper
        dataset_for_labels = train_set
        while hasattr(dataset_for_labels, 'dataset') or hasattr(dataset_for_labels, 'base_dataset'):
            if hasattr(dataset_for_labels, 'dataset'):
                dataset_for_labels = dataset_for_labels.dataset
            elif hasattr(dataset_for_labels, 'base_dataset'):
                dataset_for_labels = dataset_for_labels.base_dataset

        # Handle MixDatasetWrapper
        if isinstance(dataset_for_labels, MixDatasetWrapper):
            labels = []
            for dataset in dataset_for_labels.datasets:
                labels.extend([item['label'] for item in dataset.data])
        else:
            labels = [item['label'] for item in dataset_for_labels.data]
        
        num_classes = args.num_classifier_classes
        
        if args.task == 'multilabel_classifier':
            # For multilabel: calculate pos_weight (weight for positive examples relative to negative)
            # pos_weight[i] = num_negatives[i] / num_positives[i] for each class i
            # Convert labels to numpy array to compute per-class statistics
            # Handle various label formats (list of lists, list of arrays, etc.)
            if isinstance(labels[0], (list, np.ndarray, torch.Tensor)):
                # Convert each label to numpy array
                labels_list = []
                for l in labels:
                    if isinstance(l, torch.Tensor):
                        labels_list.append(l.cpu().numpy() if l.is_cuda else l.numpy())
                    elif isinstance(l, np.ndarray):
                        labels_list.append(l)
                    else:
                        labels_list.append(np.array(l))
                labels_array = np.stack(labels_list)
            else:
                # Try direct conversion, if it fails, handle as scalar labels
                try:
                    labels_array = np.array(labels)
                    if labels_array.ndim == 1:
                        # Single label per sample - convert to 2D
                        labels_array = labels_array.reshape(-1, 1)
                except:
                    raise ValueError(f"Unable to parse multilabel labels. Expected 2D array (N, num_classes), got labels of type {type(labels[0])}")
            
            if labels_array.ndim != 2:
                raise ValueError(f"Multilabel labels must be 2D array (N, num_classes), got shape {labels_array.shape}")
            
            if labels_array.shape[1] != num_classes:
                print_log(f"Warning: labels have {labels_array.shape[1]} classes but num_classifier_classes={num_classes}")
            
            class_weights = torch.zeros(num_classes, dtype=torch.float32)
            num_samples = labels_array.shape[0]
            
            for class_idx in range(num_classes):
                if labels_array.shape[1] > class_idx:
                    positives = labels_array[:, class_idx].sum()
                    negatives = num_samples - positives
                    
                    if positives > 0:
                        class_weights[class_idx] = float(negatives) / float(positives)
                    else:
                        # If no positive examples, set weight to 1.0 (no weighting)
                        class_weights[class_idx] = 1.0
                else:
                    class_weights[class_idx] = 1.0
            
            # Log per-class statistics
            print_log(f'Pos weights for multilabel weighted loss: {class_weights.tolist()}')
            for class_idx in range(num_classes):
                if labels_array.shape[1] > class_idx:
                    positives = int(labels_array[:, class_idx].sum())
                    negatives = int(num_samples - positives)
                    print_log(f'Class {class_idx}: {positives} positives, {negatives} negatives, pos_weight={class_weights[class_idx]:.4f}')
        else:
            # For multiclass: calculate inverse frequency weights
            label_counts = Counter(labels)
            # Calculate inverse frequency weights (similar to BalancedDynamicBatchWrapper)
            class_weights = torch.zeros(num_classes, dtype=torch.float32)
            for class_idx in range(num_classes):
                if class_idx in label_counts:
                    class_weights[class_idx] = 1.0 / label_counts[class_idx]
                else:
                    class_weights[class_idx] = 0.0
            
            # Normalize weights
            total_weight = class_weights.sum()
            if total_weight > 0:
                class_weights = class_weights / total_weight * num_classes  # Normalize so average weight is 1
            else:
                class_weights = torch.ones(num_classes, dtype=torch.float32)
            
            print_log(f'Class weights for weighted loss: {class_weights.tolist()}')
            print_log(f'Class distribution: {dict(label_counts)}')
    
    if args.max_n_vertex_per_gpu is not None:
        if args.valid_max_n_vertex_per_gpu is None:
            args.valid_max_n_vertex_per_gpu = args.max_n_vertex_per_gpu
        if args.balanced_sampler:
            if args.task in {'pretrain_torsion', 'pretrain_gaussian', 'masking', 'pretrain_torsion_masking'}:
                train_set = PretrainBalancedDynamicBatchWrapper(train_set, args.max_n_vertex_per_gpu, args.max_n_vertex_per_item, shuffle=args.shuffle)
            else:
                train_set = BalancedDynamicBatchWrapper(train_set, args.max_n_vertex_per_gpu, args.max_n_vertex_per_item, shuffle=args.shuffle)
        else:
            train_set = DynamicBatchWrapper(train_set, args.max_n_vertex_per_gpu, args.max_n_vertex_per_item, shuffle=args.shuffle)
        if valid_set is not None:
            valid_set = DynamicBatchWrapper(valid_set, args.valid_max_n_vertex_per_gpu, args.max_n_vertex_per_item, shuffle=False)
        args.batch_size, args.valid_batch_size = 1, 1
        args.num_workers = 1

    ########## define your model/trainer/trainconfig #########
    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    if args.task in ['binary_classifier', 'multiclass_classifier', 'residue_binary_classifier']:
        # maximize AURPC (or F1 macro if specified)
        metric_min_better = False
    elif args.task == 'multilabel_classifier' and args.multiclass_metric in ['auprc', 'f1_macro', None]:
        # maximize AUPRC or F1 macro for multilabel classification
        metric_min_better = False
    else:
        metric_min_better = True
    config = trainers.TrainConfig(
        args.save_dir, args.lr, args.max_epoch,
        cycle_steps=args.cycle_steps,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        warmup_end_lr=args.warmup_end_lr,
        patience=args.patience,
        grad_clip=args.grad_clip,
        save_topk=args.save_topk,
        weight_decay=args.weight_decay,
        metric_min_better=metric_min_better,
    )
    config.add_parameter(step_per_epoch=step_per_epoch,
                         final_lr=args.final_lr if args.final_lr is not None else args.lr,
                         multiclass_metric=args.multiclass_metric,
                         distillation_alpha=args.distillation_alpha if args.teacher_logits_file else None,
                         distillation_temperature=args.distillation_temperature if args.teacher_logits_file else None)
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

    # Set class weights on model if weighted_loss is enabled
    if args.weighted_loss and class_weights is not None:
        if hasattr(model, 'set_class_weights'):
            model.set_class_weights(class_weights)
        else:
            raise ValueError(f"Model {type(model)} does not support weighted_loss. Only MultiClassClassifierModel and MultiLabelClassifierModel support this option.")
    
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
                              collate_fn=train_set.collate_fn,
                              worker_init_fn=lambda x: np.random.seed(args.seed + x))
    if valid_set is not None:
        valid_loader = DataLoader(valid_set, batch_size=args.valid_batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=valid_set.collate_fn,
                                  shuffle=False)
    else:
        valid_loader = None
    trainer = create_trainer(model, train_loader, valid_loader, config, 
                             resume_state=torch.load(args.pretrain_state) if args.pretrain_state else None)
    if args.local_rank <= 0: # only log on the main process
        print_log(f"Saving model checkpoints to: {config.save_dir}")
        os.makedirs(config.save_dir, exist_ok=True)
        with open(os.path.join(config.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        if args.use_wandb:
            wandb_args = vars(args)
            wandb_args['save_dir'] = config.save_dir
            wandb.init(
                entity="ada-f",
                dir=config.save_dir,
                settings=wandb.Settings(start_method="fork"),
                project=f"InteractNN-{args.task}",
                name=args.run_name,
                config=wandb_args,
            )
    trainer.train(args.gpus, args.local_rank, use_wandb=args.use_wandb, use_raytune=args.use_raytune)
    
    return trainer.topk_ckpt_map


if __name__ == '__main__':
    args = parse()
    main(args)
