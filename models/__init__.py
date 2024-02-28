#!/usr/bin/python
# -*- coding:utf-8 -*-
from .pretrain_model import DenoisePretrainModel
from .affinity_predictor import AffinityPredictor
from .graph_classifier import GraphClassifier
from .graph_pair_classifier import GraphPairClassifier
from .graph_multi_binary_classifier import GraphMultiBinaryClassifier
from .ddG_predictor import DDGPredictor
import torch

def create_model(args):
    model_type = args.model_type
    if 'pretrain' in args.task.lower():
        if args.pretrain_ckpt:
            model: DenoisePretrainModel = torch.load(args.pretrain_ckpt, map_location='cpu')
        else:
            model = DenoisePretrainModel(
                model_type=model_type,
                hidden_size=args.hidden_size,
                n_channel=args.n_channel,
                n_rbf=args.n_rbf,
                cutoff=args.cutoff,
                edge_size=args.edge_size,
                radial_size=args.radial_size,
                k_neighbors=args.k_neighbors,
                n_layers=args.n_layers,
                n_head=args.n_head,
                atom_level=args.atom_level,
                hierarchical=args.hierarchical,
                no_block_embedding=args.no_block_embedding,
                denoising=True,
                atom_noise=args.atom_noise,
                translation_noise=args.translation_noise,
                rotation_noise=args.rotation_noise,
                global_message_passing=args.global_message_passing,
                fragmentation_method=args.fragmentation_method,
                rot_sigma=args.rot_sigma,
            )
        return model
    else:
        add_params = {}
        if args.task == 'LEP':
            Model = GraphPairClassifier
            add_params = {
                'num_class': 2
            }
        elif args.task in {'PLA', 'PPA', 'AffMix', 'PDBBind', 'NL', 'PLA_frag', 'PN'}:
            Model = AffinityPredictor
        elif args.task == 'EC':
            Model = GraphMultiBinaryClassifier
            add_params = {
                'n_task': 538
            }
        elif args.task == 'DDG':
            Model = DDGPredictor
        else:
            raise NotImplementedError(f'Model for task {args.task} not implemented')
            
        if args.pretrain_ckpt:
            if Model == AffinityPredictor or Model == DDGPredictor:
                add_params = {
                    'partial_finetune': args.partial_finetune,
                    'global_message_passing': args.global_message_passing,
                }
            model = Model.load_from_pretrained(args.pretrain_ckpt, **add_params)
            print(f"Model size: {sum(p.numel() for p in model.parameters())}")
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_trainable_params}")
            assert model.model_type == model_type
            return model
        else:
            return Model(
                model_type=model_type,
                hidden_size=args.hidden_size,
                edge_size=args.edge_size,
                n_channel=args.n_channel,
                n_rbf=args.n_rbf,
                cutoff=args.cutoff,
                radial_size=args.radial_size,
                k_neighbors=args.k_neighbors,
                n_layers=args.n_layers,
                n_head=args.n_head,
                atom_level=args.atom_level,
                hierarchical=args.hierarchical,
                no_block_embedding=args.no_block_embedding,
                global_message_passing=args.global_message_passing,
                fragmentation_method=args.fragmentation_method,
                **add_params
            )
