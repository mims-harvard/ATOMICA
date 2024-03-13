#!/usr/bin/python
# -*- coding:utf-8 -*-
from .pretrain_model import DenoisePretrainModel
from .affinity_predictor import AffinityPredictor
from .ddG_predictor import DDGPredictor
import torch

def create_model(args):
    if 'pretrain' in args.task.lower():
        if args.pretrain_ckpt:
            model: DenoisePretrainModel = torch.load(args.pretrain_ckpt, map_location='cpu')
        else:
            model = DenoisePretrainModel(
                hidden_size=args.hidden_size,
                edge_size=args.edge_size,
                k_neighbors=args.k_neighbors,
                n_layers=args.n_layers,
                atom_noise=args.atom_noise != 0,
                translation_noise=args.translation_noise != 0,
                rotation_noise=args.rotation_noise != 0,
                torsion_noise=args.torsion_noise != 0,
                global_message_passing=args.global_message_passing,
                fragmentation_method=args.fragmentation_method,
                atom_weight=args.atom_weight,
                translation_weight=args.tr_weight,
                rotation_weight=args.rot_weight,
                torsion_weight=args.tor_weight,
            )
        return model
    else:
        add_params = {}
        if args.task in {'PLA', 'PPA', 'AffMix', 'PDBBind', 'NL', 'PLA_frag', 'PN'}:
            Model = AffinityPredictor
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
            return model
        else:
            return Model(
                hidden_size=args.hidden_size,
                edge_size=args.edge_size,
                k_neighbors=args.k_neighbors,
                n_layers=args.n_layers,
                global_message_passing=args.global_message_passing,
                fragmentation_method=args.fragmentation_method,
                **add_params
            )
