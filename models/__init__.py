#!/usr/bin/python
# -*- coding:utf-8 -*-
from .pretrain_model import DenoisePretrainModel
from .affinity_predictor import AffinityPredictor, AffinityPredictorNoisyNodes, BlockAffinityPredictor
from .ddG_predictor import DDGPredictor, GLOFPredictor
from .classifier_model import ClassifierModel, MultiClassClassifierModel, RegressionPredictor
from .prediction_model import PredictionModel
from .masking_model import MaskedNodeModel
from .binary_predictors import BinaryPredictor, BinaryPredictorMSP, BinaryPredictorMSP2
import torch

def create_model(args):
    if 'pretrain' in args.task.lower():
        if args.pretrain_ckpt:
            model: DenoisePretrainModel = torch.load(args.pretrain_ckpt, map_location='cpu')
        else:
            model = DenoisePretrainModel(
                atom_hidden_size=args.atom_hidden_size,
                block_hidden_size=args.block_hidden_size,
                edge_size=args.edge_size,
                k_neighbors=args.k_neighbors,
                n_layers=args.n_layers,
                atom_noise=args.atom_noise != 0,
                translation_noise=args.translation_noise != 0,
                rotation_noise=args.rotation_noise != 0,
                torsion_noise=args.torsion_noise != 0,
                bottom_global_message_passing=args.bottom_global_message_passing,
                global_message_passing=args.global_message_passing,
                fragmentation_method=args.fragmentation_method,
                atom_weight=args.atom_weight,
                translation_weight=args.tr_weight,
                rotation_weight=args.rot_weight,
                torsion_weight=args.tor_weight,
                dropout=args.dropout,
                num_masked_block_classes=args.num_nodes,
                mask_weight=args.mask_weight,
            )
        return model
    elif args.task == "PLA_noisy_nodes":
        if args.pretrain_ckpt:
            add_params = {
                'partial_finetune': args.partial_finetune,
                'bottom_global_message_passing': args.bottom_global_message_passing,
                'global_message_passing': args.global_message_passing,
                'k_neighbors': args.k_neighbors,
                'dropout': args.dropout,
            }
            model = AffinityPredictorNoisyNodes.load_from_pretrained(args.pretrain_ckpt, noisy_nodes_weight=args.noisy_nodes_weight, **add_params)
        else:
            model = AffinityPredictorNoisyNodes(
                noisy_nodes_weight=args.noisy_nodes_weight,
                atom_hidden_size=args.atom_hidden_size,
                block_hidden_size=args.block_hidden_size,
                edge_size=args.edge_size,
                k_neighbors=args.k_neighbors,
                n_layers=args.n_layers,
                atom_noise=args.atom_noise != 0,
                translation_noise=args.translation_noise != 0,
                rotation_noise=args.rotation_noise != 0,
                torsion_noise=args.torsion_noise != 0,
                bottom_global_message_passing=args.bottom_global_message_passing,
                global_message_passing=args.global_message_passing,
                fragmentation_method=args.fragmentation_method,
                atom_weight=args.atom_weight,
                translation_weight=args.tr_weight,
                rotation_weight=args.rot_weight,
                torsion_weight=args.tor_weight,
                dropout=args.dropout,
            )
        return model
    elif args.task in {'PLA', 'AffMix', 'PDBBind', 'NL', 'PLA_frag', 'PN', 'PPA-atom'}:
        add_params = {
            "num_affinity_pred_layers": args.num_affinity_pred_layers,
            "affinity_pred_dropout": args.affinity_pred_dropout,
            'affinity_pred_hidden_size': args.affinity_pred_hidden_size,
            'bottom_global_message_passing': args.bottom_global_message_passing,
            'global_message_passing': args.global_message_passing,
            'k_neighbors': args.k_neighbors,
            'dropout': args.dropout,
        }
        if args.affinity_pred_nonlinearity == 'relu':
            add_params["nonlinearity"] = torch.nn.ReLU()
        elif args.affinity_pred_nonlinearity == 'gelu':
            add_params["nonlinearity"] = torch.nn.GELU()
        elif args.affinity_pred_nonlinearity == 'elu':
            add_params["nonlinearity"] = torch.nn.ELU()
        else:
            raise NotImplementedError(f"Nonlinearity {args.affinity_pred_nonlinearity} not implemented")
        if args.pretrain_ckpt:
            add_params["partial_finetune"] = args.partial_finetune
            model = AffinityPredictor.load_from_pretrained(args.pretrain_ckpt, **add_params)
        else:
            model = AffinityPredictor(
                atom_hidden_size=args.atom_hidden_size,
                block_hidden_size=args.block_hidden_size,
                edge_size=args.edge_size,
                n_layers=args.n_layers,
                fragmentation_method=args.fragmentation_method,
                **add_params
            )
        return model
        
    else:
        add_params = {}
        if args.task == 'PPA':
            Model = BlockAffinityPredictor
        elif args.task == 'LEP':
            Model = BinaryPredictor
        elif args.task == 'MSP' or args.task == 'GLOF':
            Model = BinaryPredictorMSP
        elif args.task == 'MSP2':
            Model = BinaryPredictorMSP2
        elif args.task == 'regression':
            Model = RegressionPredictor
        elif args.task == 'DDG':
            Model = DDGPredictor
        elif args.task == 'binary_classifier':
            Model = ClassifierModel
        elif args.task == 'multiclass_classifier':
            Model = MultiClassClassifierModel
            add_params["num_classes"] = args.num_classifier_classes
        elif args.task == 'masking':
            Model = MaskedNodeModel
            add_params["num_nodes"] = args.num_nodes
            add_params['global_message_passing'] = args.global_message_passing
            add_params['bottom_global_message_passing'] = args.bottom_global_message_passing
        else:
            raise NotImplementedError(f'Model for task {args.task} not implemented')
        
        if args.pretrain_ckpt:
            if Model in [AffinityPredictor, BinaryPredictor, DDGPredictor, GLOFPredictor, ClassifierModel, 
                         MultiClassClassifierModel, RegressionPredictor, BlockAffinityPredictor, BinaryPredictorMSP, BinaryPredictorMSP2]:
                add_params.update({
                    'partial_finetune': args.partial_finetune,
                    'bottom_global_message_passing': args.bottom_global_message_passing,
                    'global_message_passing': args.global_message_passing,
                    'k_neighbors': args.k_neighbors,
                    'dropout': args.dropout,
                })
            model = Model.load_from_pretrained(args.pretrain_ckpt, **add_params)
            print(f"Model size: {sum(p.numel() for p in model.parameters())}")
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_trainable_params}")
            return model
        else:
            return Model(
                atom_hidden_size=args.atom_hidden_size,
                block_hidden_size=args.block_hidden_size,
                edge_size=args.edge_size,
                k_neighbors=args.k_neighbors,
                n_layers=args.n_layers,
                bottom_global_message_passing=args.bottom_global_message_passing,
                global_message_passing=args.global_message_passing,
                fragmentation_method=args.fragmentation_method,
                dropout=args.dropout,
                **add_params
            )
