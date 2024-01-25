#!/usr/bin/python
# -*- coding:utf-8 -*-
from .pretrain_model import DenoisePretrainModel
from .affinity_predictor import AffinityPredictor
from .graph_classifier import GraphClassifier
from .graph_pair_classifier import GraphPairClassifier
from .graph_multi_binary_classifier import GraphMultiBinaryClassifier

def create_model(args):
    model_type = args.model_type
    if 'pretrain' in args.task.lower():
        return DenoisePretrainModel(
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
        )
    else:
        add_params = {}
        if args.task == 'LEP':
            Model = GraphPairClassifier
            add_params = {
                'num_class': 2
            }
        elif args.task == 'PLA' or args.task == 'PPA' or args.task == 'AffMix' or args.task == 'PDBBind' or args.task == 'NL' or args.task == 'PLA_frag':
            Model = AffinityPredictor
        elif args.task == 'EC':
            Model = GraphMultiBinaryClassifier
            add_params = {
                'n_task': 538
            }
        else:
            raise NotImplementedError(f'Model for task {args.task} not implemented')
            
        if args.pretrain_ckpt:
            model = Model.load_from_pretrained(args.pretrain_ckpt, **add_params)
            assert model.model_type == model_type
            return model
        else:
            return Model(
                model_type=model_type,
                hidden_size=args.hidden_size,
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
                **add_params
            )
