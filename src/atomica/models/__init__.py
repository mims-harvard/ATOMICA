from .pretrain_model import DenoisePretrainModel, DenoisePretrainModelWithBlockEmbedding
from .affinity_predictor import AffinityPredictor
from .classifier_model import ClassifierModel, MultiClassClassifierModel, RegressionPredictor, ResidueClassifierModel, MultiLabelClassifierModel
from .masking_model import MaskedNodeModel
from .prot_interface_model import ProteinInterfaceModel
from ..utils import pickled_checkpoint_error
import torch

def create_model(args):
    if 'pretrain' in args.task.lower():
        params = {
            "atom_hidden_size": args.atom_hidden_size,
            "block_hidden_size": args.block_hidden_size,
            "edge_size": args.edge_size,
            "k_neighbors": args.k_neighbors,
            "n_layers": args.n_layers,
            "atom_noise": args.atom_noise != 0,
            "translation_noise": args.translation_noise != 0,
            "rotation_noise": args.rotation_noise != 0,
            "torsion_noise": args.torsion_noise != 0,
            "bottom_global_message_passing": args.bottom_global_message_passing,
            "global_message_passing": args.global_message_passing,
            "fragmentation_method": args.fragmentation_method,
            "atom_weight": args.atom_weight,
            "translation_weight": args.tr_weight,
            "rotation_weight": args.rot_weight,
            "torsion_weight": args.tor_weight,
            "dropout": args.dropout,
            "num_masked_block_classes": args.num_nodes,
            "mask_weight": args.mask_weight,
            "modality_embedding": args.modality_embedding,
        }
        if args.block_embedding_size is None and args.block_embedding0_size is None and args.block_embedding1_size is None:
            if args.pretrain_ckpt:
                raise pickled_checkpoint_error(
                    args.pretrain_ckpt, "--pretrain_config", "--pretrain_weights")
            elif args.pretrain_config and args.pretrain_weights:
                print(f"Loading pretrain model from config {args.pretrain_config} and weights {args.pretrain_weights}")
                model = DenoisePretrainModel.load_from_config_and_weights(args.pretrain_config, args.pretrain_weights)
            else:
                model = DenoisePretrainModel(**params)
        else:
            params.update({
                'num_projector_layers': args.num_projector_layers,
                'projector_dropout': args.projector_dropout,
                'projector_hidden_size': args.projector_hidden_size,
                "block_embedding_size": args.block_embedding_size,
                "block_embedding0_size": args.block_embedding0_size,
                "block_embedding1_size": args.block_embedding1_size,
            })
            if args.pretrain_ckpt:
                raise pickled_checkpoint_error(
                    args.pretrain_ckpt, "--pretrain_config", "--pretrain_weights")
            elif args.pretrain_config and args.pretrain_weights:
                print(f"Loading pretrain model from config {args.pretrain_config} and weights {args.pretrain_weights}")
                model = DenoisePretrainModelWithBlockEmbedding.load_from_config_and_weights(args.pretrain_config, args.pretrain_weights)
            else:
                model = DenoisePretrainModelWithBlockEmbedding(**params)
        return model
    elif args.task == 'PDBBind':
        add_params = {
            'num_affinity_pred_layers': args.num_pred_layers,
            'affinity_pred_dropout': args.pred_dropout,
            'affinity_pred_hidden_size': args.pred_hidden_size,
            'num_projector_layers': args.num_projector_layers,
            'projector_dropout': args.projector_dropout,
            'projector_hidden_size': args.projector_hidden_size,
            'bottom_global_message_passing': args.bottom_global_message_passing,
            'global_message_passing': args.global_message_passing,
            'k_neighbors': args.k_neighbors,
            'dropout': args.dropout,
            'block_embedding_size': args.block_embedding_size,
            'block_embedding0_size': args.block_embedding0_size,
            'block_embedding1_size': args.block_embedding1_size,
        }
        if args.pred_nonlinearity == 'relu':
            add_params["nonlinearity"] = torch.nn.ReLU()
        elif args.pred_nonlinearity == 'gelu':
            add_params["nonlinearity"] = torch.nn.GELU()
        elif args.pred_nonlinearity == 'elu':
            add_params["nonlinearity"] = torch.nn.ELU()
        else:
            raise NotImplementedError(f"Nonlinearity {args.pred_nonlinearity} not implemented")
        if args.pretrain_ckpt:
            raise pickled_checkpoint_error(
                args.pretrain_ckpt, "--pretrain_config", "--pretrain_weights")
        elif args.pretrain_config and args.pretrain_weights:
            print(f"Loading pretrain model from config {args.pretrain_config} and weights {args.pretrain_weights}")
            model = AffinityPredictor.load_from_config_and_weights(args.pretrain_config, args.pretrain_weights, **add_params)
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
        if args.task == 'regression':
            Model = RegressionPredictor
        elif args.task == 'binary_classifier' or args.task == 'RNAScore_binary':
            Model = ClassifierModel
            add_params.update({
                'num_pred_layers': args.num_pred_layers,
                'pred_dropout': args.pred_dropout,
                'pred_hidden_size': args.pred_hidden_size,
            })
            if args.pred_nonlinearity == 'relu':
                add_params["nonlinearity"] = torch.nn.ReLU()
            elif args.pred_nonlinearity == 'gelu':
                add_params["nonlinearity"] = torch.nn.GELU()
            elif args.pred_nonlinearity == 'elu':
                add_params["nonlinearity"] = torch.nn.ELU()
            else:
                raise NotImplementedError(f"Nonlinearity {args.pred_nonlinearity} not implemented")
        elif args.task == 'multiclass_classifier':
            Model = MultiClassClassifierModel
            add_params["num_classes"] = args.num_classifier_classes
            # Add focal loss parameters
            if args.use_focal_loss:
                add_params["loss_type"] = "focal"
                add_params["focal_gamma"] = args.focal_gamma
                if args.focal_alpha is not None:
                    add_params["focal_alpha"] = args.focal_alpha
            # Add pocket embedding size if provided
            if hasattr(args, 'pocket_embedding_size') and args.pocket_embedding_size is not None:
                add_params["pocket_embedding_size"] = args.pocket_embedding_size
        elif args.task == 'multilabel_classifier':
            Model = MultiLabelClassifierModel
            add_params["num_classes"] = args.num_classifier_classes
            # Add focal loss parameters
            if args.use_focal_loss:
                add_params["loss_type"] = "focal"
                add_params["focal_gamma"] = args.focal_gamma
                if args.focal_alpha is not None:
                    add_params["focal_alpha"] = args.focal_alpha
            # Add pocket embedding size if provided
            if hasattr(args, 'pocket_embedding_size') and args.pocket_embedding_size is not None:
                add_params["pocket_embedding_size"] = args.pocket_embedding_size
        elif args.task == 'residue_binary_classifier':
            Model = ResidueClassifierModel
            add_params.update({
                'num_pred_layers': args.num_pred_layers,
                'pred_dropout': args.pred_dropout,
                'pred_hidden_size': args.pred_hidden_size,
            })
            if args.pred_nonlinearity == 'relu':
                add_params["nonlinearity"] = torch.nn.ReLU()
            elif args.pred_nonlinearity == 'gelu':
                add_params["nonlinearity"] = torch.nn.GELU()
            elif args.pred_nonlinearity == 'elu':
                add_params["nonlinearity"] = torch.nn.ELU()
            else:
                raise NotImplementedError(f"Nonlinearity {args.pred_nonlinearity} not implemented")
        elif args.task == 'masking':
            Model = MaskedNodeModel
            add_params['num_masked_block_classes'] = args.num_nodes
        elif args.task == 'prot_interface':
            Model = ProteinInterfaceModel
        else:
            raise NotImplementedError(f'Model for task {args.task} not implemented')
        
        if args.pretrain_ckpt or (args.pretrain_config and args.pretrain_weights):
            add_params.update({
                'partial_finetune': args.partial_finetune,
                'bottom_global_message_passing': args.bottom_global_message_passing,
                'global_message_passing': args.global_message_passing,
                'k_neighbors': args.k_neighbors,
                'dropout': args.dropout,
            })
            # masking-task switches; every default reproduces the released pretrained model
            if getattr(args, 'top_max_edge_length', None) is not None:
                add_params['top_max_edge_length'] = args.top_max_edge_length
            if getattr(args, 'top_long_range_edge_length', None) is not None:
                add_params['top_long_range_edge_length'] = args.top_long_range_edge_length
            if getattr(args, 'na_loss_weight', 1.0) != 1.0:
                add_params['na_loss_weight'] = args.na_loss_weight
            for flag in ('attn_pad_mask', 'masked_affine', 'bottom_repr_scale', 'top_pair_geom'):
                if getattr(args, flag, False):
                    add_params[flag] = True
            if args.pretrain_ckpt:
                raise pickled_checkpoint_error(
                    args.pretrain_ckpt, "--pretrain_config", "--pretrain_weights")
            elif args.pretrain_config and args.pretrain_weights:
                print(f"Loading pretrain model from config {args.pretrain_config} and weights {args.pretrain_weights}")
                model = Model.load_from_config_and_weights(args.pretrain_config, args.pretrain_weights, **add_params)
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
