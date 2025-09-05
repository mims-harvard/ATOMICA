from .pretrain_model import DenoisePretrainModel, DenoisePretrainModelWithBlockEmbedding
from .affinity_predictor import AffinityPredictor
from .classifier_model import ClassifierModel, MultiClassClassifierModel, RegressionPredictor
from .masking_model import MaskedNodeModel
from .prot_interface_model import ProteinInterfaceModel
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
                print(f"Loading pretrain model from checkpoint {args.pretrain_ckpt}")
                model: DenoisePretrainModel = torch.load(args.pretrain_ckpt, map_location='cpu')
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
                print(f"Loading pretrain model from checkpoint {args.pretrain_ckpt}")
                model: DenoisePretrainModelWithBlockEmbedding = torch.load(args.pretrain_ckpt, map_location='cpu')
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
            print(f"Loading pretrain model from checkpoint {args.pretrain_ckpt}")
            add_params["partial_finetune"] = args.partial_finetune
            model = AffinityPredictor.load_from_pretrained(args.pretrain_ckpt, **add_params)
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
            if args.pretrain_ckpt:
                print(f"Loading pretrain model from checkpoint {args.pretrain_ckpt}")
                model = Model.load_from_pretrained(args.pretrain_ckpt, **add_params)
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
