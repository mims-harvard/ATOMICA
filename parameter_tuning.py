import numpy as np
import argparse

import ray
from ray import tune
from ray import train as ray_train
from train import main
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.runtime_env import RuntimeEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address")
    parser.add_argument("--num_cpus_per_trial", type=int, default=8)
    parser.add_argument("--num_gpus_per_trial", type=int, default=1)
    parser.add_argument("--max_concurrent_trials", type=int, default=16)
    args = parser.parse_args()
    return args


def tune_with_setup(args):
    ray.init(address=args.address, 
             runtime_env={"working_dir": "/n/holylabs/LABS/mzitnik_lab/Users/afang/GET",
                          "excludes":['case_studies/*', 'datasets/*', 'testing_scripts/*']})
    print(ray.cluster_resources())
    main_tune_with_resources = tune.with_resources(
        main_tune, 
        {"cpu": args.num_cpus_per_trial, "gpu": args.num_gpus_per_trial})
    tuner = tune.Tuner(
        main_tune_with_resources,
        tune_config=tune.TuneConfig(
            num_samples=50,
            max_concurrent_trials=args.max_concurrent_trials, # update this to more than number of GPUs
            search_alg = OptunaSearch(
                metric="val_mask_acc",
                mode="max",
                seed=42,
            ),
            scheduler=ASHAScheduler(
                time_attr="epoch",
                metric="val_mask_acc",
                mode="max",
                max_t=20,
                grace_period=5,
                reduction_factor=2,
                brackets=2,
            )
        ),
        run_config=ray_train.RunConfig(
            stop={"time_total_s": 60*60*24}, 
            storage_path="/n/holyscratch01/mzitnik_lab/afang/GET/pretrain/models/InteractNN-tune"),
        param_space={
            "translation_noise": tune.choice([0.5, 1, 1.5]),
            "torsion_noise": tune.choice([0.25, 0.5, 1]),
            "rotation_noise": tune.choice([0.25, 0.5, 1]),
            "max_rotation": tune.choice([0.25, 0.5, 1]),
        },
    )
    results = tuner.fit()
    print(results)
    print("Best config: ", results.get_best_result().config)
    return results.get_best_result().config

def main_tune(config):
    # Set up argparse arguments manually from the config (which is passed from the tuner)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--translation_noise", type=float, default=config["translation_noise"])
    parser.add_argument("--rotation_noise", type=float, default=config["rotation_noise"])
    parser.add_argument("--torsion_noise", type=float, default=config["torsion_noise"])
    parser.add_argument("--tr_weight", type=float, default=1)
    parser.add_argument("--rot_weight", type=float, default=0.1)
    parser.add_argument("--tor_weight", type=float, default=1)
    parser.add_argument("--max_rotation", type=float, default=config["max_rotation"])
    
    # Add other required arguments for the model training process
    parser.add_argument("--gpus", type=int, default=[0])
    parser.add_argument("--task", type=str, default="pretrain_torsion_masking")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    parser.add_argument("--max_epoch", type=int, default=10000)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--atom_hidden_size", type=int, default=32)
    parser.add_argument("--block_hidden_size", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--edge_size", type=int, default=32)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_set", type=str, default="/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/CSD_PS_300_train_subsample10pc.pkl")
    parser.add_argument("--valid_set", type=str, default="/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/CSD_PS_300_valid.pkl")
    parser.add_argument("--train_set2", type=str, default="/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/QBioLiP_train_subsample10pc.pkl")
    parser.add_argument("--valid_set2", type=str, default="/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/QBioLiP_valid.pkl")
    parser.add_argument("--save_dir", type=str, default="/n/holyscratch01/mzitnik_lab/afang/GET/pretrain/models/InteractNN-tune")
    parser.add_argument("--run_name", type=str, default=f"noise_trans_"+"_".join([f"{k}_{v}" for k, v in config.items()]))
    parser.add_argument("--max_n_vertex_per_gpu", type=int, default=1024)
    parser.add_argument("--max_n_vertex_per_item", type=int, default=256)
    parser.add_argument("--mask_proportion", type=float, default=0.1)
    parser.add_argument("--fragmentation_method", type=str, default="PS_300")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--use_raytune", type=bool, default=True)
    parser.add_argument("--global_message_passing", type=bool, default=True)
    parser.add_argument("--bottom_global_message_passing", type=bool, default=False)


    parser.add_argument('--num_classifier_classes', type=int, default=None, help='number of classes for task=multiclass_classifier')
    parser.add_argument('--train_set3', type=str, default=None, help='path to the third train set')
    parser.add_argument('--valid_set3', type=str, default=None, help='path to the third valid set')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs where validation loss is not used for early stopping')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-5, help='linear learning rate warmup start lr')
    parser.add_argument('--warmup_end_lr', type=float, default=1e-3, help='linear learning rate warmup end lr')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients with too big norm')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=None, help='batch size of validation, default set to the same as training batch size')
    parser.add_argument('--valid_max_n_vertex_per_gpu', type=int, default=None, help='form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--patience', type=int, default=-1, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=-1, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--cycle_steps', type=int, default=100000, help='number of steps per cycle in lr_scheduler.CosineAnnealingWarmRestarts')
    parser.add_argument('--atom_noise', type=float, default=0, help='apply noise to atom coordinates')
    parser.add_argument('--atom_weight', type=float, default=1.0, help='Weight of atom loss')
    parser.add_argument('--mask_weight', type=float, default=1.0, help='block masking rate')
    parser.add_argument('--noisy_nodes_weight', type=float, default=0, help='coefficient for denoising loss during finetuning')
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='path of the pretrained ckpt to load')
    parser.add_argument('--pretrain_state', type=str, default=None, help='path of the pretrained training state to load for resuming training')
    parser.add_argument('--partial_finetune', action="store_true", default=False, help='only finetune energy head')
    parser.add_argument('--affinity_pred_dropout', type=float, default=0.0, help='dropout rate for affinity prediction')
    parser.add_argument('--affinity_pred_nonlinearity', type=str, default='relu', choices=['relu', 'gelu', 'elu'], help='nonlinearity for affinity prediction')
    parser.add_argument('--num_affinity_pred_layers', type=int, default=2, help='number of layers for affinity prediction')
    parser.add_argument('--affinity_pred_hidden_size', type=int, default=32, help='hidden size for affinity prediction')

    args = parser.parse_args([])  # Create a namespace from the config
    main(args)


if __name__ == "__main__":
    args = parse_args()
    tune_with_setup(args)