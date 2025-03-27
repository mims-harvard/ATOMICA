#!/bin/bash
#SBATCH -J atomica_ligand
#SBATCH -o /path/to/ATOMICA/out/%x_%j.out
#SBATCH -e /path/to/ATOMICA/out/%x_%j.err
#SBATCH -c 8
#SBATCH -t 0-24:00
#SBATCH -p partition_name
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

~/.conda/envs/atomicaenv/bin/torchrun train.py \
    --train_set datasets/binder_train.pkl \
    --valid_set datasets/binder_valid.pkl \
    --task binary_classifier \
    --gpu 0 \
    --num_workers 8 \
    --lr 1e-4 \
    --max_epoch 50 \
    --shuffle \
    --atom_hidden_size 32 \
    --block_hidden_size 32 \
    --n_layers 4 \
    --edge_size 32 \
    --k_neighbors 8 \
    --max_n_vertex_per_gpu 1000 \
    --max_n_vertex_per_item 256 \
    --fragmentation_method PS_300 \
    --global_message_passing \
    --save_dir model_checkpoints \
    --pretrain_ckpt path/to/pretrained/ATOMICA/checkpoint \
    --run_name ATOMICA-Ligand