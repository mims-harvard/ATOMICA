#!/bin/bash
#SBATCH -J atomica_interface
#SBATCH -o /path/to/ATOMICA/out/%x_%j.out
#SBATCH -e /path/to/ATOMICA/out/%x_%j.err
#SBATCH -c 32
#SBATCH -t 0-24:00
#SBATCH -p partition_name
#SBATCH --gres=gpu:4
#SBATCH --mem=512G


~/.conda/envs/atomicaenv/bin/torchrun --nnodes=1 --nproc_per_node=4 --standalone train.py \
    --train_set datasets/QBioLiP_train.pkl \
    --valid_set datasets/QBioLiP_valid.pkl \
    --task prot_interface \
    --gpu 0 1 2 3 \
    --num_workers 32 \
    --lr 5e-4 \
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
    --pretrain_weights path/to/pretrained/ATOMICA/weights \
    --pretrain_config path/to/pretrained/ATOMICA/config \
    --run_name ATOMICA-Interface