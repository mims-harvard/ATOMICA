#!/bin/bash
#SBATCH -J pretrain_atomica
#SBATCH -o /path/to/ATOMICA/out/%x_%j.out
#SBATCH -e /path/to/ATOMICA/out/%x_%j.err
#SBATCH -c 32
#SBATCH -t 0-72:00
#SBATCH -p partition_name
#SBATCH --gres=gpu:4
#SBATCH --mem=512G

~/.conda/envs/atomicaenv/bin/torchrun --nnodes=1 --nproc_per_node=4 --standalone train.py \
    --train_set datasets/QBioLiP_train.pkl \
    --valid_set datasets/QBioLiP_valid.pkl \
    --train_set2 datasets/CSD_PS_300_train.pkl \
    --valid_set2 datasets/CSD_PS_300_valid.pkl \
    --task pretrain_torsion_masking \
    --gpu 0 1 2 3 \
    --num_workers 32 \
    --lr 1e-4 \
    --final_lr 1e-6 \
    --max_epoch 200 \
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
    --cycle_steps 400000 \
    --mask_proportion 0.1 \
    --torsion_noise 0.5 \
    --translation_noise 1 \
    --rotation_noise 0.25 \
    --max_rotation 0.5 \
    --rot_weight 0.1 \
    --save_dir model_checkpoints \
    --run_name ATOMICA