#!/bin/bash
# Train ATOMICA on the RNA_Ligand multiclass pocket classification task
# (3 classes: PAR / LLL / 8UZ).
#
# Trains a single seed. Re-run with different SEED values to reproduce
# the 5-seed ensemble.
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"

SEED=${SEED:-0}
DATA_DIR=${DATA_DIR:-"${TUTORIAL_DIR}/data"}
SAVE_DIR=${SAVE_DIR:-"${TUTORIAL_DIR}/train_output/rna_ligand"}
PRETRAIN_WEIGHTS=${PRETRAIN_WEIGHTS:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_weights.pt"}
PRETRAIN_CONFIG=${PRETRAIN_CONFIG:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_config.json"}

python -m atomica.train \
    --gpu 0 \
    --task multiclass_classifier \
    --num_classifier_classes 3 \
    --weighted_loss \
    --multiclass_metric f1_macro \
    --lr 1e-5 \
    --final_lr 1e-5 \
    --weight_decay 0.1 \
    --max_epoch 400 \
    --max_n_vertex_per_gpu 256 \
    --grad_clip 1.0 \
    --shuffle \
    --random_block_sampling \
    --block_sampling_p_keep 0.9 \
    --block_sampling_p_none 0.2 \
    --train_set ${DATA_DIR}/rna_ligand_train.parquet \
    --valid_set ${DATA_DIR}/rna_ligand_val.parquet \
    --pretrain_weights ${PRETRAIN_WEIGHTS} \
    --pretrain_config ${PRETRAIN_CONFIG} \
    --save_dir ${SAVE_DIR} \
    --seed ${SEED} \
    --run_name rna_ligand_seed${SEED}
