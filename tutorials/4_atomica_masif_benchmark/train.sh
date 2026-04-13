#!/bin/bash
# Train ATOMICA on the MASIF-Ligand benchmark (8A pocket cutoff, 7-class).
#
# Trains a single seed. To reproduce the ensemble, re-run with
# SEED={0,1,2,3,4}.
#
# Expects:
#   - ATOMICA installed in the active environment
#   - Pretrain weights at <repo_root>/checkpoints/pretrain/pretrain_model_weights.pt
#   - Train/val parquets at <tutorial_dir>/data/masif_{train,val}.parquet
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"

SEED=${SEED:-0}
DATA_DIR=${DATA_DIR:-"${TUTORIAL_DIR}/data"}
SAVE_DIR=${SAVE_DIR:-"${TUTORIAL_DIR}/train_output"}
PRETRAIN_WEIGHTS=${PRETRAIN_WEIGHTS:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_weights.pt"}
PRETRAIN_CONFIG=${PRETRAIN_CONFIG:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_config.json"}

python -m atomica.train \
    --gpu 0 \
    --task multiclass_classifier \
    --num_classifier_classes 7 \
    --weighted_loss \
    --multiclass_metric f1_macro \
    --lr 3e-5 \
    --final_lr 3e-5 \
    --dropout 0.1 \
    --max_epoch 300 \
    --max_n_vertex_per_gpu 1000 \
    --shuffle \
    --train_set ${DATA_DIR}/masif_train.parquet \
    --valid_set ${DATA_DIR}/masif_val.parquet \
    --pretrain_weights ${PRETRAIN_WEIGHTS} \
    --pretrain_config ${PRETRAIN_CONFIG} \
    --save_dir ${SAVE_DIR} \
    --seed ${SEED} \
    --run_name masif_seed${SEED}
