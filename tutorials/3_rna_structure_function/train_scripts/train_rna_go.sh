#!/bin/bash
# Train ATOMICA on the RNAGo multilabel classification task (5 classes).
#
# Trains a single seed. Re-run with different SEED values to reproduce
# the 5-seed ensemble.
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"

SEED=${SEED:-0}
DATA_DIR=${DATA_DIR:-"${TUTORIAL_DIR}/data"}
SAVE_DIR=${SAVE_DIR:-"${TUTORIAL_DIR}/train_output/rna_go"}
PRETRAIN_WEIGHTS=${PRETRAIN_WEIGHTS:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_weights.pt"}
PRETRAIN_CONFIG=${PRETRAIN_CONFIG:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_config.json"}

python -m atomica.train \
    --gpu 0 \
    --task multilabel_classifier \
    --num_classifier_classes 5 \
    --multiclass_metric f1_macro \
    --use_focal_loss \
    --focal_gamma 2.0 \
    --lr 4e-5 \
    --final_lr 4e-5 \
    --max_epoch 200 \
    --max_n_vertex_per_gpu 256 \
    --grad_clip 1.0 \
    --shuffle \
    --random_block_sampling \
    --block_sampling_p_keep 0.9 \
    --block_sampling_p_none 0.2 \
    --train_set ${DATA_DIR}/rna_go_train.parquet \
    --valid_set ${DATA_DIR}/rna_go_val.parquet \
    --pretrain_weights ${PRETRAIN_WEIGHTS} \
    --pretrain_config ${PRETRAIN_CONFIG} \
    --save_dir ${SAVE_DIR} \
    --seed ${SEED} \
    --run_name rna_go_seed${SEED}
