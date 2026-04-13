#!/bin/bash
# Train ATOMICA on the RNA_Protein residue-level binary classification task
# (protein-binding residues).
#
# Trains a single seed. Re-run with different SEED values to reproduce
# the 5-seed ensemble.
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"

SEED=${SEED:-0}
DATA_DIR=${DATA_DIR:-"${TUTORIAL_DIR}/data"}
SAVE_DIR=${SAVE_DIR:-"${TUTORIAL_DIR}/train_output/rna_protein"}
PRETRAIN_WEIGHTS=${PRETRAIN_WEIGHTS:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_weights.pt"}
PRETRAIN_CONFIG=${PRETRAIN_CONFIG:-"${REPO_ROOT}/checkpoints/pretrain/pretrain_model_config.json"}

python -m atomica.train \
    --gpu 0 \
    --task residue_binary_classifier \
    --num_classifier_classes 2 \
    --lr 5e-5 \
    --final_lr 1e-6 \
    --max_epoch 400 \
    --max_n_vertex_per_gpu 256 \
    --shuffle \
    --train_set ${DATA_DIR}/rna_protein_train.parquet \
    --valid_set ${DATA_DIR}/rna_protein_val.parquet \
    --pretrain_weights ${PRETRAIN_WEIGHTS} \
    --pretrain_config ${PRETRAIN_CONFIG} \
    --save_dir ${SAVE_DIR} \
    --seed ${SEED} \
    --run_name rna_protein_seed${SEED}
