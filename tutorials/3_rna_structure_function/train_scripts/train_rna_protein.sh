#!/bin/bash
# Fine-tune ATOMICA on RNA-Protein, protein-binding residues, binary per residue.
#
# Starts from the protein-RNA-exclusion checkpoint, so the encoder has never seen a protein-RNA
# complex. Fig. 3 reports RNA-Protein FROZEN; run_probe.py produces that number.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

python -m atomica.train \
    --gpus 0 \
    --task residue_binary_classifier \
    --num_classifier_classes 2 \
    --lr 5e-5 --final_lr 1e-6 \
    --weight_decay 1e-3 \
    --max_epoch 400 \
    --max_n_vertex_per_gpu 256 \
    --shuffle \
    --train_set ${DATA_DIR}/rna_protein_train.parquet \
    --valid_set ${DATA_DIR}/rna_protein_val.parquet \
    $(pretrain_args pretrain_no_protein_rna) \
    --save_dir ${SAVE_ROOT}/rna_protein \
    --seed ${SEED} \
    --run_name rna_protein_seed${SEED}
