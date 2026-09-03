#!/bin/bash
# Fine-tune ATOMICA on RNA-Site, small-molecule-binding residues, binary per residue.
#
# Starts from the nucleic-acid-ligand-exclusion checkpoint. Fig. 3 reports RNA-Site FROZEN;
# run_probe.py produces that number. No method exceeds an AUPRC of 0.23 on this task.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

python -m atomica.train \
    --gpus 0 \
    --task residue_binary_classifier \
    --num_classifier_classes 2 \
    --lr 5e-5 --final_lr 5e-5 \
    --weight_decay 1e-3 \
    --max_epoch 400 \
    --max_n_vertex_per_gpu 256 \
    --shuffle \
    --train_set ${DATA_DIR}/rna_site_train.parquet \
    --valid_set ${DATA_DIR}/rna_site_val.parquet \
    $(pretrain_args pretrain_no_nucleic_acid_ligand) \
    --save_dir ${SAVE_ROOT}/rna_site \
    --seed ${SEED} \
    --run_name rna_site_seed${SEED}
