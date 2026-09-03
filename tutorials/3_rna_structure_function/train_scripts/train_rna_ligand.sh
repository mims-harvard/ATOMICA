#!/bin/bash
# Fine-tune ATOMICA on RNA-Ligand, three pocket ligand classes: PAR, LLL, 8UZ.
#
# Starts from the nucleic-acid-ligand-exclusion checkpoint, the same encoder the frozen arm uses.
# Fig. 3 reports RNA-Ligand FROZEN; run_probe.py produces that number. This script is for the
# fine-tuned arm reported alongside it.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

python -m atomica.train \
    --gpus 0 \
    --task multiclass_classifier \
    --num_classifier_classes 3 \
    --weighted_loss \
    --multiclass_metric f1_macro \
    --lr 1e-5 --final_lr 1e-5 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --max_epoch 1000 \
    --max_n_vertex_per_gpu 256 \
    --shuffle \
    --random_block_sampling --block_sampling_p_keep 0.9 --block_sampling_p_none 0.2 \
    --train_set ${DATA_DIR}/rna_ligand_train.parquet \
    --valid_set ${DATA_DIR}/rna_ligand_val.parquet \
    $(pretrain_args pretrain_no_nucleic_acid_ligand) \
    --save_dir ${SAVE_ROOT}/rna_ligand \
    --seed ${SEED} \
    --run_name rna_ligand_seed${SEED}
