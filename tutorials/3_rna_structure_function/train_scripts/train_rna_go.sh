#!/bin/bash
# Fine-tune ATOMICA end to end on RNA-GO, five Gene Ontology terms, multilabel.
#
# This is the one fine-tuned bar in Fig. 3. It starts from the STANDARD pretrained checkpoint,
# because GO annotations are never a pretraining input or objective, so no exclusion is called for.
# Re-run with SEED set to five different values and ensemble to reproduce the reported arm.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

python -m atomica.train \
    --gpus 0 \
    --task multilabel_classifier \
    --num_classifier_classes 5 \
    --multiclass_metric f1_macro \
    --use_focal_loss --focal_gamma 2.0 \
    --lr 4e-5 --final_lr 4e-5 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --max_epoch 200 \
    --max_n_vertex_per_gpu 256 \
    --shuffle \
    --random_block_sampling --block_sampling_p_keep 0.9 --block_sampling_p_none 0.2 \
    --train_set ${DATA_DIR}/rna_go_train.parquet \
    --valid_set ${DATA_DIR}/rna_go_val.parquet \
    $(pretrain_args pretrain) \
    --save_dir ${SAVE_ROOT}/rna_go \
    --seed ${SEED} \
    --run_name rna_go_seed${SEED}
