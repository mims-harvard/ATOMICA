#!/bin/bash
# Shared settings for the fine-tuning launchers.
#
# Fine-tuning is not the headline: Figure 3 reports a fine-tuned ATOMICA only for RNA-GO, and
# run_probe.py produces the other three. Each task starts from the checkpoint its frozen arm uses,
# so the pretraining exclusion is preserved.
#
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=${DATA_DIR:-"${TUTORIAL_DIR}/data"}
CKPT_DIR=${CKPT_DIR:-"${TUTORIAL_DIR}/checkpoints"}
SAVE_ROOT=${SAVE_ROOT:-"${TUTORIAL_DIR}/train_output"}

# The released checkpoints used seeds 8, 2026, 2025, 7 and 15, so SEED=0..4 reproduces the method
# and not those weights.
SEED=${SEED:-0}

pretrain_args() {
  echo "--pretrain_weights ${CKPT_DIR}/$1/weights.pt --pretrain_config ${CKPT_DIR}/$1/config.json"
}
