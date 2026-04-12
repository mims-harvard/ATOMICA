# Tutorial 4 — MASIF-Ligand benchmark with ATOMICA

This tutorial reproduces the ATOMICA result on the **MASIF-Ligand** protein
pocket classification benchmark using the five released checkpoints
(`checkpoints/benchmarks/masif/8A/seed{0..4}`).

## Task

The benchmark evaluates protein pocket classification across 7 common small
molecule ligands: ADP (28.9%), CoA (12.6%), FAD (16.2%), heme (12.8%),
NAD (11.4%), NAP (8.0%), and SAM (10.2%). The dataset contains 2,509 total
pockets split into 1,839 training, 203 validation, and 467 test pockets.
Binding pockets are defined as residues within 8 Å of the ligand heavy
atoms.

## Model

For protein-ligand pocket classification, we use a pocket-level multiclass
classifier to predict one of the 7 ligand classes. To address class
imbalance, we use weighted cross-entropy loss with class weights inversely
proportional to the training-set frequencies. The classifier is a 4-layer
MLP on top of the ATOMICA graph-level pocket embeddings, trained with a
constant learning rate of 3e-5, weight decay of 1e-3, no gradient clipping,
for 300 epochs, with F1-macro as the validation metric.

Five models are trained with different random seeds but identical
architecture and hyperparameters, and ensembled by mean-pooling the
predicted probabilities.

## What this tutorial does

`tutorial.py`:

1. loads the five fine-tuned ATOMICA checkpoints (seed 0–4) for the chosen
   pocket distance cutoff,
2. runs live inference on `data/masif_ligand_pdbs_<dist>A_pocket_only_test.parquet`
   (467 pockets),
3. mean-probability ensembles the five models,
4. saves per-seed and ensemble predictions to `predictions/`, and
5. reports per-seed and ensemble test-set accuracy.

## Files

```
tutorials/4_atomica_masif_benchmark/
├── README.md                                           — this file
├── tutorial.py                                         — the runnable script
├── train.sh                                            — training launcher (per seed)
├── data/
│   ├── masif_train.parquet                             — 1,839 training pockets
│   ├── masif_val.parquet                               — 203 validation pockets
│   └── masif_test.parquet                              — 467 test pockets
└── predictions/                                        — created on first run
    └── masif_test_predictions.parquet                  — ensembled test predictions
```

Checkpoints live outside the tutorial dir:

```
checkpoints/benchmarks/masif/
└── 8A/seed{0..4}/{config.json, model.pt, test_preds.parquet, ...}
```

## Requirements

- An NVIDIA GPU (A100/H100 recommended) with CUDA.
- The conda environment with ATOMICA installed (see the
  top-level `README.md` / `install_atomica_conda.sh`).

## Usage

From this directory:

```bash
# Evaluate the 5-seed ensemble on the test set.
python tutorial.py

# Retrain a single seed from scratch (pretrain weights required in
# checkpoints/pretrain/). Repeat with SEED=0..4 to reproduce the ensemble.
SEED=0 bash train.sh
```

Inference over all 5 seeds runs in a couple of minutes on an A100.

## Expected output

```
Accuracy: ATOMICA 0.8587
```
