# Tutorial 3: RNA Structure-Function Prediction

This tutorial reproduces the ATOMICA paper results on four RNA
structure-function benchmarks from [RNAglib](https://rnaglib.org/):

| Task          | Level    | Type                       | Output                         |
|---------------|----------|----------------------------|--------------------------------|
| `RNAGo`       | graph    | multi-label (5 classes)    | Gene Ontology term membership  |
| `RNA_Ligand`  | pocket   | multi-class (3 ligands)    | ligand class of an RNA pocket  |
| `RNA_Protein` | residue  | binary                     | protein-binding residues       |
| `RNA_Site`    | residue  | binary                     | small-molecule-binding residues|

For each task we load the five fine-tuned ATOMICA checkpoints shipped in
`checkpoints/benchmarks/`, run inference on the test set, ensemble the five
probability outputs by mean, and print per-seed and ensemble metrics that
match those reported in the paper.

## Requirements

- An NVIDIA H100 or A100 GPU (CUDA)
- A working ATOMICA Python environment
  (`~/.conda/envs/interactenv/bin/python` on our cluster)

## Download checkpoints

Fetch the five-seed fine-tuned checkpoints for the four RNAglib tasks
from [Hugging Face](https://huggingface.co/ada-f/ATOMICA) and place
them where `tutorial.py` expects them. Run from the repository root:

```bash
# 1. Download the rnaglib checkpoints from Hugging Face.
hf download ada-f/ATOMICA --repo-type model \
  --local-dir checkpoints --include "ATOMICA_checkpoints/rnaglib/**"

# 2. Move them to the layout tutorial.py expects
#    (rna_go stays flat; the other three nest under an "atomica/" subdir).
mkdir -p checkpoints/benchmarks
mv checkpoints/ATOMICA_checkpoints/rnaglib/rna_go \
   checkpoints/benchmarks/rna_go
for task in rna_ligand rna_protein rna_site; do
  mkdir -p checkpoints/benchmarks/${task}
  mv checkpoints/ATOMICA_checkpoints/rnaglib/${task} \
     checkpoints/benchmarks/${task}/atomica
done
```

After this, `checkpoints/benchmarks/{rna_go,rna_ligand/atomica,rna_protein/atomica,rna_site/atomica}/seed{0..4}/{config.json,model.pt}`
exist.

## Download data

Download the `RNAGlib/` directory from [Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX).

## Quick start

```bash
# run inference (all four tasks)
python tutorial.py

# run a single task
python tutorial.py --tasks RNA_Ligand

# retrain one seed of a task (pretrain weights required at
# <repo_root>/checkpoints/pretrain/). Re-run with SEED=0..4 for the ensemble.
SEED=0 bash train_scripts/train_rna_ligand.sh
```

Predictions are saved to `predictions/`:

- `{TASK}_test_predictions.parquet` — per-seed test predictions
  (`id`, `label`, `pred_probability`, `pred`, `seed`, `ckpt`).
- `{TASK}_test_ensemble.parquet` — mean-ensemble probabilities over the 5 seeds.

## What the tutorial does

1. **Loads checkpoints.** `checkpoints/benchmarks/{rna_go,rna_ligand/atomica,rna_protein/atomica,rna_site/atomica}/seed{0..4}/model.pt`
   together with the matching `config.json`. Model classes:
   - `MultiLabelClassifierModel` for RNAGo,
   - `MultiClassClassifierModel` for RNA_Ligand,
   - `ResidueClassifierModel` for RNA_Protein and RNA_Site.
2. **Runs inference** on the test set stored in `data/{task}_test.parquet`
   via `MultiClassLabelledPDBDataset`. For RNA_Protein the validation set is
   also scored because the ensemble decision threshold is tuned on val F1
   (matching the paper).
3. **Ensembles** the 5 per-seed probabilities by simple mean and computes
   metrics using the same helpers used to generate the paper tables
   (`multiclass_metrics.py`, `multilabel_metrics.py`, and `sklearn` for the
   two binary residue tasks).
4. **Prints and saves**:
   - a preview of the per-seed test predictions,
   - a per-seed + ensemble metrics table,
   - a summary (mean / std across seeds, plus the ensemble row),
   - parquet files under `predictions/`.

## Task details

### RNA-Protein (residue-level binary)
Predicts whether each RNA residue is part of a protein binding site.
The dataset contains 891 training structures (52,175 residues, 26.7% positive),
191 validation structures (11,063 residues, 27.1% positive), and 190 test
structures (10,851 residues, 27.4% positive). Positive labels are assigned to
residues within 8 Å of any protein atom in a protein-RNA complex. Splits are
defined by clustering with USalign at a similarity threshold of 0.5.

Fine-tuning passes the d\_node residue-level embeddings to a 3-layer MLP
binary classifier, trained with initial LR 5e-5, final LR 1e-6, weight decay
1e-3, gradient clipping 1.0, for 400 epochs with AUPRC as the validation
metric. Five seeds are trained with identical architecture and
hyperparameters and ensembled by the mean of predicted logits. Metrics
reported: accuracy, ROC-AUC, and AUPRC.

### RNA-GO (graph-level multi-label)
Predicts which of five Gene Ontology terms are associated with an RNA
molecule. 349 training / 75 validation / 75 test samples. Classes:
GO:0000353 (33.5%), GO:0010468 (20.0%), GO:0005682 (16.2%), GO:0005688
(15.6%), GO:0005686 (14.2%). Of the 499 total samples, 161 have no GO term,
179 have one, and 159 have two. Splits follow a 60% sequence-identity split.

Fine-tuning uses the d\_node graph-level embedding with a 4-layer MLP, focal
loss (γ=2.0) to counter class imbalance, constant LR 4e-5, weight decay 1e-3,
gradient clipping 1.0, 200 epochs, with F1-macro as the validation metric. 10%
of residues are randomly masked for 80% of training samples as an
augmentation. Five seeds are ensembled by mean of predicted logits. Metrics
reported: subset accuracy, F1 (macro/micro/weighted), AUPRC (macro/micro),
and ROC-AUC OvR (macro/micro), all at a fixed 0.5 threshold to match the
baselines.

### RNA-Site (residue-level binary)
Predicts whether each RNA residue is part of a small-molecule binding site.
157 training structures (10,092 residues, 7.8% positive), 34 validation
structures (2,162 residues, 7.8% positive), and 33 test structures (2,158
residues, 7.8% positive). A residue is labeled positive if it lies within 8 Å
of any atom of a bound ligand. Splits are defined by clustering with USalign
at a similarity threshold of 0.5.

Fine-tuning passes the d\_node residue-level embeddings to a 3-layer MLP
binary classifier, trained with constant LR 5e-5, weight decay 1e-3, no
gradient clipping, 400 epochs, with AUPRC as the validation metric. Five
seeds are ensembled by mean of predicted logits. Metrics reported: accuracy,
ROC-AUC, and AUPRC.

### RNA-Ligand (pocket-level multi-class)
Predicts the ligand type of an RNA binding pocket across three classes:
Paromomycin (PAR, 22.3%), Gentamycin C1A (LLL, 67.2%), and Aminoglycoside
TC007 (8UZ, 10.0%). 203 training / 43 validation / 44 test pockets. Pockets
are expanded from residues within 8 Å of ligand atoms. Splits are defined by
clustering with USalign at a similarity threshold of 0.5.

Fine-tuning uses the d\_node graph-level embedding with a 4-layer MLP and a
weighted cross-entropy loss (weights = inverse training-label frequency),
constant LR 1e-5, weight decay 0.1, gradient clipping 1.0, 400 epochs, with
F1-macro as the validation metric. 10% of residues are randomly masked for
80% of training samples as an augmentation. Five seeds are ensembled by mean
of predicted logits. Metrics reported: accuracy, balanced accuracy, F1
(macro/micro/weighted), AUPRC OvR (macro/weighted), and ROC-AUC OvR
(macro/weighted).

## Directory layout

```
3_rna_structure_function/
├── README.md
├── tutorial.py                 # main inference + metrics script
├── multiclass_metrics.py       # helper metrics (RNA_Ligand)
├── multilabel_metrics.py       # helper metrics (RNAGo)
├── train_scripts/              # per-task training launchers
│   ├── train_rna_go.sh
│   ├── train_rna_ligand.sh
│   ├── train_rna_protein.sh
│   └── train_rna_site.sh
├── data/                       # train / val / test parquets per task
│   ├── rna_go_{train,val,test}.parquet
│   ├── rna_ligand_{train,val,test}.parquet
│   ├── rna_protein_{train,val,test}.parquet
│   └── rna_site_{train,val,test}.parquet
└── predictions/                # outputs of tutorial.py
    ├── {TASK}_test_predictions.parquet
    └── {TASK}_test_ensemble.parquet
```

Checkpoints live in the repo root at `checkpoints/benchmarks/`.

## Expected output (per-seed test metrics)

The per-seed numbers printed by `tutorial.py` are identical to those saved in
`checkpoints/benchmarks/<task>/seed{0..4}/test_atomica_results.parquet`, so
you can diff the two to confirm your environment reproduces the paper
numbers. Representative values:

- **RNAGo** seed 4: subset accuracy 0.9200, F1-macro 0.8850, F1-micro 0.9130;
  ensemble: subset accuracy 0.9067, F1-macro 0.9514.
- **RNA_Ligand** seed 2: accuracy 0.9091, F1-macro 0.9103;
  ensemble: AUPRC OvR macro 0.9272, ROC-AUC OvR macro 0.9669.
- **RNA_Protein** ensemble: accuracy 0.7259, ROC-AUC 0.7742, AUPRC 0.6039.
- **RNA_Site** ensemble: accuracy 0.9217, ROC-AUC 0.6230, AUPRC 0.2199
  (val-tuned threshold).

## Notes

- Inference is done one example at a time (batch size 1) for simplicity; the
  full four-task run takes a few minutes on an H100.
- For both RNA_Protein and RNA_Site the ensemble decision threshold is tuned
  on the validation set's F1 curve.
- RNA_Protein/RNA_Site "ensemble" accuracy is computed using that F1-tuned
  threshold; ROC-AUC and AUPRC are threshold-free and match cleanly between
  per-seed and ensemble reporting.
