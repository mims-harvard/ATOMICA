# Tutorial 3: RNA structure-function prediction

Reproduces the four RNA panels of Figure 3 on the [RNAglib](https://rnaglib.org/) benchmarks.

| Task | Unit of prediction | Type | Primary metric |
|---|---|---|---|
| RNA-Protein | residue | binary | AUPRC |
| RNA-Site | residue | binary | AUPRC |
| RNA-Ligand | pocket | 3-class | macro-F1 |
| RNA-GO | molecule | 5-label multilabel | macro-F1 |

**The encoder is frozen.** Unless a bar is labelled fine-tuned, ATOMICA in Figure 3 means frozen
embeddings passed to the same four-layer MLP that every frozen baseline gets, and that MLP is the
only fitted component. Figure 3 reports a fine-tuned ATOMICA for RNA-GO and nothing else.

Three of the four tasks use an encoder that never saw the interaction type the labels are about:

| Task | Pretrained encoder |
|---|---|
| RNA-Protein | all protein-RNA complexes excluded from pretraining |
| RNA-Site | all nucleic-acid-ligand complexes excluded |
| RNA-Ligand | all nucleic-acid-ligand complexes excluded |
| RNA-GO | the standard checkpoint, since GO terms are never a pretraining input |

## Requirements

- An ATOMICA environment with this repository importable.
- A CUDA GPU for `extract_embeddings.py` and `run_finetuned_rna_go.py`, about 4 minutes for all
  four tasks on an A100. `run_probe.py` runs on CPU.
- About 1.5 GB of scratch space for the extracted embeddings.

## Setup

Download the RNAglib task files from [Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX) into
`data/` and the checkpoints from [Hugging Face](https://huggingface.co/ada-f/ATOMICA) into
`checkpoints/`.

```
data/
  rna_protein_{train,val,test}.parquet   structures and labels, one file per split
  rna_site_{train,val,test}.parquet
  rna_ligand_{train,val,test}.parquet
  rna_go_{train,val,test}.parquet
checkpoints/
  pretrain/                         standard ATOMICA
  pretrain_no_protein_rna/          protein-RNA exclusion
  pretrain_no_nucleic_acid_ligand/  nucleic-acid-ligand exclusion
  rna_go_finetuned/seed{0..4}/      the five released RNA-GO fine-tunes
```

`residue_labels.parquet` ships with the tutorial. It gives the two residue-level tasks their labels
keyed by PDB residue index, which is how the reported numbers were produced.

Two of the checkpoints are specific to this tutorial and are not the standard ATOMICA release:
`pretrain_no_protein_rna/` and `pretrain_no_nucleic_acid_ligand/`, 33 MB each.

## Running

```bash
python extract_embeddings.py --all   # frozen encoder -> z_block, GPU
python run_probe.py --all            # train the probe, print test-set performance
python run_finetuned_rna_go.py       # the one fine-tuned bar, GPU
```

Useful variants:

```bash
python run_probe.py --task RNA_Site --feature h_block   # 32-d readout, as a control
python run_probe.py --task RNAGo --select-loss          # validation picks the loss
python extract_embeddings.py --all --batch-size 16      # same vectors, fewer forward passes
```

Batches hold only structures that share a largest block, so `--batch-size` changes speed and not
the vectors. If extraction runs out of memory, lower it, pass `--atom-budget 12000`, or use
`--device cpu`.

## Method

| Step | Choice |
|---|---|
| representation | `z_block` per residue for RNA-Protein and RNA-Site, `z_graph` per molecule for RNA-Ligand and RNA-GO |
| width | `z_block` is 1792: a 32-d scalar readout, 544 Gram entries, 1216 atom mean and std. `z_graph` is 5376 |
| pooling | mean over blocks, standard deviation over blocks, and the global block node; no parameters |
| preprocessing | z-score fitted on the training split only |
| head | 512 to 512 to 32 to classes, ReLU, dropout 0.3, BatchNorm after each linear layer |
| training | Adam, learning rate 1e-3, weight decay 1e-4, 200 epochs, batch 256, early stopping on the validation metric with patience 20 |
| loss | binary cross-entropy on the residue tasks, weighted cross-entropy on RNA-Ligand, focal cross-entropy on RNA-GO |
| ensemble | 5 seeds, mean of predicted probabilities |
| intervals | 95% percentile bootstrap over test points, 2,000 resamples, macro metrics resampled within class |

The head, training loop, seed ensembling, metrics, bootstrap and pooling come from `atomica.probe`,
and the representation from `atomica.representations`.

The pooling has no parameters because the frozen baselines get a plain mean over residues and an
MLP; a learned pooler on top of ATOMICA would compare set aggregation rather than representations.

The `z` family is used rather than the standard 32-d readout because the latter keeps only the
scalar channels. On RNA-Site, same encoder and same head:

| `--feature` | Width | AUPRC |
|---|---|---|
| `h_block`, the scalar readout alone | 32 | 0.135 |
| `z_block_gram`, plus the block's Gram entries | 576 | 0.165 |
| `z_block`, plus the atom descriptor | 1792 | 0.243 |

## Results

`run_probe.py` prints the test-set metric for each task with its bootstrap interval. Running this
tutorial end to end gives:

| Task | Metric | This tutorial | 95% CI | Published | Best frozen baseline |
|---|---|---|---|---|---|
| RNA-Protein | AUPRC | 0.604 | [0.588, 0.620] | 0.612 | gRNAde 0.486 |
| RNA-Site | AUPRC | 0.219 | [0.164, 0.285] | 0.223 | RiNALMo 0.197 |
| RNA-Ligand | macro-F1 | 0.573 | [0.425, 0.725] | 0.556 | RiNALMo 0.477 |
| RNA-GO | macro-F1 | 0.688 | [0.495, 0.845] | 0.673 | RiNALMo 0.885 |

Fine-tuned ATOMICA on RNA-GO reproduces its published macro-F1 of 0.951 and micro-F1 of 0.920.

Expect the frozen numbers to land near rather than exactly on the published ones: the encoder
forward pass is not bit-reproducible on GPU, and each task's five-seed spread is comparable to the
differences above. RNA-Site and RNA-Ligand have 2,158 residues and 44 pockets in their test sets,
so their intervals are wide, and no method exceeds an AUPRC of 0.23 on RNA-Site.

## Fine-tuning

`train_scripts/` retrains any task from the checkpoint its frozen arm uses.

```bash
SEED=0 bash train_scripts/train_rna_go.sh    # then SEED=1..4 for the five-seed ensemble
```

The released checkpoints were trained with seeds 8, 2026, 2025, 7 and 15, so `SEED=0..4` reproduces
the method and not those weights.

## Files

```
rna_tasks.py             task table, paths, data loading
extract_embeddings.py    frozen encoder -> z_block
run_probe.py             the probe and its test-set metrics
run_finetuned_rna_go.py  the five released RNA-GO fine-tunes
residue_labels.parquet   residue-level labels keyed by PDB residue index
train_scripts/           fine-tuning launchers
```

`embeddings/` and `predictions/` are generated and are not tracked.
