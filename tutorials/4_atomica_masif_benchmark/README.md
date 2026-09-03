# MaSIF-ligand pocket classification with frozen ATOMICA

Classify a protein binding pocket by which of seven small-molecule ligands binds it, using ATOMICA
embeddings that are never updated. The encoder stays frozen; only a small classifier head is
trained on top of it.

## Requirements

ATOMICA installed, see [setup](../../setup/README.md). A GPU is recommended but not required.

## Usage

```bash
python extract_embeddings.py   # frozen pocket embeddings, writes embeddings/
python run_benchmark.py        # trains the head and scores it, writes results/
```

Extraction takes a few minutes on an A100 and longer on CPU. Training the head takes about a
minute.

## Results

467 test pockets, 95% bootstrap confidence intervals over pockets.

| metric | value |
|---|---|
| macro-F1 | 0.813 [0.769, 0.853] |
| micro-F1 | 0.848 [0.816, 0.880] |
| macro-AUPRC | 0.872 [0.835, 0.909] |
| macro-AUROC | 0.974 [0.965, 0.982] |

Per-class F1:

| ligand | test pockets | F1 |
|---|---|---|
| ADP | 150 | 0.894 |
| CoA | 49 | 0.738 |
| FAD | 79 | 0.921 |
| HEM | 62 | 0.930 |
| NAD | 49 | 0.718 |
| NAP | 28 | 0.720 |
| SAM | 50 | 0.772 |

## Data

`data/` holds the three splits. A pocket is the set of residues within 8 Å of the ligand's heavy
atoms; the ligand itself is not in the graph.

| file | pockets |
|---|---|
| `masif_train.parquet` | 1,839 |
| `masif_val.parquet` | 203 |
| `masif_test.parquet` | 467 |

## Checkpoint

`checkpoints/` holds an ATOMICA model pretrained with MaSIF-similar structures excluded, so the
encoder has not seen any protein resembling a test pocket. This is not the general released model.
See [checkpoints/README.md](checkpoints/README.md).

## How it works

| step | choice |
|---|---|
| representation | `z_graph`, the rotation-invariant graph-level embedding |
| pooling | mean, standard deviation and global node over blocks, 5,376-d, parameter-free |
| preprocessing | z-score fit on training pockets only |
| head | 5,376 to 512 to 512 to 32 to 7, dropout 0.3 |
| training | Adam at 1e-3, cross-entropy, 100 epochs, early stopping on validation macro-AUPRC |
| seeds | 5, with class probabilities averaged |

## Files

```
extract_embeddings.py    frozen pocket embeddings
run_benchmark.py         head training and metrics
data/                    the three pocket splits
checkpoints/             the pretrained model
embeddings/              created by extract_embeddings.py
results/                 created by run_benchmark.py
```
