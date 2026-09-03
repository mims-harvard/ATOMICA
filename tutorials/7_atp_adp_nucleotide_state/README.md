# Tutorial 7: ATP versus ADP nucleotide state

Take a nucleotide-binding site, delete the nucleotide, and ask which half of the catalytic cycle
the protein was captured in. ADP is post-hydrolysis, ATP is pre-hydrolysis. The nucleotide, every
other ligand, the ions and the waters are removed before the encoder runs, so only the protein
pocket is left.

The tutorial extracts a frozen ATOMICA representation of each pocket, trains a small classifier on
top, and scores it with cross-validation over sequence clusters.

## Setup

Install the repository environment, then download the released ATOMICA encoder:

```bash
hf download ada-f/ATOMICA --local-dir checkpoints/
```

That is the only checkpoint needed. The encoder is used frozen; the classifier is small and
retrained by the tutorial, so no weights are downloaded or saved for the task itself.

A GPU is needed for the first two stages.

## Run

```bash
cd tutorials/7_atp_adp_nucleotide_state
python tutorial.py
```

About 20 minutes on one A100. To run a single stage, or a fast smoke test:

```bash
python tutorial.py --stage embed     # frozen encoder over the 404 pockets, 2 min
python tutorial.py --stage probe     # cross-validation, 17 min
python tutorial.py --stage report    # scoring, seconds, no GPU
python tutorial.py --quick           # one seed and two hyperparameter settings
```

Output goes to `embeddings/`, `predictions/` and `results/`.

## Result

```
stratum                            AUROC             95% CI  pockets  clusters
all pockets                        0.760   [0.671, 0.837]      404        60
metal-concordant                   0.688   [0.578, 0.793]      245        44
with metal                         0.680   [0.533, 0.823]      129        25
metal-free                         0.714   [0.537, 0.867]      116        20
held out and metal-concordant      0.682   [0.404, 0.904]       50         9
```

Mean within-cluster AUROC: one AUROC inside each sequence cluster, averaged over clusters without
weighting by size. Chance is 0.500 in every row. Because pockets are never compared across
clusters, recognising the protein family cannot contribute to the score.

Your numbers will differ by a few hundredths. The eight hyperparameter settings score within about
0.005 of each other on the validation fold, so they reorder under any floating-point-level change,
including a different GPU. `predictions/atomica_selected.csv` records the margin per fold. The
narrow strata move much more than the first row; the last one rests on nine clusters and should be
read as consistent with the others rather than on its own.

## Benchmark

| | |
|---|---|
| ligand-free pockets | 404 |
| ATP-associated / ADP-associated | 223 / 181 |
| PDB entries | 120 |
| sequence clusters at 50% identity, 80% coverage | 60, each holding both states |

A pocket is the 50 protein residues nearest the bound ligand. The count is fixed rather than a
distance cutoff, because ATP is larger than ADP and under a cutoff the deleted ligand would set the
residue count. Each pocket is further restricted to the binding-site positions it shares with a
matched partner structure of the same protein in the other state, matching in amino acid, block
token and heavy-atom count, so the two present the same residues and differ only in coordinates.

One model is fitted. The four narrower strata re-partition the same out-of-fold predictions, and
membership depends on the crystal rather than on any label or score. The metal strata exist because
a bound ion correlates with nucleotide state: metal-concordant keeps only pairs whose two members
have the same metal status within 4 Å of the nucleotide, so metal presence cannot order them.

## Method

**Representation.** The frozen `z_graph` under `mean_std_global` pooling: the mean over blocks, the
per-dimension standard deviation, and the graph's global block, concatenated to 5,376 dimensions.

```python
from atomica import representations as R

rv = model.infer(batch, return_invariant_repr=True, invariant_pool=None)
z = R.pool_blocks(rv.block_invariant_repr, rv.batch_id, batch["B"] == model.global_block_id,
                  "mean_std_global", component_dims=model.invariant_component_dims(),
                  keep=shared_site_mask)
```

`keep` restricts the mean and standard deviation to the shared site, a median of 43 of the 50
residues. The rest of the pocket stays in the forward pass, since the encoder needs the whole
pocket to represent any part of it, and is left out of the pool.

**Classifier.** `atomica.probe`, the standard frozen-embedding recipe: a four-layer MLP, binary
cross-entropy, Adam, 60 epochs, batch size 32, five seeds with probabilities averaged.

**Cross-validation.** Five folds over sequence clusters. Within each fold the hyperparameters
(hidden width 64 or 256, dropout 0.1 or 0.3, learning rate 1e-4 or 1e-3) are chosen on a validation
fold drawn from the training clusters, then refitted on training and validation together and
applied once to the test fold. The test fold is never read during selection. Features are z-scored
per fold on the training rows, dropping columns with a training standard deviation below 1e-6.

**Intervals.** 2,000-resample percentile bootstrap over clusters, since pockets within a cluster
are near-duplicates of one protein.

## Files

```
data/
  pocket_graphs.parquet   404 processed pocket graphs, the model input
  pockets.csv             per pocket: label, cluster, shared site, metal status, strata,
                          PDB entry, experimental method, resolution
atp_adp.py                benchmark definition: paths, strata, readout, constants
tutorial.py               the three stages
```

No PDB files are needed; the graphs are the model input.

## Citation

If you use this benchmark, cite the ATOMICA paper.
