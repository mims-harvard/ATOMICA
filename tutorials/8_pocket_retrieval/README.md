# Same-ligand pocket retrieval across structurally distinct proteins

Given an empty protein pocket, find other pockets that bind the same ligand, when none of them
belongs to a structurally similar protein. Nothing is trained and no label reaches the model: the
method is a cosine between frozen ATOMICA vectors.

## Setup

Only the released ATOMICA pretrain checkpoint is needed. A GPU is recommended; extraction takes
about a minute on an A100 and also runs on CPU with `--device cpu`.

```bash
mamba activate atomica-env     # or: source atomica-env/bin/activate

hf download ada-f/ATOMICA --repo-type model --local-dir checkpoints \
    --include "ATOMICA_checkpoints/pretrain/**"
```

## The benchmark

`data/` holds 892 ligand-free pockets, 428 of them used as queries, over 44 ligand classes and
2,349 positive pairs. A pocket is the 50 protein residues nearest the bound ligand, with the
ligand deleted before anything is embedded.

For a query, a candidate pocket is

- **relevant** if it binds the query's ligand and has no detectable Foldseek alignment to it,
- **ignored** if it binds the query's ligand but the two proteins do align,
- **negative** otherwise.

Ignoring rather than penalising the alignable same-ligand pockets is what makes this a test of
site recognition rather than family recognition. Results are macro-averaged over the 205 sequence
clusters, so a cluster contributing many near-duplicate pockets is not counted many times.

## Running it

```bash
python extract_representations.py     # one frozen vector per pocket, writes representations/
python evaluate_retrieval.py          # rank by cosine and score, writes results/
```

Extraction uses `z_graph` pooled with `mean_component_normalized`. Both halves matter: `z_graph`
keeps the higher-degree channels that `h_graph` discards, and the component normalisation puts the
three parts of the representation on an equal footing, without which the atom-pooled part takes
almost all of every cosine. Use `mean_std_global` instead only when training a head on top.

## What you should see

Six ranking statistics with a 95% bootstrap interval, and the lift over a permutation reference.
Chance is obtained by permuting each query's ranking rather than assumed, since at 1% positive
prevalence none of these statistics has chance 0.5.

```
                                 mAP     AUROC      nDCG   R_precision       MRR    Hit@20
random reference              0.0184     0.500     0.246        0.0112    0.0469     0.187
ATOMICA (frozen)              0.0611     0.622     0.302        0.0586    0.1273     0.326

ATOMICA (frozen): mAP lift over chance 3.32x  (1792 dimensions, 428 queries)
```

The full run with intervals is in `expected_results/retrieval_metrics.json`.

## Scoring another representation

Write an `.npz` with `ids` and `vectors` covering the same 892 pockets, then pass it in:

```bash
python evaluate_retrieval.py --vectors "my method=path/to/vectors.npz"
```

## Files

| path | contents |
|---|---|
| `data/pockets.parquet` | the 892 ligand-free pockets, tokenised as model inputs |
| `data/benchmark.json` | pool, query set, relevant and ignored sets, ligand and cluster labels |
| `extract_representations.py` | step 1 |
| `evaluate_retrieval.py` | step 2 |
| `benchmark.py` | ranking rule, statistics, chance and macro weighting |
