# Metal coordination probes

Fit a linear classifier on frozen ATOMICA embeddings to predict a metal site's coordination number
and its coordination geometry. ATOMICA puts a metal ion into the graph as an ordinary atom node,
with no coordination-aware featurization, so this shows what the representation carries anyway.

The encoder is frozen and the only thing fitted is a logistic regression, so the result is a
statement about the representation rather than about a head trained on top of it.

## What you need

- ATOMICA installed, per the repository root README.
- The pretrained checkpoint, see `checkpoints/README.md`.
- `data/`, 28 MB, ships with the tutorial. Nothing else to download.
- A GPU for step 1, about 10 minutes. Step 2 is CPU only, about an hour.

## Running it

```bash
python extract_embeddings.py   # frozen encoder over the metal pockets, one vector per site
python run_probe.py --all      # fit the probes and print the results
```

To run a single task:

```bash
python run_probe.py --task geometry
```

Step 1 writes `embeddings/`, step 2 writes `predictions/`. Both are regenerated, not stored.

## Results

Balanced accuracy on the held-out test split, with 95% intervals from resampling PDB entries.

| task | classes | test sites | chance | frozen ATOMICA |
|---|---|---|---|---|
| coordination number, all deposited donors | 8 | 3,654 | 0.125 | **0.377** [0.337, 0.423] |
| coordination number, visible protein donors | 8 | 3,654 | 0.125 | **0.590** [0.538, 0.654] |
| coordination geometry | 14 | 2,313 | 0.071 | **0.342** [0.314, 0.375] |
| geometry at deposited coordination number 6 | 3 | 635 | 0.333 | **0.496** [0.408, 0.583] |

The last row is the sharpest test. Every site there has six deposited donors, so nothing can be
recovered by counting them and only the angular arrangement separates the three classes.

Balanced accuracy is the primary metric because the classes are strongly imbalanced; accuracy and
macro-F1 are printed alongside.

## How it works

**Representation.** `atomica.representations.get(model, batch, "z_block")`

**Probe.** `atomica.probe.fit_linear_probe`, an L2 multinomial logistic regression with no hidden
layer. Features are z-scored on the training split, the regularization strength is chosen on
validation, and the test split is read once at the end.

**Intervals.** `atomica.probe.cluster_bootstrap_ci` resamples PDB entries rather than sites,
because several metal sites can come from one structure.

**Two coordination numbers.** ATOMICA's pockets hold amino acids only, while MetalPDB counts the
complete deposited first sphere. Half the test sites have at least one donor the model cannot see,
so the all-donor target needs information the input withholds and the protein-donor target does
not. Both are reported.

## Labels

MetalPDB, with geometry assignments from FindGeo. See `data/README.md` for the columns and the
coverage. Cite Putignano et al., *Nucleic Acids Research* 46(D1):D459-D464, 2018 and Andreini
et al., *Bioinformatics* 28(12):1658-1660, 2012.
