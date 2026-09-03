# Tutorial 5: PPI interfaces and orthosteric inhibitors

An orthosteric inhibitor of a protein-protein interaction binds at the interface and blocks
the native partner. This tutorial tests whether ATOMICA places such an inhibitor and the
partner it displaces close together in embedding space, even though one is a drug-like
molecule and the other is part of a protein.

It uses the [2P2Idb](http://2p2idb.cnrs-mrs.fr/) database of protein-protein complexes
matched to protein-inhibitor structures, and reproduces the ATOMICA results of Figure 4 of
the paper. Two analyses, one per kind of partner:

- **Protein-peptide.** When the partner is a peptide of 30 residues or fewer, inhibitor
  blocks are compared with peptide blocks after superposing the shared target chain. Are
  the pairs closest in embedding space also close in space?
- **Protein-protein.** When the partner is a whole protein, 1,000 patches sampled on its
  surface are ranked by embedding distance to the inhibitor. Do the top-ranked patches lie
  on the native binding site?

A block is one residue on the protein side and one fragment on the ligand side. Fold
Change@10 is Precision@10 divided by the fraction of all candidates that are spatially
close, so 1 is what a uniform random ranking gives.

## Requirements

- The ATOMICA environment, as described in the top-level repository README.
- The `data/` directory from [Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX),
  five files and about 22 MB.
- The pretrained ATOMICA checkpoint. No other checkpoint is needed.
- A GPU for the embedding step. It runs on CPU with `--device cpu`, more slowly.

## Run

Download the checkpoint:

```bash
hf download ada-f/ATOMICA --repo-type model \
  --local-dir checkpoints --include "ATOMICA_checkpoints/pretrain/**"
```

Embed the structures. This writes three files into `embeddings/` and takes about ten
minutes on one A100:

```bash
cd tutorials/5_ppi_and_inhibitors
python compute_embeddings.py --ckpt_dir checkpoints/ATOMICA_checkpoints/pretrain
```

Run the two analyses:

```bash
python tutorial_protein_peptide_inhibitors.py
python tutorial_protein_protein_inhibitors.py
```

Each prints per-complex and per-superfamily tables and writes figures to `figures/`.

## Results

| | protein-peptide | protein-protein |
|---|---|---|
| complexes | 26 | 6 |
| matched protein-inhibitor structures | 965 | 187 |
| 2P2Idb superfamilies | 14 | 5 |
| superfamilies with Fold Change@10 > 1 | 11/14 | 5/5 |
| mean Fold Change@10 over superfamilies | 1.136 | 2.288 |
| mean Fold Change@10 over complexes | 1.148 | 2.605 |
| superfamilies with positive Spearman | 10/14 | 4/5 |

Per superfamily, protein-protein: RAS 4.186, TNFR 2.556, Interleukin 1.959, XIAP 1.411,
E2 1.330.

Figures:

- `peptide_fold_change.svg`, `peptide_precision.svg`
- `protein_fold_change.svg`, `protein_precision.svg`
- `MENIN.MLL_4OG7_2SE_blocks.svg`, the MENIN/MLL inhibitor MIV-7 against the MLL peptide
- `HRAS.SOS1_6ZL3_patches.svg`, an HRAS/SOS1 inhibitor against SOS1 surface patches

Both examples are selected by identity at the top of their script, not by rank. Edit the
`EXAMPLE` constant to feature a different one; `("KRAS/SOS1", "7RT1")` gives the KRAS panel.

## Which representation

Both embeddings come from `atomica.representations`, which names each vector the way the
paper does. Run `python -m atomica.representations --guidance` for the full list.

| name | what it is | used for |
|---|---|---|
| `h_block` | one vector per residue or ligand fragment | the protein-peptide comparison, and the protein-protein query |
| `h_interface` | learned attention pooling over one molecule's blocks, after message passing over the whole complex | the protein-protein patch candidates |

The protein-peptide analysis ranks individual blocks, so nothing is pooled there.

The protein-protein query is the unweighted mean of the inhibitor's block embeddings over
the ligand, excluding the global block node ATOMICA prepends to each segment. Those blocks
are still contextualized by the target pocket, because message passing runs over the whole
pocket-plus-ligand graph before any block is read out, but no pocket block enters the query
or the ranking.

## Aggregation

Complexes that share a target superfamily are not independent. Four of the protein-peptide
complexes are bromodomain readers of the same histone H4 tail, and KRAS-SOS1 and HRAS-SOS1
are the same interaction with a different RAS paralogue. Averaging over complexes would let
the 338 inhibitor structures matched to BRD4-1/H4 outweigh the 5 matched to BRD3-2/H3.

So every number across systems is an unweighted mean taken in stages: per structure, then
within a complex, then within a 2P2Idb superfamily, then over superfamilies. Each
superfamily contributes one value whatever its size. The grouping is the `superfamily`
column of `data/metadata.csv`.

The Spearman correlations are reported for sign and magnitude only. Block pairs within one
complex share blocks, and patches within one complex are ranked by every inhibitor of that
complex, so neither set is independent and no test is run over them.

## Files

`data/` is the download, five files and 22 MB:

```
metadata.csv                         one row per protein-inhibitor structure: chains,
                                     ligand, family, superfamily, and the PPI complex
                                     it inhibits
inhibitors_processed.parquet         pocket and ligand graphs
peptide_partners_processed.parquet   peptide partner graphs
surface_patches_processed.parquet    sampled surface patch graphs, each carrying its
                                     distance to the nearest target CA atom
peptide_inhibitor_geometry.parquet   superposed block-centre distances per matched pair
```

`compute_embeddings.py` writes three files into `embeddings/`. They are not part of the
download because they are reproducible from `data/` and the public checkpoint.

`peptide_inhibitor_geometry.parquet` holds a row index into
`inhibitors_processed.parquet`, the peptide it was superposed onto, the superposition
RMSD, and the block-centre distance matrix. It stores nothing that can be derived, and no
embeddings, so it does not change when the model does. The analysis script recovers the
family, ligand code and pair counts and applies the inclusion cut-offs, which keep a match
when the superposed inhibitor lands within 2 A of the peptide and leaves more than 10
block pairs.

## Rebuilding data/

`prepare_data.py` regenerates everything above from raw 2P2Idb structures. It needs a
directory of 2P2Idb mmCIF files and the MSMS binary from
https://ccsb.scripps.edu/msms/downloads/ on `PATH`.

```bash
python prepare_data.py --cif_dir /path/to/2p2idb/cifs
```

It builds the inhibitor index, then prints the `atomica.data.process_pdbs` command that
turns it into `inhibitors_processed.parquet`, then builds the peptide partners, the surface
patches and the geometry. `--skip` runs a subset of the stages. `metadata.csv` is the one
input it does not regenerate; it is the curated 2P2Idb selection this tutorial uses.
