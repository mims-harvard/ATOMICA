# Tutorial 5: PPI and orthosteric inhibitors

This tutorial compares ATOMICA embeddings of orthosteric PPI inhibitors
with ATOMICA embeddings of the native protein-protein (or
protein-peptide) complex they inhibit. The question: does ATOMICA
embedding similarity between an inhibitor and a partner protein/peptide
localize to the spatial region that the native partner binds?

The tutorial uses the [2P2IDB](http://2p2idb.cnrs-mrs.fr/) database of
matched PPI structures and their protein-inhibitor structures. After
quality filtering, it covers:

* 18 protein-peptide complexes matched to 268 protein-inhibitor complexes
* 6 protein-protein complexes matched to 187 protein-inhibitor complexes

Two scripts are provided:

* `tutorial_protein_peptide_inhibitors.py` — block-level comparison
  between inhibitor blocks and peptide blocks, after Kabsch alignment
  of the shared target chain.
* `tutorial_protein_protein_inhibitors.py` — interface-patch retrieval:
  for each inhibitor, rank 1,000 sampled surface patches on partner B
  by cosine distance to the inhibitor embedding and test whether
  top-ranked patches localize to the native A-B binding site on B.

## Setup

Activate the ATOMICA environment (see the top-level repository README for
installation). Then from the repository root:

```bash
cd tutorials/5_ppi_and_inhibitors
```

All data needed to run the tutorial scripts is in `data/`. No GPU is
required for the provided analysis scripts (they consume precomputed
embeddings).

## Run

```bash
python tutorial_protein_peptide_inhibitors.py
python tutorial_protein_protein_inhibitors.py
```

Each script prints summary statistics (Enrichment@10, Precision@10,
per-family Spearman correlations with FDR-BH correction, binomial test
of the fraction of positively correlated families) and writes figures to
`figures/`:

* `peptide_enrichment_at_k_10.svg`, `peptide_precision_at_k_10.svg`
* `protein_enrichment_at_k_10.svg`, `protein_precision_at_k_10.svg`
* `MENIN.MLL_4OG7_2SE_block_dist_swarm.svg` — block-distance violin +
  strip plot for the MENIN/MLL MIV-7 inhibitor example.
* `HRAS.SOS1_6ZL3_patch_dist_swarm.svg` — patch-distance violin +
  strip plot for the HRAS/SOS1 inhibitor example.

The featured swarm-plot examples are selected by system identity
(family, PDB code, ligand code) at the top of each script — edit the
`SWARM_*` constants to plot a different system.

## Data layout

```
data/
├── 2p2idb.csv                                          # 2P2IDB database export
├── ppi_inhibitor_mapping.csv                           # 2P2IDB_ID ↔ family / PDB / chains
├── inhibitors_metadata.csv                             # per-inhibitor metadata (chains, ligand codes)
├── inhibitors_processed.parquet                        # ATOMICA-processed protein-inhibitor graphs
├── inhibitors_embeddings.parquet                       # ATOMICA block embeddings for inhibitors
├── peptide_partners_processed.parquet                  # processed peptide partner graphs (≤30 residues)
├── peptide_partners_embeddings.parquet                 # ATOMICA block embeddings for peptide partners
├── peptide_inhibitor_block_results.parquet             # cached inhibitor↔peptide block distance matrices
├── protein_partner_surface_patches.parquet             # processed surface patches on partner B (>30 residues)
├── protein_partner_surface_patches_embeddings.parquet  # ATOMICA patch embeddings
└── protein_partner_surface_patches_distances.csv       # patch-to-nearest-Cα-on-A distances
```

### Inhibitors: `inhibitors_*.parquet`

The protein-inhibitor pocket is extracted around the ligand and
processed into an ATOMICA graph stored in `inhibitors_processed.parquet`
(blocks, atoms, coordinates, segment IDs). `inhibitors_embeddings.parquet`
holds the corresponding ATOMICA block embeddings.

### Peptide partners: `peptide_partners_*.parquet`

For protein-peptide complexes where the partner chain has ≤30 residues,
the partner chain is processed as a single graph. Used by
`tutorial_protein_peptide_inhibitors.py`.

### Protein partner surface patches: `protein_partner_surface_patches*`

For protein-protein complexes where the partner chain has >30 residues,
1,000 local surface patches are sampled per partner chain using
MSMS-generated molecular surfaces and area-weighted triangle sampling.
Each patch is the set of blocks within 16 Å of a sampled surface point.
`protein_partner_surface_patches_distances.csv` records the Euclidean
distance from each patch center to the nearest Cα atom on the target
chain A (the native A-B binding site).

### `peptide_inhibitor_block_results.parquet`

A cached cross-product used by the protein-peptide tutorial. For each
matched (protein-peptide PPI, protein-inhibitor) pair, it stores the
pairwise cosine distance matrix between inhibitor and peptide block
embeddings (`block_emb_dist`) and the pairwise Euclidean distance matrix
between their block centers after Kabsch alignment of the shared target
chain (`block_coords_dist`).

Regenerating this file from the raw embeddings + CIF structures requires
the 2P2IDB CIF files (not shipped with the tutorial). See the bottom of
this README.

## Regenerating the data

Every file in `data/` can be rebuilt from raw 2P2IDB structures + the
pretrained ATOMICA checkpoint with the included `prepare_data.py`:

```bash
python prepare_data.py \
  --cif_dir  /path/to/2p2idb/cifs \
  --ckpt_dir checkpoints/ATOMICA_checkpoints/pretrain
```

### Prerequisites

1. **2P2IDB structures.** A directory of `<PDB>.cif` (or `.pdb`) files
   covering all PDB codes referenced in `data/ppi_inhibitor_mapping.csv`
   and `data/inhibitors_metadata.csv`. 2P2IDB is available at
   http://2p2idb.cnrs-mrs.fr/.

2. **Pretrained ATOMICA checkpoint.**
   ```bash
   hf download ada-f/ATOMICA --repo-type model \
     --local-dir checkpoints --include "ATOMICA_checkpoints/pretrain/**"
   ```

3. **MSMS binary** on `PATH` (or `export MSMS_BIN=/path/to/msms`).
   Download from https://ccsb.scripps.edu/msms/downloads/. Required
   for generating protein-partner surface patches.

### What prepare_data.py does

1. **`inhibitors_index.csv`** — adds a `pdb_path` column pointing into
   your `--cif_dir`, fills in missing target chain IDs, and explodes
   one row per ligand residue. This is the `--data_index_file` consumed
   by `atomica.data.process_pdbs`.
2. **`peptide_partners_processed.parquet`** — extracts the partner
   chain as ATOMICA blocks for every PPI whose partner chain is ≤30
   residues.
3. **`protein_partner_surface_patches.parquet`** and
   **`protein_partner_surface_patches_distances.csv`** — for PPIs with
   partner chain >30 residues, runs MSMS on the partner chain via the
   included `surface_sampler/` package (density 3.0, probe 1.5 Å),
   area-weighted-samples 1,000 surface points (seed 42), keeps only
   points whose 16 Å neighbourhood contains ≥8 blocks, and records the
   distance from each patch centre to the nearest Cα on the target
   chain.
4. **`atomica.data.process_pdbs`** fragments the inhibitor ligand with
   `PS_300` and writes `inhibitors_processed.parquet`.
5. **`atomica.get_embeddings`** runs the pretrained model over each
   processed parquet and writes the matching `_embeddings.parquet`.
6. **`peptide_inhibitor_block_results.parquet`** — for each matched
   (peptide-PPI, protein-inhibitor) pair in the same family:
   sequence-align the shared target chain (BLOSUM62) and run iterative
   Kabsch refinement (2 Å cutoff, 5 cycles) on their Cα atoms,
   transform the inhibitor block centres into the PPI frame, then
   cache the pairwise cosine-distance matrix of block embeddings and
   the Euclidean-distance matrix of block centres (singleton blocks
   dropped).

You can skip stages with `--skip` (e.g. `--skip protein embed` to only
rebuild the peptide side). Each stage overwrites its outputs in
`data/`.
