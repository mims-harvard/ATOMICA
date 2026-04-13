# Tutorial 2: Inference with ATOMICA-Ligand

This tutorial uses the fine-tuned **ATOMICA-Ligand** models to annotate
small-molecule and metal-ion binding sites in the dark proteome. For a
chosen ligand, it loads the three released checkpoints (`v1`, `v2`,
`v3`), runs inference on the dark-proteome binding sites extracted by
PeSTo, mean-averages the three predictions, and applies the
ligand-specific threshold from `ATOMICA_ligand_thresholds.json` to
produce a final annotation table.

Supported ligands:

- **small molecules**: ADP, ATP, GTP, GDP, FAD, NAD, NAP, NDP, HEM, HEC, CIT, CLA, SAM, COA, FMN
- **metal ions**: Ca, Co, Cu, Fe, K, Mg, Mn, Na, Zn

## Requirements

- An NVIDIA H100 or A100 GPU (CUDA).
- A working ATOMICA Python environment (see the top-level
  [README](../../README.md)).
- The `hf` Hugging Face CLI (`pip install -U "huggingface_hub[cli]"`).

## Download checkpoints

Fetch the ATOMICA-Ligand checkpoints from
[Hugging Face](https://huggingface.co/ada-f/ATOMICA). Run from the
repository root:

```bash
# All ligands (both small molecules and metal ions).
hf download ada-f/ATOMICA --repo-type model \
  --local-dir checkpoints --include "ATOMICA_checkpoints/ligand/**"
```

Or, to download only one ligand (e.g. NAD):

```bash
hf download ada-f/ATOMICA --repo-type model \
  --local-dir checkpoints \
  --include "ATOMICA_checkpoints/ligand/small_molecules/NAD/**"
```

After this,
`checkpoints/ATOMICA_checkpoints/ligand/{small_molecules,metal_ions}/<LIGAND>/<LIGAND>_v{1,2,3}{.pt,_config.json}`
exist.

## Download data

The dark-proteome binding-site datasets (small molecule and ion) are
downloaded directly by the first code cell in the notebook. If you
would rather fetch them manually, they are available on
[Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX) and should be
placed at:

```
data/dark_proteome/is_dark_90_plddt_PeSTo_80_small_molecule.jsonl.gz
data/dark_proteome/is_dark_90_plddt_PeSTo_80_ion.jsonl.gz
```

## Run the notebook

Inside the notebook, use the dropdown widgets to select the ligand
type (`small_molecules` / `metal_ions`) and the specific ligand
(e.g. `NAD`, `HEM`, `ZN`).

## What the notebook does

1. **Loads checkpoints.** The three fine-tuned `ClassifierModel`
   checkpoints (`<LIGAND>_v{1,2,3}.pt` with matching
   `<LIGAND>_v{1,2,3}_config.json`) are loaded from
   `checkpoints/ATOMICA_checkpoints/ligand/<TYPE>/<LIGAND>/`.
2. **Runs inference** on the dark-proteome binding-site dataset for
   the chosen ligand type (small molecules or ions), batched at
   size 16.
3. **Ensembles** the three per-version probabilities by simple mean.
4. **Applies the per-ligand threshold** from
   `ATOMICA_ligand_thresholds.json` to produce a boolean
   `<LIGAND>_annotation` column, and prints the positive hits sorted
   by score.

## Files

```
tutorials/2_atomica_ligand/
├── README.md                         — this file
├── example_run_atomica_ligand.ipynb  — runnable notebook
└── ATOMICA_ligand_thresholds.json    — per-ligand decision thresholds
```
