# InteractScore: per-residue importance for a protein–ligand interface

This tutorial shows how to compute an **InteractScore** for every interface residue in a protein–ligand complex using the pretrained ATOMICA model. The notebook:

1. loads a protein–ligand structure from `data/example/example_inputs.csv` (the `6llw` + `UDP` entry),
2. masks each interface residue one at a time,
3. measures the cosine similarity between the complex embedding before and after masking,
4. maps each block back to its original PDB chain / residue index using `block_to_pdb_indexes`, and
5. prints every interface residue ranked by score in **increasing order** (most impactful residues first — a lower cosine similarity means masking that residue changed the representation more, i.e. it matters more for the interaction).

Requirements:

- A CUDA-capable GPU (e.g. H100 / A100).
- The `atomica` python environment set up (see the top-level [README](../../README.md)).
- The pretrained ATOMICA checkpoint downloaded into `checkpoints/` (the notebook downloads it for you if missing).

## Run the notebook

```bash
jupyter notebook example_run_interact_score.ipynb
```

The notebook resolves all paths relative to the repository root (it walks upward from its own location looking for `pyproject.toml` + `src/atomica/`), so it is portable — you can clone the repo anywhere and it will still run.

## What you get

For the included example (`6llw_A_A_UDP`, a protein bound to UDP) the notebook produces a table of interface residues like:

```
chain A residue  362  (block  25)  interact_score = 0.9854
chain A residue  340  (block  15)  interact_score = 0.9884
chain A residue  344  (block  19)  interact_score = 0.9945
...
```

where `interact_score` is the cosine similarity between the original and masked complex embeddings, `block` is the internal block index, and `chain` / `residue` are the original PDB identifiers recovered via `block_to_pdb_indexes`.

## Running it on your own structures

Edit `data/example/example_inputs.csv` (or point the notebook at your own `--data_index_file`) to include your structure, then update `EXAMPLE_ID` in the notebook to the corresponding processed-id (typically `{pdb_id}_{chain1}_{chain2}[_{lig_code}]`). See [`src/atomica/data/README.md`](../../src/atomica/data/README.md) for the input CSV schema.

## How the score is computed

Implemented in [`src/atomica/interaction_profiler/interact_score.py`](../../src/atomica/interaction_profiler/interact_score.py):

- `mask_block(data, block_idx)` replaces a single residue block with a `MASK` token (coordinates averaged, atom list collapsed).
- `get_residue_model_score(model, data, block_idx)` runs the model on the original and the masked complex and returns the cosine similarity between the two graph-level embeddings.
- `get_residue_model_scores(model, data)` repeats this for every non-global block.

Lower cosine similarity ⇒ larger representation shift under masking ⇒ more important residue.
