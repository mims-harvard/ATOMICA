# Tutorial 1 — Get representations out of ATOMICA

ATOMICA builds representatiosns for intermolecular interfaces. This tutorial shows how to get embeddings from the pretrained checkpoint.

```bash
python -m atomica.representations --guidance
``` s

## Contents

1. [Setup](#setup)
2. [Step 1: process your structures](#step-1-process-your-structures)
3. [Step 2: pick a representation](#step-2-pick-a-representation)
4. [Step 3: extract it](#step-3-extract-it)
5. [What comes out](#what-comes-out)
6. [From Python](#from-python)
7. [Other kinds of input](#other-kinds-of-input)
8. [Troubleshooting](#troubleshooting)

## Setup

An H100 or A100 GPU is recommended. ATOMICA also runs on CPU, which is slower but fine for small
inputs and for checking an installation. Add `--device cpu` to any command below. Any CUDA build
of PyTorch from 11.8 through 13.0 works; see [setup/README.md](../../setup/README.md).

Activate the environment you built during setup:

```bash
mamba activate atomica-env       # conda or mamba
source atomica-env/bin/activate  # or a virtual environment
```

Download the pretrained checkpoint from Hugging Face. Install the CLI first with
`pip install -U "huggingface_hub[cli]"` if you do not have it:

```bash
hf download ada-f/ATOMICA --repo-type model --local-dir checkpoints \
  --include "ATOMICA_checkpoints/pretrain/**"
```

## Step 1: process your structures

ATOMICA reads interaction interfaces, not whole PDB files. This step finds the interface between
the two chains or the chain and the ligand you name, and writes it in the model's block-and-atom
format. `data/example/example_inputs.csv` shows the input columns.

```bash
python -m atomica.data.process_pdbs \
  --data_index_file data/example/example_inputs.csv \
  --out_path data/example/example_processed_data.parquet
```

Each processed structure has two segments: **segment 0 is the receptor** (the first chain you
named) and **segment 1 is the ligand or partner chain**. You need this when you ask for an
interface-level representation.

## Step 2: pick a representation

Two questions decide it.

**Will you train anything on top of the vector?** If yes, use the **`z` family**. If no, and you
are going to compare vectors with a cosine or a distance, use `z` with the frozen-comparison
pooling, or the `h` family for single blocks and for visualization.

**What are you describing: the whole complex, one molecule in it, one residue, or one atom?** That
picks the level: `graph`, `interface`, `block`, `atom`.

The `h` family is the 32-number readout the model uses internally. It keeps only the rotation-order
zero channels and drops every higher-order channel the encoder computed. The `z` family converts
those channels into rotation invariants that an ordinary MLP can read, which is why it is much
wider and why every frozen benchmark in the paper trains on it.

| name | one vector per | width | reach for it when |
|---|---|---|---|
| `h_atom` | atom | 32 | you want per-atom chemistry and the scalar channels are enough |
| `h_block` | residue or fragment | 32 | comparing single residues or fragments with a cosine |
| `h_graph` | complex | 32 | a small vector per complex to visualize or browse |
| `h_interface` | molecule in the complex | 32 | comparing one molecule's side against another's, across modalities |
| `z_atom` | atom | 608 | rarely alone; it is a component of `z_block` |
| `z_block` | residue or fragment | 1792 | training a per-residue head, or probing one block's environment |
| `z_graph` | complex | 5376 or 1792 | training any head on a whole complex or pocket, or retrieval |
| `z_interface` | molecule in the complex | 5376 or 1792 | the same, restricted to one molecule's blocks |

Widths are for the released pretrain checkpoint, where the node dimension is 32. A differently
sized checkpoint reports its own widths; `python -m atomica.representations --describe` with your
config and weights prints them.

`z_graph` and `z_interface` have two widths because they have two pooling rules, and the rule is
fixed by how you will use the vector:

- `mean_std_global` (5376 wide) concatenates the mean over blocks, the standard deviation over
  blocks, and the global block node. Use it **whenever a head will be trained on top**. It is
  parameter-free, so the head stays the only fitted part of the pipeline.
- `mean_component_normalized` (1792 wide) takes the mean over blocks and normalizes each of the
  three parts of `z_block` to unit length before concatenating. Use it **when you compare frozen
  vectors directly**. The three parts differ by about an order of magnitude in norm, and a cosine
  weights each part by the product of its norms, so without this one part decides the answer.

Asking for `z_graph` without naming a rule is an error rather than a silent default. The two are
not comparable, and a benchmark that does not say which one it used cannot be reproduced.

### What the paper uses

| analysis | representation | pooling |
|---|---|---|
| RNA-Protein, RNA-Site residue-level tasks (Fig. 3) | `z_block` | — |
| RNA-GO, RNA-Ligand, MaSIF-ligand (Fig. 3) | `z_graph` | `mean_std_global` |
| ATP versus ADP pocket discrimination (Fig. 3i) | `z_graph` | `mean_std_global` |
| Same-ligand pocket retrieval (Fig. 3j, Table S2) | `z_graph` | `mean_component_normalized` |
| Metal coordination number and geometry probes (Table S1) | `z_block` | — |
| ATOMICAScore over interface blocks (Fig. 2) | `z_interface`, ligand segment | `mean_component_normalized` |
| UMAP of 2,105,459 complexes (Fig. 2) | `h_graph` | — |
| PCA of mean embedding per element and per block type (Fig. 2) | `h_atom`, `h_block` | — |
| Inhibitor blocks against peptide blocks (Fig. 4a,d) | `h_block` | — |
| Inhibitor against protein B surface patches (Fig. 4e,f) | `h_interface` | — |

## Step 3: extract it

The command is the same shape every time. Name the representations, and add the pooling rule and
the segment when the representation needs one.

**Training a head on whole complexes or pockets.** The most common case, and the one every
frozen benchmark in the paper uses:

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_z_graph.parquet \
  --representations z_graph --pool mean_std_global
```

**Comparing frozen complexes with a cosine**, for retrieval or nearest-neighbour search:

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_retrieval.parquet \
  --representations z_graph --pool mean_component_normalized
```

**Training a per-residue head.** `--drop_global_block` removes the one artificial block per
segment that stands for the whole molecule; it is not a residue and has no label:

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_z_block.parquet \
  --representations z_block --drop_global_block
```

**One molecule's side of the complex.** `--segment 1` pools the ligand or partner chain, after
message passing has run over the whole complex, so the vector still knows what it was bound to:

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_ligand_side.parquet \
  --representations h_interface,z_interface \
  --pool mean_component_normalized --segment 1
```

**Several at once.** Ask for as many names as you want in one run. The `h` family costs one
forward pass and the `z` family costs one more, however many names you list:

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_all.pkl \
  --representations h_atom,h_block,h_graph,z_block,z_graph --pool mean_std_global
```

If the package is installed, `atomica-representations` is the same command with less typing.

## What comes out

One row per input structure, written to `.parquet` or `.pkl` by the extension of
`--output_path`. Columns:

| column | contents |
|---|---|
| `id` | the structure id from the processed file |
| `block_id` | block type index per block, in the same order as the block-level rows |
| `atom_id` | atom type index per atom, in the same order as the atom-level rows |
| one column per representation | named after the representation, for example `z_graph` |

Atom-level columns hold one row per atom, block-level columns one row per block, and graph- and
interface-level columns hold a single vector. Parquet stores these as nested lists; pickle keeps
numpy arrays.

## From Python

The library API is smaller than the command line, and it is what the other tutorials call:

```python
from atomica import representations as R

model, dataset_class = R.load_model(config_path, weights_path)
model.eval()
dataset = dataset_class("data/example/example_processed_data.parquet")

rows = R.embed_dataset(model, dataset, ["z_graph"], pool="mean_std_global")
print(rows[0]["id"], rows[0]["z_graph"].shape)
```

For a single collated batch you already have, ask for one representation at a time with
`R.get(model, batch, "z_graph", pool="mean_std_global")`, or several in as few forward passes as
possible with `R.get_many(model, batch, ["h_block", "z_graph"], pool="mean_std_global")`.

Two printable references, useful when you come back to a saved file months later:

```python
R.guidance()        # which representation to use when
R.describe(model)   # the table above, with this checkpoint's widths
```

A runnable end-to-end example is in this directory. It takes about a minute on CPU:

```bash
python tutorials/1_get_embeddings/example_representations.py
```

It ends by ranking the six other example complexes against `6llw_A_A_UDP` twice, once by `z_graph`
and once by `h_graph`. The two rankings disagree, which is the point: `z_graph` puts `5kl2_A_BC`
first at cosine 0.882, `h_graph` puts `4yaz_A_A_4BW` first at 0.739. They are different
quantities, so never mix them in one comparison.

## Other kinds of input

**Any biomolecular complex.** Build an index file like `data/example/example_inputs.csv`, run
`atomica.data.process_pdbs`, then extract as above.

**Predicted binding sites.** To embed protein interfaces with an ion, small molecule, lipid,
nucleic acid or protein partner you do not have a structure for, predict the binding site with
[PeSTo](https://github.com/LBM-EPFL/PeSTo), convert its output with
`python -m atomica.data.process_PeSTo_results`, then extract as above.

**Fine-tuned checkpoints.** `--model_config` and `--model_weights` accept any ATOMICA checkpoint.
A `ProteinInterfaceModel` checkpoint is unwrapped automatically to the encoder inside it.

## Troubleshooting

**`--pool is required for z_graph and z_interface`.** Deliberate. Pick `mean_std_global` if a head
will be trained on the vectors, `mean_component_normalized` if you will compare them directly.

**`--segment is required for h_interface and z_interface`.** Segment 0 is the receptor and segment
1 is the ligand or partner chain, as written by `atomica.data.process_pdbs`.

**CUDA out of memory.** Set `--atom_budget` to cap the atoms per batch, or lower `--batch_size`.
Otherwise the extractor retries the batch one structure at a time and skips any that still fails;
`--strict` raises instead.

**`--device cuda` fails with `torch.cuda.is_available()` is False.** The installed PyTorch build
does not match the driver. See [setup/README.md](../../setup/README.md), or use `--device cpu`.

**A saved file whose provenance you have lost.** Each column is named after the representation it
holds, except that `z_graph` and `z_interface` do not record their pooling rule; on a
32-dimensional checkpoint a width of 5376 means `mean_std_global` and 1792 means
`mean_component_normalized`.

**Comparing vectors across runs.** Use one checkpoint and one pooling rule throughout. Batches are
formed from structures that share a largest block, so `--batch_size` does not change the vectors;
`--no_group_by_max_block` opts out of that.
