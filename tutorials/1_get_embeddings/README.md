# Tutorial 1: Extracting representations from ATOMICA

ATOMICA represents an intermolecular interface at the atom, block, interface and graph levels, in
two families. This tutorial covers producing those representations from the pretrained checkpoint
and choosing among them, and `python -m atomica.representations --guidance` prints the same choice
in the terminal.

## Contents

1. [Setup](#setup)
2. [Step 1: process your structures](#step-1-process-your-structures)
3. [Step 2: pick a representation](#step-2-pick-a-representation)
4. [Step 3: extract representations](#step-3-extract-representations)
5. [Output](#output)
6. [From Python](#from-python)
7. [Other kinds of input](#other-kinds-of-input)
8. [Troubleshooting](#troubleshooting)

## Setup

An H100 or A100 GPU is recommended, although ATOMICA also runs on CPU, which is slower but
sufficient for small inputs and for checking an installation. Add `--device cpu` to any command
below. Any CUDA build of PyTorch from 11.8 through 13.0 works, as described in
[setup/README.md](../../setup/README.md).

Activate the environment built during setup.

```bash
mamba activate atomica-env       # conda or mamba
source atomica-env/bin/activate  # or a virtual environment
```

Download the pretrained checkpoint from Hugging Face, installing the CLI with
`pip install -U "huggingface_hub[cli]"` if it is not already available.

```bash
hf download ada-f/ATOMICA --repo-type model --local-dir checkpoints \
  --include "ATOMICA_checkpoints/pretrain/**"
```

## Step 1: process your structures

ATOMICA reads interaction interfaces rather than whole PDB files, so this step locates the
interface between two named chains, or between a chain and a ligand, and writes it in the
block-and-atom format the model expects. The input columns are shown in
`data/example/example_inputs.csv`.

```bash
python -m atomica.data.process_pdbs \
  --data_index_file data/example/example_inputs.csv \
  --out_path data/example/example_processed_data.parquet
```

Each processed structure carries two segments, where segment 0 is the receptor named first and
segment 1 is the ligand or partner chain. Interface-level representations need that distinction.

## Step 2: pick a representation

Two questions decide the choice. The first is whether a head will be trained on the vectors, in
which case use the `z` family, and if instead the vectors are compared directly with a cosine or a
distance, use `z` with the frozen-comparison pooling or the `h` family for single blocks and for
visualization. The second question is what the vector describes, whether a whole complex, one
molecule within it, one residue, or one atom, which selects the graph, interface, block or atom
level.

The `h` family is the 32-number readout the model uses internally, and it keeps the rotation-order
zero channels while dropping the higher-order channels the encoder computes. The `z` family
converts those channels into rotation invariants that an ordinary MLP can read, which makes it
much wider and is why the frozen benchmarks in the paper train on it.

| name | one vector per | width | use it when |
|---|---|---|---|
| `h_atom` | atom | 32 | describing per-atom chemistry, where the scalar channels are enough |
| `h_block` | residue or fragment | 32 | comparing single residues or fragments with a cosine |
| `h_graph` | complex | 32 | a small vector per complex, for visualization or nearest-neighbour browsing |
| `h_interface` | molecule in the complex | 32 | comparing one molecule's side against another's, across modalities |
| `z_atom` | atom | 608 | rarely alone, since it is a component of `z_block` |
| `z_block` | residue or fragment | 1792 | training a per-residue head, or probing what one block's environment encodes |
| `z_graph` | complex | 5376 or 1792 | training a head on a whole complex or pocket, or retrieval |
| `z_interface` | molecule in the complex | 5376 or 1792 | the same, restricted to the blocks of one molecule |

Widths are given for the released pretrain checkpoint, whose node dimension is 32. A checkpoint of
another size reports its own widths, which `python -m atomica.representations --describe` prints
when given a config and weights.

`z_graph` and `z_interface` have two widths because they have two pooling rules, and the rule
follows from how the vector is used.

- `mean_std_global`, 5376 wide, concatenates the mean over blocks, the standard deviation over
  blocks, and the global block node. Use it whenever a head is trained on top, since it is
  parameter-free and leaves the head as the only fitted part of the pipeline.
- `mean_component_normalized`, 1792 wide, takes the mean over blocks and normalizes each of the
  three parts of `z_block` to unit length before concatenating. Use it when comparing frozen
  vectors, because the three parts differ by about an order of magnitude in norm and a cosine
  weights each part by the product of its norms, so one part would otherwise decide the answer.

Asking for `z_graph` without naming a rule raises an error rather than selecting a default,
because the two are not comparable and a result that does not record the rule cannot be
reproduced.

### Representations used in the paper

| analysis | representation | pooling |
|---|---|---|
| RNA-Protein and RNA-Site residue-level tasks (Fig. 3) | `z_block` | none |
| RNA-GO, RNA-Ligand and MaSIF-ligand (Fig. 3) | `z_graph` | `mean_std_global` |
| ATP versus ADP pocket discrimination (Fig. 3i) | `z_graph` | `mean_std_global` |
| Same-ligand pocket retrieval (Fig. 3j, Table S2) | `z_graph` | `mean_component_normalized` |
| Metal coordination number and geometry probes (Table S1) | `z_block` | none |
| ATOMICAScore over interface blocks (Fig. 2) | `z_interface`, ligand segment | `mean_component_normalized` |
| UMAP of 2,105,459 complexes (Fig. 2) | `h_graph` | none |
| PCA of the mean embedding per element and per block type (Fig. 2) | `h_atom`, `h_block` | none |
| Inhibitor blocks against peptide blocks (Fig. 4a,d) | `h_block` | none |
| Inhibitor against protein B surface patches (Fig. 4e,f) | `h_interface` | none |

## Step 3: extract representations

The command has the same shape every time. Name the representations, and add the pooling rule and
the segment when the representation requires them.

Training a head on whole complexes or pockets is the most common case, and the one the frozen
benchmarks in the paper use.

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_z_graph.parquet \
  --representations z_graph --pool mean_std_global
```

Retrieval and nearest-neighbour search compare frozen vectors, which changes the pooling rule.

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_retrieval.parquet \
  --representations z_graph --pool mean_component_normalized
```

A per-residue head uses the block level, where `--drop_global_block` removes the artificial block
that stands for the whole molecule and carries no label.

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_z_block.parquet \
  --representations z_block --drop_global_block
```

To describe one molecule's side of a complex, `--segment 1` pools the ligand or partner chain
after message passing has run over the whole complex, so the vector retains what that molecule was
bound to.

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_ligand_side.parquet \
  --representations h_interface,z_interface \
  --pool mean_component_normalized --segment 1
```

Several representations can come from a single run, where the `h` family costs one forward pass
and the `z` family costs a second, however many names are listed.

```bash
python -m atomica.representations \
  --model_config checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
  --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt \
  --data_path data/example/example_processed_data.parquet \
  --output_path data/example/example_all.pkl \
  --representations h_atom,h_block,h_graph,z_block,z_graph --pool mean_std_global
```

With the package installed, `atomica-representations` runs the same command.

## Output

The extractor writes one row per input structure, in `.parquet` or `.pkl` according to the
extension of `--output_path`.

| column | contents |
|---|---|
| `id` | the structure id from the processed file |
| `block_id` | block type index per block, in the same order as the block-level rows |
| `atom_id` | atom type index per atom, in the same order as the atom-level rows |
| one column per representation | named after the representation, for example `z_graph` |

Atom-level columns hold one row per atom, block-level columns one row per block, and graph- and
interface-level columns hold a single vector. Parquet stores these as nested lists, and pickle
keeps numpy arrays.

## From Python

The library API is smaller than the command line, and it is what the other tutorials call.

```python
from atomica import representations as R

model, dataset_class = R.load_model(config_path, weights_path)
model.eval()
dataset = dataset_class("data/example/example_processed_data.parquet")

rows = R.embed_dataset(model, dataset, ["z_graph"], pool="mean_std_global")
print(rows[0]["id"], rows[0]["z_graph"].shape)
```

For a batch that is already collated, `R.get(model, batch, "z_graph", pool="mean_std_global")`
returns one representation, and `R.get_many(model, batch, ["h_block", "z_graph"],
pool="mean_std_global")` returns several from as few forward passes as possible.

Two printable references remain available when returning to a saved file later.

```python
R.guidance()        # which representation to use when
R.describe(model)   # the table above, with this checkpoint's widths
```

A runnable example in this directory takes about a minute on CPU.

```bash
python tutorials/1_get_embeddings/example_representations.py
```

It ranks the six other example complexes against `6llw_A_A_UDP` twice, once by `z_graph` and once
by `h_graph`. The two rankings differ, with `z_graph` placing `5kl2_A_BC` first at cosine 0.882
and `h_graph` placing `4yaz_A_A_4BW` first at 0.739, so the two quantities should not be mixed in
one comparison.

## Other kinds of input

Any biomolecular complex follows the same path, where an index file like
`data/example/example_inputs.csv` is processed with `atomica.data.process_pdbs` and then
extracted as above.

For a protein interface whose partner is not present in the structure, predict the binding site
with [PeSTo](https://github.com/LBM-EPFL/PeSTo), convert its output with
`python -m atomica.data.process_PeSTo_results`, and extract as above. This covers ion, small
molecule, lipid, nucleic acid and protein partners.

`--model_config` and `--model_weights` accept any ATOMICA checkpoint, and a
`ProteinInterfaceModel` checkpoint is unwrapped to the encoder inside it.

## Troubleshooting

`--pool` and `--segment` are required rather than defaulted. Use `mean_std_global` when a head
will be trained on the vectors and `mean_component_normalized` when they are compared directly,
and give segment 0 for the receptor or segment 1 for the ligand or partner chain, following the
order written by `atomica.data.process_pdbs`.

When CUDA runs out of memory, set `--atom_budget` to cap the atoms in one batch or lower
`--batch_size`. The extractor otherwise retries a failing batch one structure at a time and skips
any structure that still fails, and `--strict` raises the error instead.

If `--device cuda` reports that `torch.cuda.is_available()` is False, the installed PyTorch build
does not match the driver, which [setup/README.md](../../setup/README.md) covers, and the
problem is avoided with `--device cpu`.

A saved file records each representation in its column name, although `z_graph` and `z_interface`
do not record their pooling rule. On a 32-dimensional checkpoint, a width of 5376 indicates
`mean_std_global` and a width of 1792 indicates `mean_component_normalized`.

Comparisons across runs use one checkpoint and one pooling rule throughout. Batches are formed
from structures that share a largest block, so `--batch_size` does not change the vectors, and
`--no_group_by_max_block` opts out of that.
