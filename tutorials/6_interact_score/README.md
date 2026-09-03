# ATOMICAScore: ranking interface residues by how much the ligand depends on them

ATOMICAScore masks one interface residue at a time and measures how far that moves the pretrained
model's representation of the **ligand**. For block `i` of an interaction graph `G`, build `G \ i` by
replacing that block with the mask block and its atoms with a single mask atom, then

```
a_i = cosine( r(G), r(G \ i) )
```

A **low** `a_i` means masking the residue changed the readout a lot, so that residue matters more.

The readout `r` is the component-normalized mean of `z_block` over the ligand's blocks. It comes from
[`representations.py`](../../src/atomica/representations.py), the one place a representation is defined:

```python
from atomica import representations as R

R.get(model, batch, "z_interface", pool="mean_component_normalized", segment=ligand_segment)
```

## Requirements

- The `atomica` environment, from the top-level [README](../../README.md).
- The pretrained ATOMICA checkpoint. The notebook downloads it if it is missing. No other checkpoint
  is needed.
- A GPU is faster but not required; the example complexes score on CPU in about a minute.

## Run it

```bash
jupyter notebook example_run_interact_score.ipynb
```

Paths resolve relative to the repository root, so the notebook runs from any checkout.

## Results

For `6llw_A_A_UDP`, a glycosyltransferase bound to UDP with 29 interface residues, the top of the
ranking against PLIP annotations:

```
 rank  residue  type   score     annotated
    1   A_340   TRP    0.99744   yes (pi-stacking)
    2   A_279   ASN    0.99794   yes (hydrogen bond)
    3   A_343   GLN    0.99831   yes (hydrogen bond)
    4   A_362   ASN    0.99831   yes (hydrogen bond)
    5   A_278   GLY    0.99832   no
    6   A_280   ARG    0.99851   yes (hydrogen bond)
    7   A_358   HIS    0.99908   no
    8   A_363   SER    0.99909   yes (hydrogen bond)
    9   A_341   VAL    0.99925   yes (hydrogen bond)
   10   A_366   GLU    0.99928   yes (hydrogen bond)
```

precision@10 is 0.800 and AUROC is 0.958 on this complex. All eight annotated residues are in the top
ten, so 0.800 is the most attainable here. `precision_at_k` returns a fraction of k.

## Files

| file | what it is |
|---|---|
| `example_run_interact_score.ipynb` | the tutorial |
| `make_plip_labels.py` | regenerates the annotations; needs PLIP, which is not an ATOMICA dependency |

The notebook reads three files, all small: `data/example/example_inputs.csv`,
`data/example/example_processed_data.parquet` and `data/example/example_plip_labels.csv`. It also needs the
`.cif` structures in `data/example/example_data/` if you rebuild the processed file yourself.
`make_plip_labels.py` downloads whatever structures it needs from RCSB, so no extra files are required.

No checkpoint beyond the pretrained ATOMICA model is used.

## Your own structures

```bash
python -m atomica.data.process_pdbs \
    --data_index_file my_inputs.csv --out_path my_processed.parquet \
    --interface_dist_th 8.0 --fragmentation_method PS_300

python -m atomica.interaction_profiler.interact_score \
    --data_path my_processed.parquet --output_path my_scores.jsonl \
    --model_config  checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_config.json \
    --model_weights checkpoints/ATOMICA_checkpoints/pretrain/pretrain_model_weights.pt
```

Each output line holds the complex id, the scored block indices, the scores, the ligand segment, the
batch size and the readout. Complexes with no amino-acid residue block, or an ambiguous ligand side, are
skipped and counted. Re-running appends only what is missing.

## API

From [`interact_score.py`](../../src/atomica/interaction_profiler/interact_score.py):

| function | what it does |
|---|---|
| `atomica_score(model, data)` | the score for every amino-acid residue block, at a fixed batch size |
| `find_ligand_segment(data)` | which segment holds the ligand, inferred from block types |
| `scorable_blocks(data, ligand_segment)` | the blocks that get masked |
| `mask_block(data, block_idx)` | one block replaced by the mask block and a single mask atom |
| `precision_at_k(importance, labels, k)` | fraction of the top k that are annotated |
| `auroc(importance, labels)` | rank-based, NaN when one class is absent |

`atomica_score` returns `block_idx`, `score` (the cosine, low means important), `importance` (the sign
flipped, which is what the metrics take), `ranking()` and `batch_size`.
