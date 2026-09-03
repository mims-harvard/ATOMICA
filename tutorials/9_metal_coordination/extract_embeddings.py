"""Step 1: run the frozen ATOMICA encoder over the metal pockets and save the ion block's z_block.

Every pocket has two segments: segment 0 is the protein residues, segment 1 is a global node plus
exactly one metal block. The row saved per pocket is the metal's own block.

    python extract_embeddings.py
    python extract_embeddings.py --splits test --device cpu
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from atomica import representations as R
from atomica.data.dataset import PDBDataset
from atomica.data.pdb_utils import VOCAB
from atomica.models.prediction_model import PredictionModel

import metal_tasks as T


def load_frozen_encoder(device: str):
    """The pretrained encoder stack, in eval mode with gradients off."""
    config = os.path.join(T.CHECKPOINT, "pretrain_model_config.json")
    weights = os.path.join(T.CHECKPOINT, "pretrain_model_weights.pt")
    for path in (config, weights):
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}; see checkpoints/README.md")
    model = PredictionModel.load_from_config_and_weights(config, weights).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def ion_block_rows(model, batch: dict) -> torch.Tensor:
    """Row index of each graph's metal block within the batch's concatenated blocks."""
    is_ion = (batch["segment_ids"] == 1) & (batch["B"] != model.global_block_id)
    rows = torch.nonzero(is_ion, as_tuple=False).flatten()
    assert len(rows) == int(batch["lengths"].shape[0]), "expected one ion block per graph"
    return rows


@torch.no_grad()
def extract_split(model, split: str, batch_size: int, device: str):
    """One pass over a split. Returns (meta, {component: array}).

    Batching goes through `atomica.representations.group_batches`, which puts only structures that
    share a largest block into one batch. The cross-attention pad width is the largest block in the
    batch, so within such a group it equals what a batch of one would give, and the vectors do not
    depend on which other pockets shared a batch.
    """
    items = [it for it in PDBDataset(T.POCKETS).data if it["split"] == split]
    assert items, f"no pockets for split {split!r}"
    component_dims = model.invariant_component_dims()      # h_block / gram / atom, in that order
    offsets, start = {}, 0
    for name, width in component_dims.items():
        offsets[name] = (start, start + int(width))
        start += int(width)

    batches = R.group_batches(items, batch_size)
    z_of_row: list = [None] * len(items)
    symbols: list = [None] * len(items)

    for positions, batch_items in tqdm(batches, desc=f"  {split}", leave=False):
        batch = PDBDataset.collate_fn([it["data"] for it in batch_items])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rv = model.infer(batch, return_invariant_repr=True, invariant_pool=None)
        z = R.get(model, batch, "z_block", return_value=rv)

        rows = ion_block_rows(model, batch)
        z = z[rows].float().cpu().numpy()
        block_symbols = [VOCAB.idx_to_symbol(int(b)) for b in batch["B"][rows].cpu()]
        for k, position in enumerate(positions):
            z_of_row[position] = z[k]
            symbols[position] = block_symbols[k]
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    stacked = np.stack(z_of_row)
    meta = pd.DataFrame({"id": [it["id"] for it in items], "block_symbol": symbols})
    arrays = {name: stacked[:, lo:hi].astype(np.float32) for name, (lo, hi) in offsets.items()}
    assert meta["id"].is_unique and "&" not in set(symbols), "unexpected block selected as the ion"
    print(f"  {split}: {len(meta):,} metal sites, metals={sorted(set(symbols))}")
    return meta, arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(T.SPLITS), choices=list(T.SPLITS))
    ap.add_argument("--batch-size", type=int, default=T.EXTRACT_BATCH_SIZE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(T.EMBEDDINGS, exist_ok=True)
    model = load_frozen_encoder(args.device)
    dims = model.invariant_component_dims()
    print(f"\ndevice={args.device}  batch_size={args.batch_size}\n"
          f"z_block = {' + '.join(f'{k} {v}' for k, v in dims.items())} = {sum(dims.values())}")

    counts = {}
    for split in args.splits:
        meta, arrays = extract_split(model, split, args.batch_size, args.device)
        np.savez(os.path.join(T.EMBEDDINGS, f"{split}_z_block.npz"), **arrays)
        meta.to_parquet(os.path.join(T.EMBEDDINGS, f"{split}_meta.parquet"), index=False)
        counts[split] = int(len(meta))

    with open(os.path.join(T.EMBEDDINGS, "extraction.json"), "w") as fh:
        json.dump({"sites": counts, "batch_size": args.batch_size,
                   "component_dims": {k: int(v) for k, v in dims.items()}}, fh, indent=1)
    print(f"  saved -> {os.path.relpath(T.EMBEDDINGS, T.HERE)}")


if __name__ == "__main__":
    main()
