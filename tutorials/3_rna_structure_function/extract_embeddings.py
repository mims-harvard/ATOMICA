"""Run the frozen ATOMICA encoder over the RNAglib structures and save z_block per block.

z_block is 1792 wide: the 32-d scalar readout, 544 Gram entries, and 1216 atom mean and std. It is
saved per block so residue-level tasks use the rows directly and graph-level tasks pool at probe
time.

  python extract_embeddings.py --all                  # published checkpoint per task
  python extract_embeddings.py --all --batch-size 16  # same vectors, fewer forward passes
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from atomica import representations as R
from atomica.models.prediction_model import PredictionModel

import rna_tasks as T


def load_frozen_encoder(backbone: str, device: str):
    """The pretrained encoder with the denoising heads removed, in eval mode, no gradients."""
    ckpt = T.BACKBONES[backbone]
    config, weights = os.path.join(ckpt, "config.json"), os.path.join(ckpt, "weights.pt")
    for path in (config, weights):
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}; see the README section on checkpoints")
    model = PredictionModel.load_from_config_and_weights(config, weights).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters()), "encoder is not frozen"
    return model


def extract_split(model, task, split, device, batch_size, atom_budget,
                  group_by_max_block, strict):
    residue_level = T.TASKS[task]["residue_level"]
    dataset = T.load_dataset(task, split)
    label_map = T.residue_label_map(task, split) if residue_level else {}

    dims = model.invariant_component_dims()          # h_block / gram / atom, in that order
    offsets, start = {}, 0
    for name, width in dims.items():
        offsets[name] = (start, start + int(width))
        start += int(width)

    # The global block node is kept, so a block's row index is its block index.
    embedded = R.embed_dataset(model, dataset, ["z_block"], batch_size=batch_size, device=device,
                               group_by_max_block=group_by_max_block, atom_budget=atom_budget,
                               drop_global_block=False, strict=strict, progress=True)
    if len(embedded) != len(dataset.data):
        raise RuntimeError(f"{task}/{split}: {len(embedded)} rows for {len(dataset.data)} "
                           f"structures; re-run with --strict")

    rows = {k: [] for k in ("id", "graph_id", "block_idx", "is_global", "pdb_index", "label")}
    feats = {name: [] for name in dims}

    for out, item in zip(embedded, dataset.data):
        graph_id = str(item["id"])
        if str(out["id"]) != graph_id:
            raise RuntimeError(f"{task}/{split}: rows are out of order at {graph_id!r}")
        z = np.asarray(out["z_block"], dtype=np.float32)
        block_ids = np.asarray(out["block_id"])
        block_to_pdb = item.get("block_to_pdb_indexes", {}) or {}
        labelled = sorted(block_to_pdb)

        if residue_level:
            # One row per labelled residue, with the label looked up by residue index.
            selected = [(b, block_to_pdb[b], label_map[f"{graph_id}_{block_to_pdb[b]}"])
                        for b in labelled]
        else:
            # Every block is a feature row; the label belongs to the graph.
            selected = [(b, block_to_pdb.get(b), float("nan")) for b in range(z.shape[0])]

        for b, pdb_index, label in selected:
            rows["id"].append(f"{graph_id}_{pdb_index}" if residue_level else graph_id)
            rows["graph_id"].append(graph_id)
            rows["block_idx"].append(int(b))
            rows["is_global"].append(bool(block_ids[b] == model.global_block_id))
            rows["pdb_index"].append(pdb_index)
            rows["label"].append(label)
            for name, (lo, hi) in offsets.items():
                feats[name].append(z[b, lo:hi])

    meta = pd.DataFrame(rows)
    arrays = {k: np.stack(v).astype(np.float32) for k, v in feats.items()}
    for name, array in arrays.items():
        assert array.shape[0] == len(meta), f"{name}: {array.shape[0]} rows, meta has {len(meta)}"
    if residue_level:
        assert meta["id"].nunique() == len(meta), "residue point ids are not unique"
        print(f"  {task}/{split}: {len(meta)} residues in {meta['graph_id'].nunique()} structures")
    else:
        print(f"  {task}/{split}: {meta['graph_id'].nunique()} graphs, {len(meta)} blocks")
    return meta, arrays


def run(task, backbone, splits, device, batch_size, atom_budget, group_by_max_block, strict):
    out_dir = T.embedding_dir(task, backbone)
    os.makedirs(out_dir, exist_ok=True)
    model = load_frozen_encoder(backbone, device)
    dims = model.invariant_component_dims()
    print(f"\n{task}  backbone={backbone}  batch_size={batch_size}"
          f"{'' if group_by_max_block else '  (file-order batching: not comparable)'}\n"
          f"z_block = {' + '.join(f'{k} {v}' for k, v in dims.items())} = {sum(dims.values())}")

    counts = {}
    for split in splits:
        meta, arrays = extract_split(model, task, split, device, batch_size, atom_budget,
                                     group_by_max_block, strict)
        np.savez(os.path.join(out_dir, f"{task}_{split}_z_block.npz"), **arrays)
        meta.to_parquet(os.path.join(out_dir, f"{task}_{split}_meta.parquet"), index=False)
        counts[split] = int(len(meta))
    with open(os.path.join(out_dir, "extraction.json"), "w") as fh:
        json.dump({"task": task, "backbone": backbone, "rows": counts,
                   "component_dims": {k: int(v) for k, v in dims.items()},
                   "batch_size": batch_size, "group_by_max_block": group_by_max_block,
                   "atom_budget": atom_budget, "strict": strict}, fh, indent=1)
    print(f"  saved -> {os.path.relpath(out_dir, T.HERE)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(T.TASKS))
    ap.add_argument("--all", action="store_true", help="all four at their published checkpoint")
    ap.add_argument("--backbone", choices=list(T.BACKBONES), default=None)
    ap.add_argument("--splits", nargs="+", default=list(T.SPLITS), choices=list(T.SPLITS))
    ap.add_argument("--batch-size", type=int, default=T.EXTRACT_BATCH_SIZE,
                    help="structures per forward pass; a speed setting, not a numerical one")
    ap.add_argument("--atom-budget", type=int, default=None,
                    help="cap atoms per batch if a batch runs out of memory")
    ap.add_argument("--no-group-by-max-block", dest="group_by_max_block", action="store_false",
                    help="batch in file order; vectors then depend on batch size and item order")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="skip structures that run out of memory instead of raising")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tasks = list(T.TASKS) if args.all else ([args.task] if args.task else [])
    if not tasks:
        ap.error("pass --task NAME or --all")
    for task in tasks:
        backbone = args.backbone or T.PUBLISHED_BACKBONE[task]
        if backbone != T.PUBLISHED_BACKBONE[task]:
            print(f"[note] {task} is published on {T.PUBLISHED_BACKBONE[task]}")
        run(task, backbone, args.splits, args.device, args.batch_size, args.atom_budget,
            args.group_by_max_block, args.strict)


if __name__ == "__main__":
    main()
