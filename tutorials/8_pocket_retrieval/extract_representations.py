"""Step 1: one frozen ATOMICA vector per ligand-free pocket.

    R.get(model, batch, "z_graph", pool="mean_component_normalized")

Both halves matter. z_graph keeps the higher-degree channels that h_graph discards, as rotation
invariants a cosine can read. mean_component_normalized scales the three parts of z_block to unit
length before concatenating; without it the atom-pooled part takes about 99% of every cosine,
because a cosine over concatenated vectors weights each part by the product of its norms. Use
mean_std_global instead only when a head will be trained on top and can learn that scaling itself.

Usage:
    python extract_representations.py
    python extract_representations.py --limit 20 --device cpu
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from atomica import representations as R
from atomica.data.dataset import PDBDataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPRESENTATIONS = os.path.join(HERE, "representations")
DEFAULT_CHECKPOINT = os.path.join(
    HERE, "..", "..", "checkpoints", "ATOMICA_checkpoints", "pretrain")


def load_frozen_encoder(checkpoint_dir: str, device: str):
    """The released pretrained encoder, in eval mode with gradients off."""
    config = os.path.join(checkpoint_dir, "pretrain_model_config.json")
    weights = os.path.join(checkpoint_dir, "pretrain_model_weights.pt")
    for path in (config, weights):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}\nDownload it with:\n"
                f"  hf download ada-f/ATOMICA --repo-type model --local-dir checkpoints "
                f"--include 'ATOMICA_checkpoints/pretrain/**'")
    model, _ = R.load_model(config, weights)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def embed(model, dataset, batch_size: int, device: str, atom_budget=None):
    """(ids, matrix) with one row per pocket, in dataset order.

    embed_dataset skips a structure that fails rather than raising, which would silently shrink
    the pool and reweight every cluster it belonged to, so the row count is checked here.
    """
    rows = R.embed_dataset(model, dataset, ["z_graph"], pool="mean_component_normalized",
                           batch_size=batch_size, device=device, atom_budget=atom_budget)
    if len(rows) != len(dataset.data):
        raise SystemExit(
            f"got {len(rows)} rows for {len(dataset.data)} pockets: "
            f"{len(dataset.data) - len(rows)} were skipped. Lower --batch_size or --atom_budget "
            f"and re-extract the whole pool.")
    ids = [item["structure"] for item in dataset.data]
    mismatched = [(row["id"], name) for row, name in zip(rows, ids)
                  if str(row["id"]).split("_")[0] != name]
    if mismatched:
        raise SystemExit(f"row order does not match the dataset: {mismatched[:3]}")
    return ids, np.stack([np.asarray(row["z_graph"]) for row in rows], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pockets", default=os.path.join(DATA, "pockets.parquet"))
    parser.add_argument("--out", default=os.path.join(REPRESENTATIONS, "atomica_z_graph_cn.npz"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--atom_budget", type=int, default=None,
                        help="cap the atoms per batch on a small GPU")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device

    model = load_frozen_encoder(args.checkpoint, device)
    dataset = PDBDataset(args.pockets)
    if args.limit:
        dataset.data = dataset.data[:args.limit]

    dims = model.invariant_component_dims()
    print(f"device {device}, {len(dataset.data)} pockets, batch size {args.batch_size}")
    print(f"z_graph, mean_component_normalized -> {sum(dims.values())} dimensions "
          f"({' | '.join(f'{k} {v}' for k, v in dims.items())})\n")

    ids, matrix = embed(model, dataset, args.batch_size, device, atom_budget=args.atom_budget)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids, dtype=object),
                        vectors=matrix.astype(np.float32),
                        representation="z_graph", pool="mean_component_normalized")
    print(f"\nwrote {matrix.shape[0]} x {matrix.shape[1]} to {args.out}")


if __name__ == "__main__":
    main()
