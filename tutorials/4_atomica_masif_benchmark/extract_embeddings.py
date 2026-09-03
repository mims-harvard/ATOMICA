"""Extract frozen ATOMICA pocket embeddings for the MaSIF-ligand benchmark.

Writes embeddings/atomica_{train,val,test}.npz. The encoder is frozen; nothing is trained here.

Usage:
    python extract_embeddings.py
"""

import argparse
import os

import numpy as np

from atomica import representations as R

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("train", "val", "test")
EXPECTED_ROWS = {"train": 1839, "val": 203, "test": 467}


def extract(model, dataset_class, parquet, split, batch_size, device, atom_budget):
    dataset = dataset_class(parquet)
    n = len(dataset.data)
    print(f"[{split}] {n} pockets")

    # strict=True: fail on a batch that does not fit in memory, rather than skipping structures.
    rows = R.embed_dataset(model, dataset, ["z_graph"], pool="mean_std_global",
                           batch_size=batch_size, device=device, strict=True,
                           atom_budget=atom_budget, progress=True)
    if len(rows) != n:
        raise RuntimeError(f"[{split}] got {len(rows)} of {n} pockets")
    if n != EXPECTED_ROWS[split]:
        print(f"[{split}] warning: expected {EXPECTED_ROWS[split]} pockets")

    labels = {item["id"]: item["label"] for item in dataset.data}
    ids = [row["id"] for row in rows]
    X = np.stack([np.asarray(row["z_graph"], dtype=np.float32) for row in rows])
    y = np.asarray([labels[i] for i in ids], dtype=np.int64)
    print(f"[{split}] {X.shape[0]} x {X.shape[1]}")
    return np.asarray(ids), X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=os.path.join(HERE, "checkpoints"))
    parser.add_argument("--weights", default="masif_excluded_pretrain.pt")
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--output-dir", default=os.path.join(HERE, "embeddings"))
    parser.add_argument("--batch-size", type=int, default=16, help="Structures per batch.")
    parser.add_argument("--atom-budget", type=int, default=None,
                        help="Cap atoms per batch, if a batch does not fit in memory.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    config = os.path.join(args.checkpoint_dir, "config.json")
    weights = os.path.join(args.checkpoint_dir, args.weights)
    for path in (config, weights):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; see checkpoints/README.md")

    device = R._resolve_device(args.device)
    print(f"checkpoint {os.path.basename(weights)}\ndevice     {device}\n")

    model, dataset_class = R.load_model(config, weights)
    model = model.eval().to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    for split in SPLITS:
        parquet = os.path.join(args.data_dir, f"masif_{split}.parquet")
        ids, X, y = extract(model, dataset_class, parquet, split, args.batch_size, device,
                            args.atom_budget)
        np.savez(os.path.join(args.output_dir, f"atomica_{split}.npz"), ids=ids, X=X, y=y)

    print("\nDone. Next: python run_benchmark.py")


if __name__ == "__main__":
    main()
