"""
Tutorial: MASIF-Ligand benchmark with ATOMICA

Loads the five fine-tuned ATOMICA checkpoints (seed 0-4) from
``checkpoints/benchmarks/masif/8A/seed{0..4}``, runs inference on the
MASIF-Ligand test set (467 binding pockets, 7-class ligand
classification), mean-probability ensembles the five models, and reports
the ensemble test-set accuracy.

Requirements:
  - An NVIDIA GPU (CUDA)
  - A conda environment with ATOMICA installed

Usage:
  python tutorial.py
"""

import argparse
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from atomica.data.dataset import MultiClassLabelledPDBDataset
from atomica.models import MultiClassClassifierModel
from atomica.trainers import Trainer

warnings.filterwarnings("ignore", category=UserWarning)

TUTORIAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TUTORIAL_DIR, "..", ".."))
CHECKPOINTS_ROOT = os.path.join(REPO_ROOT, "checkpoints", "benchmarks", "masif", "8A")
DATA_FILE = os.path.join(TUTORIAL_DIR, "data", "masif_test.parquet")
OUTPUT_DIR = os.path.join(TUTORIAL_DIR, "predictions")
N_SEEDS = 5
MASIF_LIGAND_LABELS = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]


def load_model(seed: int) -> Tuple[MultiClassClassifierModel, str]:
    ckpt_dir = os.path.join(CHECKPOINTS_ROOT, f"seed{seed}")
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    model = MultiClassClassifierModel.load_from_config_and_weights(config_path, ckpt_path)
    return model, ckpt_path


def run_inference(model, device: str) -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Test data not found: {DATA_FILE}")
    dataset = MultiClassLabelledPDBDataset(DATA_FILE)

    ids = [x["id"] for x in dataset.data]
    labels = [x["label"] for x in dataset.data]
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="  inference"):
            batch = MultiClassLabelledPDBDataset.collate_fn([dataset[i]])
            batch = Trainer.to_device(batch, device)
            probs.append(model.infer(batch).cpu().numpy()[0])

    probs_arr = np.stack(probs, axis=0)
    preds = np.argmax(probs_arr, axis=1)
    return pd.DataFrame({
        "id": ids,
        "label": labels,
        "pred": preds,
        "pred_probability": list(probs_arr),
    })


def average_seed_predictions(per_seed: List[pd.DataFrame]) -> pd.DataFrame:
    """Average predicted probabilities across the 5 seeds."""
    stacked = np.stack([np.stack(df["pred_probability"].values) for df in per_seed])
    mean_prob = stacked.mean(axis=0)
    pred = np.argmax(mean_prob, axis=1)
    return pd.DataFrame({
        "id": per_seed[0]["id"].values,
        "label": per_seed[0]["label"].values,
        "pred": pred,
        "pred_probability": list(mean_prob),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available).",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")

    print(f"Checkpoints root: {CHECKPOINTS_ROOT}")
    print(f"Test data file:   {DATA_FILE}")
    print(f"Predictions dir:  {OUTPUT_DIR}")
    print(f"Device:           {args.device}")
    print(f"\n{'=' * 72}\nMASIF-Ligand test set (8A pocket cutoff)\n{'=' * 72}")

    per_seed: List[pd.DataFrame] = []
    for seed in range(N_SEEDS):
        print(f"\n-- seed {seed} --")
        model, ckpt_path = load_model(seed)
        model.eval().to(args.device)
        print(f"   checkpoint: {ckpt_path}")

        seed_df = run_inference(model, args.device)
        seed_df["seed"] = seed
        seed_df["ckpt"] = ckpt_path
        per_seed.append(seed_df)

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions_df = average_seed_predictions(per_seed)
    out_path = os.path.join(OUTPUT_DIR, "masif_test_predictions.parquet")
    predictions_df.to_parquet(out_path, index=False)
    print(f"\nSaved test predictions -> {out_path}")

    accuracy = (predictions_df["pred"] == predictions_df["label"]).mean()
    print(f"Accuracy: ATOMICA {accuracy:.4f}")


if __name__ == "__main__":
    main()
