"""Score the five released RNA-GO fine-tuned checkpoints, the one fine-tuned bar in Fig. 3.

Everywhere else in this tutorial the encoder is frozen. RNA-GO is the exception: frozen ATOMICA
reaches 0.673 macro-F1 while fine-tuned ATOMICA reaches 0.951.

  python run_finetuned_rna_go.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from atomica.data.dataset import PDBDataset
from atomica.models import MultiLabelClassifierModel
from atomica.probe import bootstrap_ci

import rna_tasks as T

TASK = "RNAGo"
TRAINING_SEEDS = [8, 2026, 2025, 7, 15]     # not 0-4; SEED=0..4 reproduces the method, not these
METRICS = ["f1_macro", "f1_micro", "auprc_macro", "auroc_macro"]


@torch.no_grad()
def infer(model, dataset, device):
    """Per-graph probabilities, one graph per forward pass."""
    probs = []
    for item in tqdm(dataset.data, desc="  inference", leave=False):
        batch = PDBDataset.collate_fn([item["data"]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        probs.append(model.infer(batch).cpu().numpy())
    return np.concatenate(probs).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = T.load_dataset(TASK, "test")
    labels = np.stack([np.asarray(item["label"], dtype=np.float32) for item in dataset.data])
    ids = [str(item["id"]) for item in dataset.data]

    root = os.path.join(T.CHECKPOINTS, "rna_go_finetuned")
    per_seed = []
    for i, training_seed in enumerate(TRAINING_SEEDS):
        weights = os.path.join(root, f"seed{i}", "model.pt")
        config = os.path.join(root, f"seed{i}", "config.json")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"missing {weights}; see the README section on checkpoints")
        print(f"-- seed{i} (trained with seed {training_seed}) --")
        model = MultiLabelClassifierModel.load_from_config_and_weights(config, weights)
        model.eval().to(args.device)
        per_seed.append(infer(model, dataset, args.device))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    per_seed = np.stack(per_seed)
    ensemble = per_seed.mean(0)

    info = T.TASKS[TASK]
    print(f"\n{'=' * 72}\nRNA-GO, fine-tuned, {len(per_seed)}-seed ensemble, "
          f"{len(labels)} test structures\n{'=' * 72}")
    print(f"  {'metric':<14}{'value':>10}  {'95% CI':>18}")
    table = {}
    for metric in METRICS:
        value, lo, hi = bootstrap_ci(info["task_type"], labels, ensemble, metric,
                                     n_boot=T.N_BOOTSTRAP, seed=T.BOOTSTRAP_SEED,
                                     class_names=info["class_names"])
        print(f"  {metric:<14}{value:>10.4f}  [{lo:.3f}, {hi:.3f}]")
        table[metric] = {"value": value, "ci_low": lo, "ci_high": hi}

    os.makedirs(T.PREDICTIONS, exist_ok=True)
    np.savez(os.path.join(T.PREDICTIONS, "probs__RNAGo__finetuned.npz"),
             probs=per_seed, ids=np.asarray(ids), y=labels)
    with open(os.path.join(T.PREDICTIONS, "summary__RNAGo__finetuned.json"), "w") as fh:
        json.dump({"task": TASK, "arm": "fine-tuned", "training_seeds": TRAINING_SEEDS,
                   "n_test": int(len(labels)), "metrics": table}, fh, indent=1)


if __name__ == "__main__":
    main()
