"""
Tutorial: RNA Structure-Function prediction with ATOMICA

Reproduces the ATOMICA paper results on the four rnaglib benchmark tasks
(RNAGo, RNA_Ligand, RNA_Protein, RNA_Site). For each task this script

  1. loads five fine-tuned ATOMICA checkpoints (seed 0-4),
  2. runs inference on the test set (and, where needed, the validation set
     used for threshold selection),
  3. ensembles the five model probabilities by mean,
  4. prints the ensemble test-set metrics, and
  5. saves per-seed and ensemble test predictions to `predictions/`.

Requirements:
  - An A100/H100 GPU (CUDA)
  - The `interactenv` conda environment with ATOMICA installed
      ~/.conda/envs/interactenv/bin/python tutorial.py

Usage:
  python tutorial.py                       # run all four tasks
  python tutorial.py --tasks RNA_Ligand    # run a single task
"""

import argparse
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm

from atomica.data.dataset import MultiClassLabelledPDBDataset
from atomica.models import (
    MultiClassClassifierModel,
    MultiLabelClassifierModel,
    ResidueClassifierModel,
)
from atomica.trainers import Trainer

from multiclass_metrics import compute_multiclass_metrics
from multilabel_metrics import compute_multilabel_metrics

warnings.filterwarnings("ignore", category=UserWarning)

TUTORIAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TUTORIAL_DIR, "..", ".."))
CHECKPOINTS_ROOT = os.path.join(REPO_ROOT, "checkpoints", "benchmarks")
DATA_DIR = os.path.join(TUTORIAL_DIR, "data")
OUTPUT_DIR = os.path.join(TUTORIAL_DIR, "predictions")
N_SEEDS = 5

TASKS = {
    "RNAGo": {
        "ckpt_subdir": "rna_go",  # seeds live directly under this
        "data_prefix": "rna_go",
        "model_cls": MultiLabelClassifierModel,
        "level": "graph",
    },
    "RNA_Ligand": {
        "ckpt_subdir": "rna_ligand/atomica",
        "data_prefix": "rna_ligand",
        "model_cls": MultiClassClassifierModel,
        "level": "graph",
    },
    "RNA_Protein": {
        "ckpt_subdir": "rna_protein/atomica",
        "data_prefix": "rna_protein",
        "model_cls": ResidueClassifierModel,
        "level": "residue",
    },
    "RNA_Site": {
        "ckpt_subdir": "rna_site/atomica",
        "data_prefix": "rna_site",
        "model_cls": ResidueClassifierModel,
        "level": "residue",
    },
}


def load_model(task_name: str, seed: int):
    info = TASKS[task_name]
    ckpt_dir = os.path.join(CHECKPOINTS_ROOT, info["ckpt_subdir"], f"seed{seed}")
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = info["model_cls"].load_from_config_and_weights(config_path, ckpt_path)
    return model, ckpt_path


def run_inference(task_name: str, split: str, model, device: str) -> pd.DataFrame:
    data_prefix = TASKS[task_name]["data_prefix"]
    data_file = os.path.join(DATA_DIR, f"{data_prefix}_{split}.parquet")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data not found: {data_file}")
    dataset = MultiClassLabelledPDBDataset(data_file)

    preds_chunks = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"  inference[{task_name}/{split}]"):
            batch = MultiClassLabelledPDBDataset.collate_fn([dataset[i]])
            batch = Trainer.to_device(batch, device)
            preds_chunks.append(model.infer(batch).cpu().numpy())
    preds = np.concatenate(preds_chunks)

    level = TASKS[task_name]["level"]
    if level == "residue":
        labels = np.concatenate([x["label"] for x in dataset.data])
        probs = preds.flatten()
        if task_name == "RNA_Protein":
            ids = sum([[x["id"]] * len(x["label"]) for x in dataset.data], [])
        else:  # RNA_Site: encode pdb residue index into the id
            ids = []
            for x in dataset.data:
                assert len(x["label"]) == len(x["block_to_pdb_indexes"])
                for _, pdb_index in sorted(x["block_to_pdb_indexes"].items()):
                    ids.append(f"{x['id']}_{pdb_index}")
        df = pd.DataFrame({"id": ids, "label": labels, "pred_probability": probs})
        df["pred"] = (df["pred_probability"] > 0.5).astype(int)
    else:
        ids = [x["id"] for x in dataset.data]
        labels = [x["label"] for x in dataset.data]
        df = pd.DataFrame(
            {"id": ids, "label": labels, "pred_probability": list(preds)}
        )
        if task_name == "RNA_Ligand":
            df["pred"] = df["pred_probability"].apply(lambda p: int(np.argmax(p)))
        else:  # RNAGo
            df["pred"] = df["pred_probability"].apply(
                lambda p: (np.asarray(p) > 0.5).astype(int)
            )
    return df


# ---------------------------------------------------------------------------
# Per-task metric blocks — the printed numbers match those reported in the
# ATOMICA paper.
# ---------------------------------------------------------------------------

def _stack_probs(series: pd.Series) -> np.ndarray:
    return np.stack(series.values)


def metrics_rnago(per_seed_test: List[pd.DataFrame]) -> Tuple[pd.DataFrame, np.ndarray]:
    """RNAGo: multilabel, ensemble by mean probability, fixed 0.5 threshold."""
    stacked = np.stack([_stack_probs(d["pred_probability"]) for d in per_seed_test])
    ensemble_prob = stacked.mean(axis=0)
    labels = _stack_probs(per_seed_test[0]["label"])
    m = compute_multilabel_metrics(y_true=labels, y_proba=ensemble_prob, threshold=0.5)
    row = {
        "model": "atomica",
        "subset_accuracy": m.subset_accuracy,
        "f1_macro": m.f1_macro,
        "f1_micro": m.f1_micro,
        "f1_weighted": m.f1_weighted,
        "auprc_macro": m.auprc_macro,
        "auprc_micro": m.auprc_micro,
        "roc_auc_ovr_macro": m.roc_auc_ovr_macro,
        "roc_auc_ovr_micro": m.roc_auc_ovr_micro,
    }
    return pd.DataFrame([row]), ensemble_prob


def metrics_rna_ligand(per_seed_test: List[pd.DataFrame]) -> Tuple[pd.DataFrame, np.ndarray]:
    """RNA_Ligand: 3-class multiclass, ensemble by mean probability, argmax."""
    stacked = np.stack([_stack_probs(d["pred_probability"]) for d in per_seed_test])
    ensemble_prob = stacked.mean(axis=0)
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    labels = np.stack(per_seed_test[0]["label"].values)
    m = compute_multiclass_metrics(
        y_true=labels, y_pred=ensemble_pred, y_proba=ensemble_prob, labels=[0, 1, 2]
    )
    row = {
        "model": "atomica",
        "accuracy": m.accuracy,
        "balanced_accuracy": m.balanced_accuracy,
        "f1_macro": m.f1_macro,
        "f1_micro": m.f1_micro,
        "f1_weighted": m.f1_weighted,
        "auprc_ovr_macro": m.auprc_ovr_macro,
        "auprc_ovr_weighted": m.auprc_ovr_weighted,
        "roc_auc_ovr_macro": m.roc_auc_ovr_macro,
        "roc_auc_ovr_weighted": m.roc_auc_ovr_weighted,
    }
    return pd.DataFrame([row]), ensemble_prob


def _binary_residue_metrics(
    per_seed_test: List[pd.DataFrame],
    per_seed_val: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """RNA_Protein / RNA_Site: binary per-residue, AUPRC + AUROC + accuracy.

    Ensemble by mean of per-seed probabilities. The decision threshold is
    chosen by sweeping 0..1 in 101 steps on the validation ensemble
    probability, taking the F1 score at the argmax threshold (`best_f1`), and
    thresholding the averaged test probability at that value.
    """
    test_probs = np.stack([d["pred_probability"].values.astype(np.float64)
                           for d in per_seed_test])
    ensemble_prob = test_probs.mean(axis=0)

    val_labels = per_seed_val[0]["label"].values
    val_ensemble_prob = np.mean(
        np.stack([d["pred_probability"].values.astype(np.float64)
                  for d in per_seed_val]),
        axis=0,
    )
    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = [
        f1_score(val_labels, (val_ensemble_prob >= t).astype(int))
        for t in thresholds
    ]
    best_f1 = max(f1s)

    test_labels = per_seed_test[0]["label"].values
    ensemble_pred = (ensemble_prob > best_f1).astype(int)
    precision, recall, _ = precision_recall_curve(test_labels, ensemble_prob)
    row = {
        "model": "atomica",
        "accuracy": float(np.mean(test_labels == ensemble_pred)),
        "roc_auc": float(roc_auc_score(test_labels, ensemble_prob)),
        "auprc": float(auc(recall, precision)),
    }
    return pd.DataFrame([row]), ensemble_prob


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_task(task_name: str, device: str) -> None:
    print(f"\n{'=' * 72}\nTask: {task_name}\n{'=' * 72}")
    info = TASKS[task_name]
    # Threshold for the ensemble prediction is tuned on the validation set for
    # both RNA_Protein and RNA_Site (binary residue-level tasks).
    needs_val = task_name in ("RNA_Protein", "RNA_Site")

    per_seed_test: List[pd.DataFrame] = []
    per_seed_val: List[pd.DataFrame] = []
    ckpt_paths: List[str] = []

    for seed in range(N_SEEDS):
        print(f"\n-- seed {seed} --")
        model, ckpt_path = load_model(task_name, seed)
        model.eval().to(device)
        ckpt_paths.append(ckpt_path)
        print(f"   checkpoint: {ckpt_path}")

        test_df = run_inference(task_name, "test", model, device)
        test_df["seed"] = seed
        test_df["ckpt"] = ckpt_path
        per_seed_test.append(test_df)

        if needs_val:
            val_df = run_inference(task_name, "val", model, device)
            val_df["seed"] = seed
            val_df["ckpt"] = ckpt_path
            per_seed_val.append(val_df)

        del model
        torch.cuda.empty_cache()

    # Save per-seed + ensemble test predictions.
    all_test = pd.concat(per_seed_test, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, f"{task_name}_test_predictions.parquet")
    all_test.to_parquet(out_path, index=False)
    print(f"\nSaved per-seed test predictions -> {out_path}")

    # Compute and print metrics.
    if task_name == "RNAGo":
        metrics_df, ensemble_prob = metrics_rnago(per_seed_test)
    elif task_name == "RNA_Ligand":
        metrics_df, ensemble_prob = metrics_rna_ligand(per_seed_test)
    elif task_name in ("RNA_Protein", "RNA_Site"):
        metrics_df, ensemble_prob = _binary_residue_metrics(
            per_seed_test, per_seed_val
        )
    else:
        raise ValueError(task_name)

    # Save ensemble probabilities alongside per-seed parquet.
    ensemble_path = os.path.join(OUTPUT_DIR, f"{task_name}_test_ensemble.parquet")
    level = info["level"]
    if level == "residue":
        ensemble_df = pd.DataFrame({
            "id": per_seed_test[0]["id"].values,
            "label": per_seed_test[0]["label"].values,
            "ensemble_probability": ensemble_prob,
        })
    else:
        ensemble_df = pd.DataFrame({
            "id": per_seed_test[0]["id"].values,
            "label": list(per_seed_test[0]["label"].values),
            "ensemble_probability": list(ensemble_prob),
        })
    ensemble_df.to_parquet(ensemble_path, index=False)
    print(f"Saved ensemble probabilities   -> {ensemble_path}")

    print(f"\n--- {task_name} ensemble test predictions (head) ---")
    ensemble_preview = ensemble_df.head(10)
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(ensemble_preview.to_string(index=False))

    print(f"\n--- {task_name} ensemble test-set metrics ---")
    with pd.option_context("display.float_format", lambda x: f"{x: .4f}"):
        print(metrics_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help="Which task(s) to run (default: all four).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available).",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Checkpoints root: {CHECKPOINTS_ROOT}")
    print(f"Test data dir:    {DATA_DIR}")
    print(f"Predictions dir:  {OUTPUT_DIR}")
    print(f"Device:           {args.device}")

    for task in args.tasks:
        run_task(task, args.device)

    print("\nDone. Predictions written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
