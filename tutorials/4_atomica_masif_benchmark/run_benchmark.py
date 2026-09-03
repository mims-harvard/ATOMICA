"""Train a classifier head on frozen ATOMICA pocket embeddings and score it.

Only the head is trained. The ATOMICA encoder was frozen in extract_embeddings.py.

Usage:
    python run_benchmark.py
"""

import argparse
import json
import os

import numpy as np

from atomica.probe import (ProbeConfig, apply_standardizer, bootstrap_ci, fit_standardizer,
                           metrics_from_prob, train_probe)

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
METRICS = ["f1_macro", "f1_micro", "auprc_macro", "auroc_macro"]
TITLES = {"f1_macro": "macro-F1", "f1_micro": "micro-F1",
          "auprc_macro": "macro-AUPRC", "auroc_macro": "macro-AUROC"}

PROBE = ProbeConfig(hidden_dim=512, final_hidden_dim=32, dropout=0.3, lr=1e-3, weight_decay=0.0,
                    epochs=100, patience=15, batch_size=64, seeds=[0, 1, 2, 3, 4],
                    min_delta=1e-4, use_batchnorm=True)
PRIMARY = "auprc_macro"
N_BOOT = 2000
BOOT_SEED = 0


def load_embeddings(directory):
    out = {}
    for split in ("train", "val", "test"):
        path = os.path.join(directory, f"atomica_{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run extract_embeddings.py first")
        z = np.load(path, allow_pickle=True)
        out[split] = (z["ids"].astype(str), z["X"], z["y"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default=os.path.join(HERE, "embeddings"))
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data = load_embeddings(args.embeddings_dir)
    test_ids, _, y_test = data["test"]
    print(f"pockets  train {len(data['train'][2])}, val {len(data['val'][2])}, test {len(y_test)}")
    print(f"features {data['train'][1].shape[1]}-d\n")

    # Standardisation is fit on training pockets only.
    mu, sd = fit_standardizer(data["train"][1])
    X = {s: apply_standardizer(data[s][1], mu, sd) for s in data}

    print(f"Training {len(PROBE.seeds)} seeds, early stopping on validation {PRIMARY}")
    run = train_probe(X["train"], data["train"][2], X["val"], data["val"][2], X["test"], y_test,
                      task_type="multiclass", primary=PRIMARY, loss="ce", cfg=PROBE,
                      class_names=LABELS, device=args.device)
    probs = run["test_probs"].mean(0)

    scored = {k: bootstrap_ci("multiclass", y_test, probs, k, n_boot=N_BOOT, seed=BOOT_SEED)
              for k in METRICS}
    print(f"\n{len(y_test)} test pockets, {N_BOOT} bootstrap resamples")
    for k in METRICS:
        value, lo, hi = scored[k]
        print(f"  {TITLES[k]:<14}{value:.3f} [{lo:.3f}, {hi:.3f}]")

    per_class = metrics_from_prob("multiclass", y_test, probs, LABELS)
    print("\nPer-class F1")
    for ligand in LABELS:
        print(f"  {ligand:<6}{per_class[f'F1_{ligand}']:.3f}")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as handle:
        json.dump({
            "n_test": int(len(y_test)),
            "n_seeds": len(PROBE.seeds),
            "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED},
            "metrics": {k: dict(zip(("value", "ci_low", "ci_high"), scored[k])) for k in METRICS},
            "per_class_f1": {c: per_class[f"F1_{c}"] for c in LABELS},
        }, handle, indent=2)
    np.savez_compressed(os.path.join(args.results_dir, "test_probabilities.npz"),
                        ids=test_ids, y=y_test, probabilities=probs,
                        per_seed=run["test_probs"], class_names=np.array(LABELS))
    print(f"\nWrote {args.results_dir}/")


if __name__ == "__main__":
    main()
