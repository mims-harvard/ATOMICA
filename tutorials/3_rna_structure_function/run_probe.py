"""Train the frozen-embedding probe and report test-set performance.

The encoder stays frozen; the only fitted component is a four-layer MLP on the saved z_block
vectors. The head, training loop, seed ensembling, metrics and bootstrap all come from
atomica.probe, so nothing here reimplements the recipe.

  python run_probe.py --all
  python run_probe.py --task RNA_Site --feature h_block   # representation ablation
  python run_probe.py --task RNAGo --select-loss          # validation picks the loss
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from atomica.probe import (FEATURE_SETS, ProbeConfig, apply_standardizer,
                           bootstrap_ci, build_features, fit_standardizer, pool_saved_blocks,
                           train_probe)

import rna_tasks as T


def load_split(task, split, backbone, feature):
    """(ids, X, y) for one split.

    Residue tasks give one row per residue. Graph tasks pool each molecule's blocks by the
    parameter-free mean-std-global rule, which triples the width.
    """
    directory = T.embedding_dir(task, backbone)
    npz_path = os.path.join(directory, f"{task}_{split}_z_block.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"missing {npz_path}\nRun: python extract_embeddings.py "
                                f"--task {task} --backbone {backbone}")
    with np.load(npz_path) as npz:
        components = {k: npz[k] for k in npz.files}
    meta = pd.read_parquet(os.path.join(directory, f"{task}_{split}_meta.parquet"))
    X = build_features(components, feature)

    if T.TASKS[task]["residue_level"]:
        return list(meta["id"].astype(str)), X, meta["label"].values.astype(np.float32)

    graph_ids, pooled = pool_saved_blocks(X, meta["graph_id"].astype(str).values,
                                          meta["is_global"].values.astype(bool),
                                          mode="mean_std_global")
    dataset = T.load_dataset(task, split)
    labels = {str(item["id"]): np.atleast_1d(np.asarray(item["label"], dtype=np.float32))
              for item in dataset.data}
    y = np.stack([labels[g] for g in graph_ids])
    if T.TASKS[task]["task_type"] == "multiclass":
        y = y.reshape(-1).astype(np.int64)
    return list(graph_ids), pooled, y


def train_one(task, data, loss, cfg):
    """Fit the z-score on train only, then hand everything to atomica.probe.train_probe."""
    info = T.TASKS[task]
    (_, X_tr, y_tr), (_, X_va, y_va), (_, X_te, y_te) = data["train"], data["val"], data["test"]
    mu, sd = fit_standardizer(X_tr)
    X_tr, X_va, X_te = (apply_standardizer(X_tr, mu, sd), apply_standardizer(X_va, mu, sd),
                        apply_standardizer(X_te, mu, sd))
    return train_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, info["task_type"], info["primary"],
                       loss, cfg, class_names=info["class_names"])


def run_task(task, backbone, feature, select_loss, cfg):
    info = T.TASKS[task]
    metrics, primary = T.REPORTED_METRICS[info["residue_level"]], info["primary"]
    print(f"\n{'=' * 88}\n{task}   {info['blurb']}\nbackbone: {backbone}"
          f"{'' if backbone == T.PUBLISHED_BACKBONE[task] else '   (not the published arm)'}\n"
          f"{'=' * 88}")

    data = {s: load_split(task, s, backbone, feature) for s in T.SPLITS}
    print(f"representation: {feature}, {data['train'][1].shape[1]} wide")
    print("points: " + "   ".join(f"{s}={len(data[s][2])}" for s in T.SPLITS))

    losses = T.losses_for(task) if select_loss else [info["loss"]]
    runs = {loss: train_one(task, data, loss, cfg) for loss in losses}
    for loss, r in runs.items():
        print(f"  loss={loss:11s}  val {primary}={r['val_primary_ensemble']:.4f}   "
              f"test {primary}={r['ensemble'][primary]:.4f}")
    loss = max(losses, key=lambda l: runs[l]["val_primary_ensemble"])   # selection on validation
    result = runs[loss]
    if len(losses) > 1:
        print(f"  validation selects loss={loss}")

    y_test, probs = data["test"][2], result["test_probs"]
    ensemble = probs.mean(0)
    print(f"\n  test-set performance, {len(y_test)} points, "
          f"{len(cfg.seeds)}-seed ensemble")
    print(f"  {'metric':<14}{'value':>10}  {'95% CI':>18}   {'per-seed mean +- ci':>22}")
    table = {}
    for metric in metrics:
        value, lo, hi = bootstrap_ci(info["task_type"], y_test, ensemble, metric,
                                     n_boot=T.N_BOOTSTRAP, seed=T.BOOTSTRAP_SEED,
                                     class_names=info["class_names"])
        print(f"{' *' if metric == primary else '  '}{metric:<14}{value:>10.4f}  "
              f"[{lo:.3f}, {hi:.3f}]{'':>4}{result['per_seed_mean'][metric]:>12.4f} +- "
              f"{result['per_seed_ci95'][metric]:.4f}")
        table[metric] = {"value": value, "ci_low": lo, "ci_high": hi,
                         "per_seed_mean": result["per_seed_mean"][metric],
                         "per_seed_ci95": result["per_seed_ci95"][metric]}
    print("  * primary metric")

    os.makedirs(T.PREDICTIONS, exist_ok=True)
    tag = f"{task}__frozen_{backbone}__{feature}__{loss}"
    np.savez(os.path.join(T.PREDICTIONS, f"probs__{tag}.npz"), probs=probs,
             ids=np.asarray(data["test"][0]), y=y_test)
    summary = {"task": task, "backbone": backbone, "published_arm": T.PUBLISHED_BACKBONE[task],
               "feature": feature, "loss": loss, "primary": primary,
               "dim": int(result["dim"]), "n_test": int(len(y_test)),
               "seeds": list(cfg.seeds), "metrics": table}
    with open(os.path.join(T.PREDICTIONS, f"summary__{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(T.TASKS))
    ap.add_argument("--all", action="store_true", help="all four at their published checkpoint")
    ap.add_argument("--backbone", choices=list(T.BACKBONES), default=None)
    ap.add_argument("--feature", default="z_block", choices=list(FEATURE_SETS),
                    help="z_block is the full descriptor; the others are nested slices of it")
    ap.add_argument("--select-loss", action="store_true",
                    help="on the graph tasks, train both losses and pick on validation")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(T.PROBE.seeds))
    args = ap.parse_args()

    tasks = list(T.TASKS) if args.all else ([args.task] if args.task else [])
    if not tasks:
        ap.error("pass --task NAME or --all")

    cfg = ProbeConfig(hidden_dim=T.PROBE.hidden_dim, final_hidden_dim=T.PROBE.final_hidden_dim,
                      dropout=T.PROBE.dropout, lr=T.PROBE.lr, weight_decay=T.PROBE.weight_decay,
                      epochs=T.PROBE.epochs, patience=T.PROBE.patience,
                      batch_size=T.PROBE.batch_size, seeds=args.seeds,
                      use_batchnorm=T.PROBE.use_batchnorm)

    summaries = [run_task(task, args.backbone or T.PUBLISHED_BACKBONE[task], args.feature,
                          args.select_loss, cfg) for task in tasks]

    print(f"\n{'=' * 88}\ntest-set performance, primary metric per task\n{'=' * 88}")
    print(f"{'task':<14}{'metric':<12}{'value':>10}  {'95% CI':>18}")
    for s_ in summaries:
        m = s_["metrics"][s_["primary"]]
        print(f"{s_['task']:<14}{s_['primary']:<12}{m['value']:>10.4f}  "
              f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
