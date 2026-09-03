"""Step 2: fit the linear probes on the frozen embeddings and report balanced accuracy.

    python run_probe.py --all
    python run_probe.py --task geometry
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from atomica.probe import (build_features, cluster_bootstrap_ci, fit_linear_probe,
                           hard_label_metrics)

import metal_tasks as T

_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def features(split: str, df: pd.DataFrame) -> np.ndarray:
    """The z_block rows for these sites, in the order they appear in `df`."""
    if split not in _CACHE:
        npz_path = os.path.join(T.EMBEDDINGS, f"{split}_z_block.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"missing {npz_path}\nRun: python extract_embeddings.py")
        with np.load(npz_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        meta = pd.read_parquet(os.path.join(T.EMBEDDINGS, f"{split}_meta.parquet"))
        arrays["_row_of_id"] = pd.Series(np.arange(len(meta)), index=meta["id"].values)
        _CACHE[split] = arrays

    arrays = _CACHE[split]
    rows = arrays["_row_of_id"].reindex(df["id"].to_numpy())
    assert rows.notna().all(), f"{split}: {int(rows.isna().sum())} sites have no saved embedding"
    return build_features({k: v[rows.to_numpy(dtype=int)] for k, v in arrays.items()
                           if k != "_row_of_id"}, "z_block")


def run_task(task: str, splits: Dict[str, pd.DataFrame], n_boot: int) -> dict:
    spec = T.TASKS[task]
    label = spec["label"]
    rows = {name: T.task_rows(task, df) for name, df in splits.items()}

    keep = T.classes_to_keep(task, rows["train"], rows["test"])
    rows = {k: df[df[label].isin(keep)] for k, df in rows.items()}
    if len(keep) < 2 or len(rows["test"]) < T.MIN_TEST_SITES:
        print(f"\n{task}: skipped, {len(keep)} classes and {len(rows['test'])} test sites")
        return {}

    print(f"\n{'=' * 84}\n{task}   {spec['blurb']}\n{'=' * 84}")
    print(f"  {len(keep)} classes, chance balanced accuracy {T.chance_level(len(keep)):.3f}")
    print("  sites: " + "   ".join(
        f"{k}={len(df):,} ({df['pdb_code'].nunique():,} entries)" for k, df in rows.items()))

    y = {k: df[label].to_numpy() for k, df in rows.items()}
    X = {k: features(k, df) for k, df in rows.items()}
    fit = fit_linear_probe(X["train"], y["train"], X["valid"], y["valid"], X["test"], cfg=T.LINEAR)
    pred = fit["test_pred"]
    print(f"  z_block: {fit['dim']}-d, C={fit['C']} chosen on validation")

    print(f"\n  {'metric':<14}{'value':>10}{'95% CI':>18}")
    table = {}
    for metric in T.METRICS:
        value, lo, hi = cluster_bootstrap_ci(y["test"], pred, rows["test"][T.BOOTSTRAP_CLUSTER],
                                             metric, n_boot=n_boot, seed=T.BOOTSTRAP_SEED)
        star = "*" if metric == T.PRIMARY_METRIC else " "
        print(f" {star}{metric:<14}{value:>10.3f}  [{lo:.3f}, {hi:.3f}]")
        table[metric] = {"value": value, "ci_low": lo, "ci_high": hi}

    predicted = rows["test"][["id", "pdb_code", "element"]].copy()
    predicted["task"] = task
    predicted["y_true"] = y["test"]
    predicted["y_pred"] = pred

    return {"task": task, "blurb": spec["blurb"], "n_classes": len(keep),
            "chance": T.chance_level(len(keep)), "C": fit["C"], "dim": fit["dim"],
            "n_sites": {k: int(len(df)) for k, df in rows.items()},
            "n_entries": {k: int(df["pdb_code"].nunique()) for k, df in rows.items()},
            "metrics": table, "_predictions": predicted}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(T.TASKS))
    ap.add_argument("--all", action="store_true", help="all four tasks")
    ap.add_argument("--n-boot", type=int, default=T.N_BOOTSTRAP)
    args = ap.parse_args()

    tasks = list(T.TASKS) if args.all else ([args.task] if args.task else [])
    if not tasks:
        ap.error("pass --task NAME or --all")

    splits = T.probe_split()
    print("probe split, grouped by PDB entry:")
    for name, df in splits.items():
        print(f"  {name:<6} {len(df):>7,} sites   {df['pdb_code'].nunique():>6,} entries")

    summaries: List[dict] = [s for s in (run_task(t, splits, args.n_boot) for t in tasks) if s]
    if not summaries:
        ap.error("every task was skipped; nothing to write")

    os.makedirs(T.PREDICTIONS, exist_ok=True)
    pd.concat([s.pop("_predictions") for s in summaries], ignore_index=True).to_parquet(
        os.path.join(T.PREDICTIONS, "test_predictions.parquet"), index=False)
    with open(os.path.join(T.PREDICTIONS, "summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=1, default=float)

    print(f"\n{'=' * 84}\nbalanced accuracy\n{'=' * 84}")
    print(f"{'task':<16}{'classes':>8}{'n test':>8}{'chance':>9}{'ATOMICA':>10}{'95% CI':>18}")
    for s in summaries:
        m = s["metrics"][T.PRIMARY_METRIC]
        print(f"{s['task']:<16}{s['n_classes']:>8}{s['n_sites']['test']:>8}"
              f"{s['chance']:>9.3f}{m['value']:>10.3f}"
              f"  [{m['ci_low']:.3f}, {m['ci_high']:.3f}]")
    print(f"\nwrote predictions and summary to {os.path.relpath(T.PREDICTIONS, T.HERE)}")


if __name__ == "__main__":
    main()
