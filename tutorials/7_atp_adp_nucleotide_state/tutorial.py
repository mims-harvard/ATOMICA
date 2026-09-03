#!/usr/bin/env python
"""ATP versus ADP nucleotide state from ligand-free pockets.

Three stages:

    embed    frozen encoder over the 404 pocket graphs -> one pooled z_graph row per pocket
    probe    five-fold cluster-disjoint cross-validation -> one probability per pocket
    report   cluster-macro AUROC over the five strata, with bootstrap intervals

    python tutorial.py --stage all
"""

from __future__ import annotations

import argparse
import os
import sys

# Must precede `import torch`: deterministic cuBLAS reductions need this set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

import atp_adp as T                                                        # noqa: E402
from atomica import representations as R                                   # noqa: E402
from atomica.probe import train_probe                                      # noqa: E402


def make_deterministic() -> None:
    """TF32 is on by default from Ampere onwards and makes different GPUs disagree. Off here, the
    encoder pass is reproducible at a fixed batch size."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)


# --------------------------------------------------------------------------------------- embed
def _items(graphs: pd.DataFrame) -> list:
    """Graph rows as {"id", "data"}, the shape `atomica.representations` works in."""
    keys = ["X", "B", "A", "atom_positions", "block_lengths", "segment_ids"]

    def cast(key, value):
        if key == "X":
            # X is an object array of per-atom arrays, and whole-number coordinates round-trip
            # through parquet as integers, which the model's norm layer rejects.
            return np.stack([np.asarray(p, dtype=np.float32).reshape(-1)[:3]
                             for p in value]).tolist()
        return np.asarray(value).tolist()

    return [{"id": str(row["id"]),
             "data": dict({k: cast(k, row[k]) for k in keys}, label=int(row["label"]))}
            for _, row in graphs.iterrows()]


def _collate(chunk: list, device: str) -> dict:
    from atomica.data.dataset import PDBDataset

    batch = PDBDataset.collate_fn([item["data"] for item in chunk])
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def stage_embed(args) -> str:
    """Pooled z_graph per pocket, under both poolings, from one forward pass each."""
    from atomica.data.pdb_utils import VOCAB
    from atomica.models.prediction_model import PredictionModel

    for path in (T.MODEL_CONFIG, T.MODEL_WEIGHTS):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}\nDownload the checkpoint first:\n"
                             f"  hf download ada-f/ATOMICA --local-dir checkpoints/")
    if T.EXTRACT_BATCH_SIZE > 1 and not T.GROUP_BY_MAX_BLOCK:
        raise SystemExit("batching without grouping by largest block makes the features depend on "
                         "file order; set EXTRACT_BATCH_SIZE to 1 or turn grouping on")

    pockets, graphs = T.load_pockets(), T.load_graphs()
    positions = T.shared_site_positions(graphs, pockets)
    sizes = np.asarray([len(v) for v in positions.values()])
    print(f"{len(pockets)} pockets, {pockets.fold_unit.nunique()} clusters, "
          f"shared site {sizes.min()}-{sizes.max()} residues (median {int(np.median(sizes))})")

    VOCAB.load_tokenizer("PS_300")
    model = PredictionModel.load_from_config_and_weights(T.MODEL_CONFIG, T.MODEL_WEIGHTS)
    model = model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dims = model.invariant_component_dims()

    items = _items(graphs)
    batches = R.group_batches(items, T.EXTRACT_BATCH_SIZE,
                              group_by_max_block=T.GROUP_BY_MAX_BLOCK,
                              atom_budget=T.ATOM_BUDGET)
    print(f"{len(batches)} batches at size {T.EXTRACT_BATCH_SIZE}, grouped by largest block")

    by_index, done = {}, 0
    with torch.no_grad():
        for indices, chunk in batches:
            batch = _collate(chunk, args.device)
            rv = model.infer(batch, return_invariant_repr=True, invariant_pool=None)
            is_global = batch["B"] == model.global_block_id
            keep = torch.zeros_like(is_global)
            offset = 0
            for item in chunk:
                keep[offset + torch.as_tensor(positions[item["id"]], device=keep.device)] = True
                offset += len(item["data"]["B"])
            # keep restricts the mean and standard deviation to the shared site; the global
            # block is taken whole and is never in keep
            shared = R.pool_blocks(rv.block_invariant_repr, rv.batch_id, is_global, T.POOLING,
                                   component_dims=dims, keep=keep).float().cpu().numpy()
            whole = R.get(model, batch, T.REPRESENTATION, pool=T.POOLING,
                          return_value=rv).float().cpu().numpy()
            for k, index in enumerate(indices):
                by_index[index] = (shared[k], whole[k])
            done += len(indices)
            if done % 100 < T.EXTRACT_BATCH_SIZE:
                print(f"  {done}/{len(items)}", flush=True)

    order = sorted(by_index)
    if order != list(range(len(items))):
        raise RuntimeError(f"{len(items) - len(order)} pockets were never embedded")
    ids = [items[i]["id"] for i in order]
    if ids != list(pockets.id):
        raise RuntimeError("embedding order does not match pockets.csv")

    os.makedirs(T.EMBEDDINGS, exist_ok=True)
    out = os.path.join(T.EMBEDDINGS, "atomica_z_graph.npz")
    np.savez(out,
             shared_site=np.stack([by_index[i][0] for i in order]).astype(np.float32),
             whole_pocket=np.stack([by_index[i][1] for i in order]).astype(np.float32),
             ids=np.asarray(ids), labels=pockets.label.to_numpy().astype(np.int64),
             fold_units=pockets.fold_unit.to_numpy().astype(str),
             extract_batch_size=np.int64(T.EXTRACT_BATCH_SIZE),
             group_by_max_block=np.bool_(T.GROUP_BY_MAX_BLOCK))
    print(f"wrote {os.path.relpath(out, HERE)}")
    return out


# --------------------------------------------------------------------------------------- probe
def stage_probe(args) -> str:
    """Cluster-disjoint cross-validation with the grid selected on a validation fold.

    The test fold is never read during selection: the validation fold is a separate set of
    clusters drawn from the training side, and the winning configuration is refitted on the
    training and validation clusters together before being applied once to the test fold.
    """
    path = os.path.join(T.EMBEDDINGS, "atomica_z_graph.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run --stage embed first")
    z = np.load(path, allow_pickle=True)
    if "extract_batch_size" not in z.files:
        raise SystemExit("the embeddings carry no batching record; re-run --stage embed")
    if int(z["extract_batch_size"]) > 1 and not bool(z["group_by_max_block"]):
        raise SystemExit("these embeddings were batched without grouping, so they depend on file "
                         "order; delete them and re-run --stage embed")

    X = z[args.pool].astype(np.float32)
    y = z["labels"].astype(np.int64)
    clusters = np.asarray([str(v) for v in z["fold_units"]])
    ids = np.asarray([str(v) for v in z["ids"]])
    which = ("shared site only" if args.pool == "shared_site"
             else "all 50 residues, the ablation rather than the reported arm")
    print(f"{X.shape[0]} pockets x {X.shape[1]} features, {len(set(clusters))} clusters, "
          f"{int((y == 1).sum())} ATP / {int((y == 0).sum())} ADP; pooling {which}")

    folds = T.fold_assignment(clusters, T.FOLDS)
    configurations = T.grid()[:2] if args.quick else T.grid()
    seeds = [args.seed] if args.quick else [args.seed + i for i in range(len(T.PROBE.seeds))]

    scores = np.full(len(y), np.nan)
    chosen, grid_scores = [], []
    for fold in range(T.FOLDS):
        test = folds == fold
        val = folds == ((fold + 1) % T.FOLDS)
        train = ~test & ~val

        best, best_val, scored = configurations[0], -np.inf, []
        X_tr, X_va = T.fold_standardize(X, train, train, val)
        for hidden, dropout, lr in configurations:
            cfg = T.config_for(hidden, dropout, lr, seeds=seeds)
            # validation is passed as the test argument too; only the validation score is read
            run = train_probe(X_tr, y[train], X_va, y[val], X_va, y[val],
                              T.TASK_TYPE, T.PRIMARY, T.LOSS, cfg, device=args.device)
            value = run["val_primary_ensemble"]
            scored.append(dict(fold=fold, hidden=hidden, dropout=dropout, lr=lr,
                               val_primary=float(value)))
            if np.isfinite(value) and value > best_val:
                best, best_val = (hidden, dropout, lr), value

        # The margin over the runner-up is usually small, which is why the final figure moves by
        # a few hundredths between runs. See the README.
        ranked = sorted((c["val_primary"] for c in scored), reverse=True)
        margin = (ranked[0] - ranked[1]) if len(ranked) > 1 else float("nan")
        print(f"fold {fold}: hidden={best[0]} dropout={best[1]} lr={best[2]}, "
              f"validation {T.PRIMARY} {best_val:.4f}, margin {margin:.4f}", flush=True)
        chosen.append(dict(fold=fold, hidden=best[0], dropout=best[1], lr=best[2],
                           val_primary=float(best_val), margin_over_runner_up=float(margin)))
        grid_scores.extend(scored)

        fit = ~test
        X_fit, X_va, X_te = T.fold_standardize(X, fit, fit, val, test)
        run = train_probe(X_fit, y[fit], X_va, y[val], X_te, y[test], T.TASK_TYPE, T.PRIMARY,
                          T.LOSS, T.config_for(*best, seeds=seeds), device=args.device)
        scores[test] = run["test_probs"].mean(0)          # averaged over the seeds

    if np.isnan(scores).any():
        raise RuntimeError(f"{int(np.isnan(scores).sum())} pockets never landed in a test fold")

    os.makedirs(T.PREDICTIONS, exist_ok=True)
    pockets = T.load_pockets().set_index("id")
    name = "atomica" if args.pool == "shared_site" else f"atomica_{args.pool}"
    out = os.path.join(T.PREDICTIONS, f"{name}.csv")
    pd.DataFrame(dict(id=ids, cluster=pockets.loc[ids].cluster.to_numpy(), fold_unit=clusters,
                      label=y, prob_atp=scores)).to_csv(out, index=False)
    pd.DataFrame(chosen).to_csv(os.path.join(T.PREDICTIONS, f"{name}_selected.csv"), index=False)
    pd.DataFrame(grid_scores).to_csv(os.path.join(T.PREDICTIONS, f"{name}_grid.csv"), index=False)
    print(f"wrote {os.path.relpath(out, HERE)}")
    return out


# -------------------------------------------------------------------------------------- report
def stage_report(args) -> pd.DataFrame:
    """Score this run's predictions against the values reported in the paper."""
    pockets = T.load_pockets()
    name = "atomica" if args.pool == "shared_site" else f"atomica_{args.pool}"
    path = os.path.join(T.PREDICTIONS, f"{name}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run --stage probe first")
    scores = pd.read_csv(path)
    scores["id"] = scores.id.astype(str)
    scores["fold_unit"] = scores.fold_unit.astype(str)
    result = T.evaluate(scores, pockets, strata=args.strata)
    result.insert(0, "model", T.DISPLAY.get(name, name))

    os.makedirs(T.RESULTS, exist_ok=True)
    out = os.path.join(T.RESULTS, "cluster_macro_auroc.csv")
    result.to_csv(out, index=False)

    print("\nMean within-cluster AUROC, ATP against ADP. Chance is 0.500 everywhere.\n")
    print(f"{'stratum':<32s} {'AUROC':>7s} {'95% CI':>18s} {'pockets':>8s} {'clusters':>9s}")
    for row in result.itertuples(index=False):
        print(f"{row.stratum_label:<32s} {row.cluster_macro_auroc:>7.3f}   "
              f"[{row.ci_low:.3f}, {row.ci_high:.3f}] {row.n_pockets:>8d} "
              f"{row.n_eval_clusters:>9d}")
    print(f"\nwrote {os.path.relpath(out, HERE)}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=["embed", "probe", "report", "all"])
    p.add_argument("--pool", default="shared_site", choices=["shared_site", "whole_pocket"],
                   help="shared_site is the reported arm; whole_pocket is an ablation that pools "
                        "all 50 residues and scores higher")
    p.add_argument("--strata", nargs="*", default=list(T.STRATA), choices=list(T.STRATA))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="one seed, two grid points; a smoke test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    make_deterministic()
    if args.stage in ("embed", "all"):
        stage_embed(args)
    if args.stage in ("probe", "all"):
        stage_probe(args)
    if args.stage in ("report", "all"):
        stage_report(args)


if __name__ == "__main__":
    main()
