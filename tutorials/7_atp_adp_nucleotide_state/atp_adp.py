"""Benchmark definition for ATP/ADP nucleotide state. No modelling code here.

The frozen-representation recipe is `atomica.probe`; the representation names are
`atomica.representations`. This module holds the paths, the evaluation strata, the readout and
the training constants.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from atomica.probe import ProbeConfig, apply_standardizer, fit_standardizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
EMBEDDINGS = os.path.join(HERE, "embeddings")
PREDICTIONS = os.path.join(HERE, "predictions")
RESULTS = os.path.join(HERE, "results")

POCKETS_CSV = os.path.join(DATA, "pockets.csv")
GRAPHS_PARQUET = os.path.join(DATA, "pocket_graphs.parquet")

# The standard released ATOMICA pretrained encoder, used frozen. No other checkpoint is needed.
CHECKPOINT_DIR = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain")
MODEL_CONFIG = os.path.join(CHECKPOINT_DIR, "pretrain_model_config.json")
MODEL_WEIGHTS = os.path.join(CHECKPOINT_DIR, "pretrain_model_weights.pt")

REPRESENTATION = "z_graph"
POOLING = "mean_std_global"

# Pool only the binding-site positions a pocket shares with its matched partner. The other
# residues stay in the forward pass and are left out of the mean and standard deviation.
POOL_DEFAULT = "shared_site"

# Batches hold only structures with the same largest block, so each is embedded at the padding
# width it would have alone and the batch size does not change any vector.
EXTRACT_BATCH_SIZE = 8
GROUP_BY_MAX_BLOCK = True
ATOM_BUDGET = None                 # cap atoms per batch on a small GPU; splitting a group is safe

#: Evaluation strata. `n_clusters` counts the clusters holding both states, which are the ones
#: that contribute an AUROC.
STRATA: Dict[str, dict] = {
    "all": dict(column="stratum_all", label="all pockets", n_pockets=404, n_clusters=60),
    "metal_concordant": dict(column="stratum_metal_concordant", label="metal-concordant",
                             n_pockets=245, n_clusters=44),
    "with_metal": dict(column="stratum_with_metal", label="with metal",
                       n_pockets=129, n_clusters=25),
    "metal_free": dict(column="stratum_metal_free", label="metal-free",
                       n_pockets=116, n_clusters=20),
    "held_out": dict(column="stratum_held_out", label="held out and metal-concordant",
                     n_pockets=50, n_clusters=9),
}

DISPLAY = {"atomica": "ATOMICA", "atomica_whole_pocket": "ATOMICA (whole pocket)"}

FOLDS = 5
GRID_HIDDEN = (64, 256)
GRID_DROPOUT = (0.1, 0.3)
GRID_LR = (1e-4, 1e-3)
MIN_COL_STD = 1e-6                 # drop columns this flat on the training rows

# use_batchnorm=False makes this the same four-layer MLP the comparison models were scored
# through. early_stopping=False keeps the final weights of the fixed 60-epoch budget.
PROBE = ProbeConfig(hidden_dim=256, final_hidden_dim=32, dropout=0.3, lr=1e-3, weight_decay=0.0,
                    epochs=60, batch_size=32, seeds=(0, 1, 2, 3, 4), use_batchnorm=False,
                    early_stopping=False)

TASK_TYPE = "binary"
PRIMARY = "auprc"                  # what a grid configuration is selected on
LOSS = "bce"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


# --------------------------------------------------------------------------------------- data
def load_pockets() -> pd.DataFrame:
    """One row per pocket: label, cluster, shared site, metal status and the strata."""
    P = pd.read_csv(POCKETS_CSV)
    P["id"] = P.id.astype(str)
    P["fold_unit"] = P.fold_unit.astype(str)
    if len(P) != 404:
        raise ValueError(f"expected 404 pockets, found {len(P)}")
    return P


def load_graphs() -> pd.DataFrame:
    """The 404 pocket graphs, ordered to match :func:`load_pockets`."""
    G = pd.read_parquet(GRAPHS_PARQUET)
    G["id"] = G.id.astype(str)
    return G.set_index("id").loc[load_pockets().id].reset_index()


# ------------------------------------------------------------------ residue keys to block index
def residue_block_index(block_to_pdb_indexes, n_blocks: int) -> Dict[str, int]:
    """Map a residue key to its row in ``B``.

    Graph builders number these keys from either 0 or 1, and a global block sits at position 0
    either way. Normalising by the smallest key handles both; using the enumeration position
    instead would shift every residue by one.
    """
    if isinstance(block_to_pdb_indexes, str):
        block_to_pdb_indexes = json.loads(block_to_pdb_indexes)
    else:
        block_to_pdb_indexes = dict(block_to_pdb_indexes)
    if not block_to_pdb_indexes:
        return {}
    numeric = {str(k): int(k) for k in block_to_pdb_indexes}
    lo = min(numeric.values())
    out: Dict[str, int] = {}
    for k, ki in numeric.items():
        j = ki - lo + 1                                  # +1 skips the global block
        if 0 < j < n_blocks:
            out[str(block_to_pdb_indexes[k])] = j
    if out and (min(out.values()) < 1 or max(out.values()) >= n_blocks):
        raise ValueError("a residue mapped onto the global block or past the end of the graph")
    return out


def shared_site_positions(graphs: pd.DataFrame, pockets: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Block indices to pool for each pocket, from the shared site recorded in pockets.csv."""
    index = {str(r.id): residue_block_index(r.block_to_pdb_indexes, len(r.B))
             for r in graphs.itertuples()}
    out = {}
    for pocket_id, site in zip(pockets.id, pockets.shared_site):
        k2pos = index[pocket_id]
        pos = sorted({k2pos[k] for k in str(site).split(";") if k in k2pos})
        if len(pos) < 5:
            raise ValueError(f"{pocket_id}: only {len(pos)} shared residues resolved to a block")
        out[pocket_id] = np.asarray(pos, dtype=np.int64)
    return out


# ------------------------------------------------------------------------------------- readout
def cluster_macro_auroc(labels, scores, clusters) -> Tuple[float, np.ndarray, List[str]]:
    """Mean over clusters of the AUROC computed inside each cluster.

    Never compares pockets from different clusters, so protein family cannot contribute, and
    chance is 0.500 in every stratum. A cluster holding one state gives no AUROC and is dropped.
    """
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(labels) == 1).astype(int)
    s = np.asarray(scores, dtype=float)
    c = np.asarray([str(v) for v in clusters])
    per, used = [], []
    for cluster in sorted(set(c)):
        idx = np.flatnonzero(c == cluster)
        if len(set(y[idx].tolist())) < 2:
            continue
        per.append(roc_auc_score(y[idx], s[idx]))
        used.append(cluster)
    if not per:
        raise ValueError("no cluster in this stratum holds both states")
    return float(np.mean(per)), np.asarray(per, dtype=float), used


def cluster_bootstrap(per_cluster, n_boot: int = N_BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """95% percentile interval, resampling clusters. Pockets within a cluster are near-duplicates
    of one protein, so the cluster is the independent unit."""
    rng = np.random.default_rng(seed)
    per = np.asarray(per_cluster, dtype=float)
    draws = per[rng.integers(0, len(per), size=(n_boot, len(per)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(scores: pd.DataFrame, pockets: pd.DataFrame,
             strata: Sequence[str] = tuple(STRATA)) -> pd.DataFrame:
    """Cluster-macro AUROC per stratum. The strata re-partition one set of predictions."""
    joined = scores.merge(pockets[["id"] + [STRATA[s]["column"] for s in strata]], on="id")
    if len(joined) != len(scores):
        raise ValueError(f"{len(scores)} predictions joined to {len(joined)} rows")
    rows = []
    for stratum in strata:
        keep = joined[joined[STRATA[stratum]["column"]]]
        value, per, _ = cluster_macro_auroc(keep.label, keep.prob_atp, keep.fold_unit)
        lo, hi = cluster_bootstrap(per)
        rows.append(dict(stratum=stratum, stratum_label=STRATA[stratum]["label"],
                         n_pockets=int(len(keep)), n_eval_clusters=int(len(per)),
                         cluster_macro_auroc=value, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ probe plumbing
def fold_standardize(X: np.ndarray, fit_mask: np.ndarray, *masks: np.ndarray) -> List[np.ndarray]:
    """Drop near-constant columns then z-score, both fit on the training rows of this fold."""
    sd = X[fit_mask].std(0)
    keep = sd > MIN_COL_STD
    if not keep.any():
        raise ValueError("every column is constant on the training rows of this fold")
    mu, sigma = fit_standardizer(X[fit_mask][:, keep])
    return [apply_standardizer(X[m][:, keep], mu, sigma).astype(np.float32) for m in masks]


def fold_assignment(clusters: Sequence[str], folds: int = FOLDS) -> np.ndarray:
    """Round-robin over sorted clusters, so no cluster is split across the train/test boundary."""
    unique = sorted(set(str(c) for c in clusters))
    fold_of = {c: i % folds for i, c in enumerate(unique)}
    return np.asarray([fold_of[str(c)] for c in clusters], dtype=int)


def grid() -> List[Tuple[int, float, float]]:
    """The eight (hidden, dropout, learning rate) configurations, in a fixed order."""
    return [(h, d, lr) for h in GRID_HIDDEN for d in GRID_DROPOUT for lr in GRID_LR]


def config_for(hidden: int, dropout: float, lr: float, seeds: Sequence[int] = None) -> ProbeConfig:
    """:data:`PROBE` with one grid point substituted in."""
    import dataclasses
    changes = dict(hidden_dim=hidden, dropout=dropout, lr=lr)
    if seeds is not None:
        changes["seeds"] = list(seeds)
    return dataclasses.replace(PROBE, **changes)
