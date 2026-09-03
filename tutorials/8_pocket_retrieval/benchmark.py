"""Scoring for the pocket retrieval benchmark. Nothing here calls ATOMICA.

A candidate pocket is relevant if it binds the query's ligand and has no detectable Foldseek
alignment to the query. A same-ligand pocket that does align is ignored rather than counted as a
negative. Values are macro-averaged over 30% sequence identity clusters.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

METRICS = ("mAP", "AUROC", "nDCG", "R_precision", "MRR", "Hit@20")

N_BOOT = 2000
N_PERM = 20000
SEED = 0


def load_benchmark(path: str = None) -> dict:
    """Pool, query set, relevant and ignored sets, and per-pocket ligand and cluster labels."""
    with open(path or os.path.join(DATA, "benchmark.json")) as handle:
        return json.load(handle)


def load_vectors(path: str):
    """(ids, matrix) with rows L2-normalized, so ranking is a plain inner product."""
    blob = np.load(path, allow_pickle=True)
    ids = [str(x) for x in blob["ids"]]
    matrix = np.asarray(blob["vectors"], dtype=np.float64)
    return ids, matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def query_layout(ids, bench: dict) -> dict:
    """Per query: its row, the candidate rows, the relevant mask and a fixed tie-break.

    Ties are broken by a permutation seeded from the query name, not by pool order, which is
    grouped by ligand and would give free signal to constant similarities.
    """
    row_of = {name: i for i, name in enumerate(ids)}
    corpus = [name for name in bench["pool"] if name in row_of]
    seat_of = {name: j for j, name in enumerate(corpus)}
    corpus_rows = np.array([row_of[name] for name in corpus])

    layout = {}
    for query in bench["queries"]:
        if query not in seat_of:
            continue
        relevant = [r for r in bench["relevant"][query] if r in seat_of]
        if not relevant:
            continue
        drop = {seat_of[query]}
        drop |= {seat_of[x] for x in bench["ignored"].get(query, ()) if x in seat_of}
        keep = np.array([j for j in range(len(corpus)) if j not in drop])
        relevant_seats = {seat_of[r] for r in relevant}
        mask = np.fromiter((j in relevant_seats for j in keep), bool, len(keep))
        if not mask.any():
            continue
        rng = np.random.default_rng(int(hashlib.sha256(query.encode()).hexdigest()[:8], 16))
        layout[query] = {
            "row": row_of[query],
            "candidate_rows": corpus_rows[keep],
            "relevant": mask,
            "tiebreak": rng.permutation(len(keep)),
        }
    return layout


def metrics_from_ranks(ranks, m: int, N: int) -> dict:
    """The six statistics from the 1-indexed ranks of the m relevant items among N candidates."""
    ranks = np.asarray(ranks, dtype=np.float64)
    ordered = np.sort(ranks)
    ideal = 1.0 / np.log2(np.arange(1, m + 1) + 1.0)
    return {
        "AUROC": float((np.sum(N - ranks) - m * (m - 1) / 2) / (m * (N - m))),
        "nDCG": float((1.0 / np.log2(ranks + 1.0)).sum() / ideal.sum()),
        "mAP": float(((np.arange(m) + 1) / ordered).mean()),
        "MRR": float(1.0 / ordered[0]),
        "R_precision": float(np.sum(ranks <= m) / m),
        "Hit@20": float(np.any(ranks <= 20)),      # a success rate, not Recall@20
    }


def score_query(similarity, relevant, tiebreak) -> dict:
    order = np.lexsort((tiebreak, -np.asarray(similarity, dtype=np.float64)))
    ranked = np.asarray(relevant, dtype=bool)[order]
    m, N = int(ranked.sum()), len(ranked)
    if m == 0 or m == N:
        return {k: np.nan for k in METRICS}
    return metrics_from_ranks(np.flatnonzero(ranked) + 1, m, N)


def score_all_queries(matrix, ids, bench: dict, layout: dict = None) -> dict:
    """{query: {statistic: value}} by cosine retrieval over the pool."""
    layout = layout if layout is not None else query_layout(ids, bench)
    return {
        query: score_query(matrix[L["candidate_rows"]] @ matrix[L["row"]],
                           L["relevant"], L["tiebreak"])
        for query, L in layout.items()
    }


def macro_weights(queries, bench: dict, by: str = "cluster") -> np.ndarray:
    """Weights that make a weighted mean an equal-weight-per-cluster macro average.

    One cluster can contribute several near-duplicate query pockets, which a plain mean over
    queries would count several times.
    """
    if by is None:
        return np.ones(len(queries))
    counts = collections.Counter(bench["pockets"][q][by] for q in queries)
    return np.array([1.0 / counts[bench["pockets"][q][by]] for q in queries])


def null_distribution(m: int, N: int, n_perm: int, seed: int) -> dict:
    """E[statistic] over n_perm random rankings of m relevant items among N.

    None of these statistics has chance 0.5 at 1% prevalence, and several depend on m and N, so
    each query gets its own null. The relevant ranks under a random ranking are a uniform
    m-subset of 1..N, so the subsets are drawn directly.
    """
    rng = np.random.default_rng(seed)
    ranks = np.sort(rng.random((n_perm, N)).argpartition(m - 1, axis=1)[:, :m], axis=1) + 1.0
    ideal = float((1.0 / np.log2(np.arange(1, m + 1) + 1.0)).sum())
    return {
        "AUROC": float(np.mean((np.sum(N - ranks, axis=1) - m * (m - 1) / 2) / (m * (N - m)))),
        "nDCG": float(np.mean((1.0 / np.log2(ranks + 1.0)).sum(axis=1) / ideal)),
        "mAP": float(np.mean(((np.arange(m) + 1) / ranks).mean(axis=1))),
        "MRR": float(np.mean(1.0 / ranks[:, 0])),
        "R_precision": float(np.mean((ranks <= m).sum(axis=1) / m)),
        "Hit@20": float(np.mean((ranks <= 20).any(axis=1))),
    }


def chance_table(layout: dict, bench: dict, n_perm: int = N_PERM, seed: int = SEED,
                 by: str = "cluster"):
    """Chance value of every statistic, aggregated the same way the results are.

    Lift is a ratio, so the reference has to use the same weighting as the numerator. Nulls are
    cached by (m, N), which the queries share heavily.
    """
    cache, per_query = {}, {}
    for query, L in layout.items():
        key = (int(L["relevant"].sum()), len(L["relevant"]))
        if key not in cache:
            cache[key] = null_distribution(key[0], key[1], n_perm,
                                           seed + 1_000_003 * key[0] + key[1])
        per_query[query] = cache[key]
    queries = list(per_query)
    weights = macro_weights(queries, bench, by)
    return {key: float((np.array([per_query[q][key] for q in queries]) * weights).sum()
                       / weights.sum()) for key in METRICS}, per_query


def aggregate(per_query: dict, bench: dict, key: str, by: str = "cluster") -> float:
    queries = list(per_query)
    values = np.array([per_query[q][key] for q in queries], dtype=float)
    weights = macro_weights(queries, bench, by)
    ok = np.isfinite(values)
    if not ok.any():
        return float("nan")
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum())


def bootstrap_ci(per_query: dict, bench: dict, key: str, by: str = "cluster",
                 n_boot: int = N_BOOT, seed: int = SEED):
    """95% percentile interval, resampling queries with the macro weights carried inside."""
    rng = np.random.default_rng(seed)
    queries = list(per_query)
    values = np.array([per_query[q][key] for q in queries], dtype=float)
    weights = macro_weights(queries, bench, by)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(values), len(values))
        v, w = values[pick], weights[pick]
        ok = np.isfinite(v)
        draws[b] = (v[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else np.nan
    finite = draws[np.isfinite(draws)]
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def summarise(per_query: dict, bench: dict, chance: dict, by: str = "cluster",
              n_boot: int = N_BOOT) -> dict:
    row = {"n_queries": len(per_query)}
    for key in METRICS:
        row[key] = aggregate(per_query, bench, key, by)
        row[f"{key}_ci"] = bootstrap_ci(per_query, bench, key, by, n_boot)
    row["mAP_lift"] = row["mAP"] / chance["mAP"]
    return row


def format_row(label: str, row: dict) -> str:
    cells = []
    for key in METRICS:
        digits = 4 if row[key] < 0.1 else 3
        if f"{key}_ci" in row:
            low, high = row[f"{key}_ci"]
            cells.append(f"{row[key]:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]")
        else:
            cells.append(f"{row[key]:.{digits}f}")     # chance is exact, so no interval
    return f"{label:<26}" + "  ".join(f"{c:>24}" for c in cells)


def format_header() -> str:
    return f"{'':<26}" + "  ".join(f"{k:>24}" for k in METRICS)
