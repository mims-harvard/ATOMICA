"""Integrity checks for the ATP/ADP benchmark data and evaluation.

No GPU, no checkpoint and no encoder pass: these read the shipped tables and exercise the readout,
so they are cheap enough to run on every change.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

TUTORIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tutorials", "7_atp_adp_nucleotide_state")

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(TUTORIAL, "data")),
                                reason="tutorial 8 data is not present")


@pytest.fixture(scope="module")
def benchmark():
    sys.path.insert(0, TUTORIAL)
    import atp_adp

    return atp_adp


def test_benchmark_composition(benchmark):
    """404 pockets, 223 ATP and 181 ADP, from 120 PDB entries in 60 sequence clusters."""
    pockets = benchmark.load_pockets()
    assert len(pockets) == 404
    assert int((pockets.label == 1).sum()) == 223
    assert int((pockets.label == 0).sum()) == 181
    assert pockets.cluster.nunique() == 60
    assert pockets.pdb_entry.nunique() == 120


def test_every_cluster_holds_both_states(benchmark):
    """The construction guarantees it, and it is what makes the within-cluster readout defined.

    A cluster with one state contributes no AUROC. If this ever fails, the evaluable-cluster counts
    below move and the readout silently rests on fewer proteins.
    """
    pockets = benchmark.load_pockets()
    per_cluster = pockets.groupby("cluster").label.nunique()
    assert set(per_cluster) == {2}, f"{int((per_cluster < 2).sum())} clusters hold a single state"


def test_pockets_and_graphs_line_up(benchmark):
    """Every pocket has a graph, in the same order, and every graph's shared site resolves."""
    pockets, graphs = benchmark.load_pockets(), benchmark.load_graphs()
    assert list(graphs.id) == list(pockets.id)
    positions = benchmark.shared_site_positions(graphs, pockets)
    assert len(positions) == len(pockets)
    # the shared site is a subset of the 50-residue pocket and never reaches the global block
    for pocket_id, pos in positions.items():
        assert pos.min() >= 1, f"{pocket_id}: a shared residue mapped onto the global block"
        assert len(pos) >= 5










def test_strata_are_nested_subsets(benchmark):
    """Each narrower stratum is a subset of the one above it. If it is not, the four panels are
    not a re-partition of one set of predictions and cannot be read as such."""
    pockets = benchmark.load_pockets().set_index("id")
    conc = set(pockets.index[pockets.stratum_metal_concordant])
    assert set(pockets.index[pockets.stratum_with_metal]) <= conc
    assert set(pockets.index[pockets.stratum_metal_free]) <= conc
    assert set(pockets.index[pockets.stratum_held_out]) <= conc
    # with-metal and metal-free partition the concordant set
    both = set(pockets.index[pockets.stratum_with_metal])
    free = set(pockets.index[pockets.stratum_metal_free])
    assert not (both & free)
    assert both | free == conc


def test_cluster_bootstrap_brackets_the_point_estimate(benchmark):
    """A percentile interval over the clusters has to contain the mean it resamples."""
    rng = np.random.default_rng(0)
    for n in (9, 25, 60):
        per = rng.uniform(0.3, 1.0, size=n)
        lo, hi = benchmark.cluster_bootstrap(per)
        assert lo <= per.mean() <= hi
        assert lo < hi


def test_cluster_macro_auroc_ignores_class_balance(benchmark):
    """Chance is 0.500 in every stratum because AUROC inside a cluster does not depend on that
    cluster's prevalence. A random score must sit near 0.5 whatever the balance."""
    rng = np.random.default_rng(0)
    labels, scores, clusters = [], [], []
    for c, n_pos in enumerate([1, 5, 9]):
        for k in range(10):
            labels.append(1 if k < n_pos else 0)
            scores.append(rng.random())
            clusters.append(f"c{c}")
    value, per, used = benchmark.cluster_macro_auroc(labels, scores, clusters)
    assert len(used) == 3
    assert abs(value - 0.5) < 0.25

    # a cluster holding one state contributes nothing
    value, per, used = benchmark.cluster_macro_auroc(
        labels + [1] * 4, scores + list(rng.random(4)), clusters + ["lonely"] * 4)
    assert "lonely" not in used and len(per) == 3


def test_batching_cannot_change_the_features(benchmark):
    """The batch settings must be a combination that gives every pocket its batch-of-one value.

    ATOMICA's per-block cross-attention pads each block out to the largest block in the batch, so
    naive batching above size 1 makes a pocket's descriptor depend on what else was in its batch.
    Grouping by largest block removes that: every structure is embedded at the pad width it would
    have had alone. So either the batch size is 1, or the grouping is on. Anything else is a way to
    produce features that depend on how the file happens to be ordered.
    """
    assert benchmark.EXTRACT_BATCH_SIZE >= 1
    assert benchmark.EXTRACT_BATCH_SIZE == 1 or benchmark.GROUP_BY_MAX_BLOCK is True


def test_grouped_batches_share_a_largest_block(benchmark):
    """The property the whole batching argument rests on, checked on this benchmark's own graphs.

    ``atom_budget`` may split a group, which is harmless; it may never merge two. So every batch
    must be internally uniform in largest block, and the indices must cover the dataset exactly
    once so the rows can be put back in the order ``data/pockets.csv`` uses.
    """
    from atomica import representations as R
    import tutorial as runner

    items = runner._items(benchmark.load_graphs())
    batches = R.group_batches(items, benchmark.EXTRACT_BATCH_SIZE,
                              group_by_max_block=benchmark.GROUP_BY_MAX_BLOCK,
                              atom_budget=benchmark.ATOM_BUDGET)
    seen = []
    for indices, chunk in batches:
        assert len(indices) == len(chunk)
        assert len(chunk) <= benchmark.EXTRACT_BATCH_SIZE
        widths = {int(max(item["data"]["block_lengths"])) for item in chunk}
        assert len(widths) == 1, f"a batch mixes largest blocks {sorted(widths)}"
        seen.extend(indices)
    assert sorted(seen) == list(range(len(items))), "batches do not cover the pockets exactly once"




def test_pooling_uses_the_shared_site_only(benchmark):
    """The non-shared residues must not enter the pool.

    They stay in the forward pass, because the encoder needs the whole pocket to represent any part
    of it, and are excluded when the block vectors are averaged. Pooling all 50 instead scores
    higher, so this default is what stops the benchmark reporting an inflated number.
    """
    assert benchmark.POOL_DEFAULT == "shared_site"
    pockets = benchmark.load_pockets()
    graphs = benchmark.load_graphs()
    positions = benchmark.shared_site_positions(graphs, pockets)

    # the pooled set is a strict subset of the 50-residue pocket for all but a handful, and every
    # pooled block is one the shared site names
    n_pooled = np.array([len(positions[i]) for i in pockets.id])
    assert (n_pooled <= 50).all()
    assert (n_pooled < 50).sum() >= 400
    by_id = graphs.set_index("id")
    for pocket_id, site in list(zip(pockets.id, pockets.shared_site))[:40]:
        row = by_id.loc[pocket_id]
        index = benchmark.residue_block_index(row.block_to_pdb_indexes, len(row.B))
        allowed = {index[k] for k in str(site).split(";") if k in index}
        assert set(positions[pocket_id].tolist()) == allowed


def test_probe_config_is_the_documented_protocol(benchmark):
    """The training constants the README documents. Changing one changes the result."""
    cfg = benchmark.PROBE
    assert cfg.epochs == 60 and cfg.batch_size == 32
    assert cfg.final_hidden_dim == 32 and cfg.weight_decay == 0.0
    assert len(cfg.seeds) == 5
    assert cfg.use_batchnorm is False
    assert cfg.early_stopping is False, "a fixed 60-epoch budget, not a best-validation checkpoint"
    assert benchmark.FOLDS == 5
    assert sorted(benchmark.grid()) == sorted(
        [(h, d, lr) for h in (64, 256) for d in (0.1, 0.3) for lr in (1e-4, 1e-3)])




def test_folds_are_cluster_disjoint(benchmark):
    """No cluster may appear in two folds. This is the whole design of the split."""
    pockets = benchmark.load_pockets()
    folds = benchmark.fold_assignment(pockets.fold_unit.to_numpy())
    for cluster in set(pockets.fold_unit):
        assigned = set(folds[pockets.fold_unit.to_numpy() == cluster])
        assert len(assigned) == 1, f"cluster {cluster} spans folds {assigned}"
    assert len(set(folds)) == benchmark.FOLDS


def test_fold_standardize_drops_constant_columns(benchmark):
    """The 1e-6 column filter, and the z-score fit on training rows only."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    X[:, 3] = 7.0                                   # constant everywhere
    fit = np.zeros(40, dtype=bool)
    fit[:20] = True
    train, held = benchmark.fold_standardize(X, fit, fit, ~fit)
    assert train.shape[1] == 5 and held.shape[1] == 5
    assert np.allclose(train.mean(0), 0, atol=1e-5)
    assert np.allclose(train.std(0), 1, atol=1e-3)
