"""``atomica.probe`` -- the frozen-representation protocol, in one place.

Every frozen benchmark in the paper uses this recipe:

    representation   the ``z`` family from :mod:`atomica.representations`: ``z_graph`` for graph-
                     and pocket-level tasks, ``z_block`` for residue-level ones
    pooling          ``mean_std_global``, parameter-free, so the head is the only fitted component
    preprocessing    train-fit z-score (:mod:`atomica.probe.features`)
    head             :class:`AtomicaProbeHead`, the frozen-baseline MLP plus BatchNorm
    training         N seeds, early stopping on the validation metric, probabilities ensembled
    fusion           late fusion: average the probabilities of two independently trained probes

:func:`train_probe` answers whether a head built on a representation can do the task.
:func:`fit_linear_probe` fits a logistic regression with no hidden layer and answers whether the
representation encodes the quantity as a direction. Everything under either head is shared, so a
difference between them is a difference between heads.

:func:`cluster_bootstrap_ci` and :func:`paired_cluster_bootstrap` resample groups rather than
points, for benchmarks whose evaluation points are not independent, such as several residues from
one structure.

The invariant descriptors themselves live with the model, in
:mod:`atomica.models.atomica.invariants`.
"""

from .extract import (GRAPH_LEVEL_TASKS, RESIDUE_LEVEL_TASKS, extract_block_features,
                      extract_graph_features)
from .features import (FEATURE_SETS, Z_BLOCK_COMPONENTS, apply_standardizer,
                       build_features, fit_standardizer, l2_normalize, pool_saved_blocks,
                       split_z_block, standardize)
from .fusion import fuse_probe_outputs, late_fusion, select_weight_on_validation
from .head import AtomicaProbeHead, num_outputs
from .linear import LinearProbeConfig, fit_linear_probe, majority_baseline, one_hot
from .metrics import (HARD_LABEL_METRICS, SeedStats, bootstrap_ci, cluster_bootstrap_ci,
                      hard_label_metrics, mcnemar, metrics_from_prob, paired_bootstrap,
                      paired_cluster_bootstrap, probabilities_from_logits, seed_stats)
from .training import ProbeConfig, load_probe, make_loss, save_probe, train_one_seed, train_probe

__all__ = [
    # extraction -- the seam onto atomica.representations
    "extract_graph_features", "extract_block_features",
    "GRAPH_LEVEL_TASKS", "RESIDUE_LEVEL_TASKS",
    # features
    "Z_BLOCK_COMPONENTS", "FEATURE_SETS", "build_features", "split_z_block",
    "pool_saved_blocks", "standardize", "fit_standardizer", "apply_standardizer", "l2_normalize",
    # heads -- the MLP rung and the linear rung
    "AtomicaProbeHead", "num_outputs",
    "LinearProbeConfig", "fit_linear_probe", "majority_baseline", "one_hot",
    # training
    "ProbeConfig", "train_probe", "train_one_seed", "make_loss", "save_probe", "load_probe",
    # metrics
    "SeedStats", "seed_stats", "metrics_from_prob", "probabilities_from_logits",
    "HARD_LABEL_METRICS", "hard_label_metrics",
    "bootstrap_ci", "paired_bootstrap", "mcnemar",
    "cluster_bootstrap_ci", "paired_cluster_bootstrap",
    # fusion
    "late_fusion", "select_weight_on_validation", "fuse_probe_outputs",
]
