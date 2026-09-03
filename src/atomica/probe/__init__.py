"""``atomica.probe`` -- the frozen-representation protocol, in one place.

Every frozen benchmark in the paper uses this recipe, so that "how do you get a representation out
of ATOMICA and train on it?" has one answer rather than one per benchmark.

    representation   the ``z`` family from :mod:`atomica.representations`.
                     ``z_graph`` for graph- and pocket-level tasks, ``z_block`` for residue-level
                     ones. Extracted by :mod:`atomica.probe.extract`.
    pooling          ``mean_std_global``, fixed by the Methods because a head is trained on top.
                     Parameter-free, so the head stays the only fitted component.
    preprocessing    train-fit z-score (:mod:`atomica.probe.features`)
    head             :class:`AtomicaProbeHead`, the frozen-baseline MLP plus BatchNorm
    training         N seeds, early stopping on the task's validation metric, probabilities
                     ensembled across seeds
    fusion           late fusion: average the probabilities of two independently trained probes

**Two heads, answering two different questions.** :func:`train_probe` fits the MLP above and
answers "can a head built on this representation do the task?", which is the right question for a
benchmark. :func:`fit_linear_probe` (:mod:`atomica.probe.linear`) fits an L2-penalized multinomial
logistic regression with no hidden layer and answers "does the representation encode the quantity
as a direction?", which is the right question for a claim about the representation itself. An MLP
can manufacture an answer out of a representation that merely fails to destroy the information, so
a statement about what the embedding *contains* has to be made with the linear head. Everything
either head sits on -- representation, pooling, z-score, metrics, model selection on validation
only -- is shared, so a difference between them is a difference between heads.

**Clustered intervals.** :func:`cluster_bootstrap_ci` and :func:`paired_cluster_bootstrap`
resample groups rather than points, for benchmarks whose evaluation points are not independent
(several residues from one structure, several metal sites from one PDB entry). Resampling points
there returns an interval that is too narrow.

**Why ``z`` and not ``h``.** The ``h`` family is built from the ``lambda = 0`` channels alone and
discards every ``l > 0`` channel, which is most of the geometry the encoder computed. ``z`` turns
those channels into rotation invariants a non-equivariant head can use. On MaSIF-ligand the
difference is large: 0.589 accuracy from the 32-d ``h_block`` readout against 0.837 from the full
1792-d ``z_block``.

**Why the head is the only trained part.** The frozen baselines get a plain mean over residues and
an MLP. Giving ATOMICA a learned pooler on top of that would make the comparison a comparison of
set-aggregation rather than of representations. The pooling here is parameter-free for that reason,
and it costs nothing measurable.

The invariant descriptors themselves live with the model, in
:mod:`atomica.models.atomica.invariants`, because they depend on the irrep layout.
"""

from .extract import (GRAPH_LEVEL_TASKS, RESIDUE_LEVEL_TASKS, extract_block_features,
                      extract_graph_features)
from .features import (COMMITTED_FEATURE, FEATURE_SETS, Z_BLOCK_COMPONENTS, apply_standardizer,
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
    "Z_BLOCK_COMPONENTS", "FEATURE_SETS", "COMMITTED_FEATURE", "build_features", "split_z_block",
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
