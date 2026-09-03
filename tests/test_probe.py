"""The probe consumes the z family and produces a trainable feature matrix.

Skipped unless the pretrain checkpoint and the processed example data are both present.
"""
import os

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain")
CONFIG = os.path.join(CKPT, "pretrain_model_config.json")
WEIGHTS = os.path.join(CKPT, "pretrain_model_weights.pt")
DATA = os.path.join(REPO, "data", "example", "example_processed_data.parquet")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CONFIG) and os.path.exists(WEIGHTS) and os.path.exists(DATA)),
    reason="needs the pretrain checkpoint and the processed example data",
)


@pytest.fixture(scope="module")
def model_and_batch():
    from atomica.models.prediction_model import PredictionModel
    from atomica.data.dataset import PDBDataset
    model = PredictionModel.load_from_config_and_weights(CONFIG, WEIGHTS)
    model.eval()
    ds = PDBDataset(DATA)
    return model, ds


def test_feature_sets_use_the_representation_vocabulary():
    from atomica import probe
    assert probe.Z_BLOCK_COMPONENTS == ("h_block", "gram", "atom")
    assert probe.FEATURE_SETS["z_block"] == ("h_block", "gram", "atom")
    # the feature sets are nested, so a narrower one is a slice of the same extraction
    assert set(probe.FEATURE_SETS["h_block"]) < set(probe.FEATURE_SETS["z_block_gram"])
    assert set(probe.FEATURE_SETS["z_block_gram"]) < set(probe.FEATURE_SETS["z_block"])


def test_component_names_match_the_model(model_and_batch):
    """The split must never drift from how the model concatenated the vector."""
    from atomica import probe
    model, _ = model_and_batch
    dims = model.invariant_component_dims()
    assert tuple(dims) == probe.Z_BLOCK_COMPONENTS
    assert dims == {"h_block": 32, "gram": 544, "atom": 1216}


def test_split_and_rebuild_is_lossless(model_and_batch):
    from atomica import probe, representations as R
    model, ds = model_and_batch
    batch = ds.collate_fn([ds[0]])
    with torch.no_grad():
        z = R.get(model, batch, "z_block").numpy()
    parts = probe.split_z_block(z, model.invariant_component_dims())
    assert parts["h_block"].shape[1] == 32
    assert parts["gram"].shape[1] == 544
    assert parts["atom"].shape[1] == 1216
    assert np.allclose(probe.build_features(parts, "z_block"), z)
    assert probe.build_features(parts, "h_block").shape[1] == 32
    assert probe.build_features(parts, "z_block_gram").shape[1] == 576


def test_extract_graph_features(model_and_batch):
    from atomica import probe
    model, ds = model_and_batch
    n = min(3, len(ds))
    batches = [ds.collate_fn([ds[i]]) for i in range(n)]
    X, ids = probe.extract_graph_features(model, batches)
    assert X.shape == (n, 3 * 1792)
    assert np.isfinite(X).all()


def test_extract_block_features(model_and_batch):
    from atomica import probe
    model, ds = model_and_batch
    batches = [ds.collate_fn([ds[i]]) for i in range(min(2, len(ds)))]
    X, gid = probe.extract_block_features(model, batches)
    assert X.shape[1] == 1792
    assert X.shape[0] == gid.shape[0]
    assert sorted(set(gid.tolist())) == list(range(len(batches)))


def test_saved_and_live_pooling_agree(model_and_batch):
    """The numpy path on saved arrays must match the torch path inside the model."""
    from atomica import probe, representations as R
    model, ds = model_and_batch
    batch = ds.collate_fn([ds[0]])
    with torch.no_grad():
        live = R.get(model, batch, "z_graph", pool="mean_std_global").numpy()
        z_block = R.get(model, batch, "z_block").numpy()
    is_global = (batch["B"] == model.global_block_id).numpy()
    keys = np.zeros(len(z_block), dtype=int)
    _, saved = probe.pool_saved_blocks(z_block, keys, is_global, mode="mean_std_global")
    assert np.allclose(live, saved, atol=1e-4), np.abs(live - saved).max()


def test_saved_component_normalized_matches(model_and_batch):
    from atomica import probe, representations as R
    model, ds = model_and_batch
    batch = ds.collate_fn([ds[0]])
    with torch.no_grad():
        live = R.get(model, batch, "z_graph", pool="mean_component_normalized").numpy()
        z_block = R.get(model, batch, "z_block").numpy()
    is_global = (batch["B"] == model.global_block_id).numpy()
    keys = np.zeros(len(z_block), dtype=int)
    _, saved = probe.pool_saved_blocks(z_block, keys, is_global,
                                       mode="mean_component_normalized",
                                       component_dims=model.invariant_component_dims())
    assert np.allclose(live, saved, atol=1e-4), np.abs(live - saved).max()


def test_head_trains_on_extracted_features(model_and_batch):
    """A short end-to-end run on synthetic labels: the head accepts the real feature width."""
    from atomica import probe
    model, ds = model_and_batch
    n = min(4, len(ds))
    batches = [ds.collate_fn([ds[i]]) for i in range(n)]
    X, _ = probe.extract_graph_features(model, batches)
    X = np.repeat(X, 4, axis=0)
    y = np.array([0, 1] * (len(X) // 2))
    Xs, = probe.standardize(X)
    cfg = probe.ProbeConfig(epochs=2, patience=2, batch_size=4, seeds=[0], hidden_dim=16)
    out = probe.train_probe(Xs, y, Xs, y, Xs, y, task_type="multiclass",
                            primary="accuracy", cfg=cfg, device="cpu")
    assert out["dim"] == 3 * 1792
    assert "ensemble" in out and "accuracy" in out["ensemble"]


def test_old_component_name_gives_a_useful_error(model_and_batch):
    from atomica import probe
    with pytest.raises(KeyError, match="block_repr is now h_block"):
        probe.build_features({"block_repr": np.zeros((2, 32), dtype=np.float32)}, "z_block")


# --------------------------------------------------------------------------- the linear rung
# These need neither the checkpoint nor the example data, so they run on synthetic arrays.
def test_linear_probe_selects_C_on_validation_only():
    """The chosen C is the one with the best VALIDATION score, and test is never consulted."""
    from atomica.probe import LinearProbeConfig, fit_linear_probe
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 12))
    y = np.array(["a", "b", "c"])[(X @ rng.normal(size=(12, 3))).argmax(1)]
    cfg = LinearProbeConfig(C_grid=(0.001, 1.0), max_iter=500)
    out = fit_linear_probe(X[:200], y[:200], X[200:300], y[200:300], X[300:], cfg=cfg)
    assert out["C"] in cfg.C_grid
    assert out["dim"] == 12
    assert len(out["test_pred"]) == 100
    # predict and predict_proba have to agree, since the reported metrics come from the hard
    # labels and any saved probabilities have to describe the same classifier
    assert (np.asarray(out["classes"])[out["test_prob"].argmax(1)] == out["test_pred"]).all()


def test_linear_probe_has_no_hidden_layer():
    """Guards the claim the probe rests on: a linear map, so a positive result is about the
    representation rather than about what a head can build from it."""
    import inspect

    from atomica.probe import linear
    source = inspect.getsource(linear.fit_linear_probe)
    assert "LogisticRegression" in source
    assert "hidden" not in source.lower()


def test_hard_label_metrics_match_the_probability_ones():
    """One results table has to be able to hold both heads, so the names and values must agree."""
    from atomica.probe import hard_label_metrics, metrics_from_prob
    y = np.array([0, 1, 2, 1, 0, 2, 2, 1])
    prob = np.eye(3)[np.array([0, 1, 2, 2, 0, 2, 1, 1])]
    hard = hard_label_metrics(y, prob.argmax(1))
    soft = metrics_from_prob("multiclass", y, prob)
    for key in ("balanced_acc", "accuracy", "f1_macro"):
        assert hard[key] == pytest.approx(soft[key])


def test_cluster_bootstrap_is_wider_than_resampling_points():
    """The reason the clustered version exists: when points inside a cluster are perfectly
    correlated the effective sample size is the number of CLUSTERS, and resampling points
    instead returns an interval that is far too narrow."""
    from atomica.probe import bootstrap_ci, cluster_bootstrap_ci
    rng = np.random.default_rng(0)
    n_clusters, per_cluster = 40, 25
    y_cluster = rng.integers(0, 2, n_clusters)
    right = rng.random(n_clusters) < 0.7          # a whole cluster is right or wrong together
    y = np.repeat(y_cluster, per_cluster)
    pred = np.repeat(np.where(right, y_cluster, 1 - y_cluster), per_cluster)
    groups = np.repeat(np.arange(n_clusters), per_cluster)

    point, clo, chi = cluster_bootstrap_ci(y, pred, groups, "accuracy", n_boot=500)
    _, plo, phi = bootstrap_ci("multiclass", y, np.eye(2)[pred], "accuracy", n_boot=500)
    assert point == pytest.approx(right.mean())
    assert (chi - clo) > 3 * (phi - plo)


def test_paired_cluster_bootstrap_is_paired():
    """Two identical arms differ by exactly zero in every resample, which an unpaired
    comparison could not show."""
    from atomica.probe import paired_cluster_bootstrap
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    pred = rng.integers(0, 3, 200)
    out = paired_cluster_bootstrap(y, pred, pred, rng.integers(0, 20, 200), n_boot=200)
    assert out["delta"] == 0.0 and out["lo"] == 0.0 and out["hi"] == 0.0
    assert not out["significant"]


def test_majority_baseline_and_one_hot():
    from atomica.probe import hard_label_metrics, majority_baseline, one_hot
    y_train = np.array(["x"] * 10 + ["y"] * 3)
    pred = majority_baseline(y_train, 5)
    assert list(pred) == ["x"] * 5
    # a constant prediction scores exactly 1/k balanced accuracy, whatever the prior is
    y_test = np.array(["x", "x", "y", "y", "y"])
    assert hard_label_metrics(y_test, pred)["balanced_acc"] == pytest.approx(0.5)
    # levels are supplied so every split shares one column layout
    assert one_hot(["Zn", "Mg"], ["Ca", "Mg", "Zn"]).tolist() == [[0, 0, 1], [0, 1, 0]]
