"""The linear rung of the probe: multinomial logistic regression, no hidden layer.

`atomica.probe.head` trains an MLP on a frozen representation, which answers "can a head built on
this representation do the task?". This answers the stricter "does the representation encode the
quantity as a direction?". An MLP can manufacture an answer out of a representation that merely
fails to destroy the information, so a claim about the representation itself needs this head.

Everything else is shared with the MLP rung: the z representation, the train-fit z-score, the
metrics, and model selection on validation only.

The regularization strength is swept rather than fixed because it is not transferable: the
descriptors here range from 32 to 5,376 dimensions over the same rows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .metrics import hard_label_metrics

__all__ = ["LinearProbeConfig", "fit_linear_probe", "majority_baseline", "one_hot"]


@dataclass
class LinearProbeConfig:
    """Hyperparameters of the linear probe. Only ``C`` is fitted, and only on validation."""

    #: Inverse L2 strength. Swept, never assumed -- see the module docstring.
    C_grid: Sequence[float] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    #: lbfgs iteration cap. Reaching it is reported, not swallowed: an under-fitted probe
    #: understates the representation, which is the direction that would fake a negative result.
    max_iter: int = 2000
    #: 1e-3 rather than sklearn's 1e-4. At 5,376 dimensions the last decade of tolerance costs a
    #: large multiple of the runtime and moves balanced accuracy in the fourth decimal.
    tol: float = 1e-3
    #: Fit the z-score on train and apply it to every split. Off only if the caller already did it.
    standardize: bool = True
    seed: int = 0
    n_jobs: int = -1
    #: Validation metric that chooses ``C``. A key of :func:`atomica.probe.metrics.hard_label_metrics`.
    primary: str = "balanced_acc"


def one_hot(values: Sequence, levels: Sequence) -> np.ndarray:
    """Categorical column -> indicator matrix over ``levels``, in the order given.

    ``levels`` is passed in rather than inferred per split so train, validation and test share one
    column layout even when a level is missing from one of them.
    """
    values = np.asarray(values)
    return np.stack([(values == level).astype(float) for level in levels], axis=1)


def majority_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """The most frequent training class, repeated. The floor every other arm is read against."""
    values, counts = np.unique(np.asarray(y_train), return_counts=True)
    return np.repeat(values[counts.argmax()], n_test)


def fit_linear_probe(X_tr, y_tr, X_va, y_va, X_te, *,
                     cfg: Optional[LinearProbeConfig] = None) -> Dict:
    """Sweep ``C`` on validation, refit nothing, and return the winner's test predictions.

    Returns a dict with ``test_pred`` and ``val_pred`` (hard labels, dtype of ``y_tr``),
    ``test_prob`` / ``classes`` (the softmax and its column order), the selected ``C``, the
    validation score that selected it, ``hit_max_iter`` and the input width.

    The test split is never consulted here. It is scored once by the caller, after selection.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    cfg = cfg or LinearProbeConfig()
    X_tr, X_va, X_te = (np.asarray(a, dtype=np.float64) for a in (X_tr, X_va, X_te))
    y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)

    if cfg.standardize:
        from .features import apply_standardizer, fit_standardizer
        mu, sd = fit_standardizer(X_tr)
        X_tr, X_va, X_te = (apply_standardizer(a, mu, sd) for a in (X_tr, X_va, X_te))

    best, best_score, best_C, hit_max = None, -np.inf, cfg.C_grid[0], False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for C in cfg.C_grid:
            clf = LogisticRegression(C=C, max_iter=cfg.max_iter, tol=cfg.tol,
                                     n_jobs=cfg.n_jobs, random_state=cfg.seed)
            clf.fit(X_tr, y_tr)
            score = hard_label_metrics(y_va, clf.predict(X_va))[cfg.primary]
            if score > best_score:
                best, best_score, best_C = clf, score, C
                hit_max = bool(np.max(clf.n_iter_) >= cfg.max_iter)

    return {"test_pred": best.predict(X_te), "val_pred": best.predict(X_va),
            "test_prob": best.predict_proba(X_te), "classes": list(best.classes_),
            "C": float(best_C), "val_primary": float(best_score),
            "primary": cfg.primary, "hit_max_iter": hit_max, "dim": int(X_tr.shape[1])}
