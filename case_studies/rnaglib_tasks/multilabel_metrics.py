#!/usr/bin/env python3
"""
multilabel_metrics.py

Compute multilabel classification metrics: 
- F1 (macro/micro/weighted/samples + per-label)
- ROC AUC (OvR micro/macro/weighted + per-label)
- Jaccard index (macro/micro/weighted/samples)
- Subset Accuracy (exact match)

Expected inputs:
- y_true: numpy array of shape (N, L) with {0,1} indicators for each of L labels
- y_pred: numpy array of shape (N, L) with {0,1} predictions (optional if y_proba given)
- y_proba: numpy array of shape (N, L) with independent probabilities or scores per label (optional)

Usage (as a library):
    from multilabel_metrics import compute_multilabel_metrics
    metrics = compute_multilabel_metrics(y_true, y_pred=None, y_proba=proba, label_names=None, threshold=0.5)

Author: ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    roc_auc_score,
    classification_report,
    accuracy_score,  # used for subset accuracy via exact match
    average_precision_score
)


@dataclass
class MultilabelMetricsResult:
    # Global metrics
    subset_accuracy: float  # exact match ratio

    # F1
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    f1_samples: float

    # Jaccard
    jaccard_macro: float
    jaccard_micro: float
    jaccard_weighted: float
    jaccard_samples: float

    # Per-label summaries (non-optional)
    per_label: Dict[Any, Dict[str, float]]  # precision/recall/f1/support/jaccard per label

    # Optional ROC-AUC (OvR)
    roc_auc_ovr_macro: Optional[float] = None
    roc_auc_ovr_weighted: Optional[float] = None
    roc_auc_ovr_micro: Optional[float] = None

    # Optional AUPRC
    auprc_macro: Optional[float] = None
    auprc_weighted: Optional[float] = None
    auprc_micro: Optional[float] = None

    # Optional per-label AUCs and AUPRCs
    per_label_ovr_auc: Optional[Dict[Any, Optional[float]]] = None
    per_label_auprc: Optional[Dict[Any, Optional[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_and_prepare(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_proba: Optional[np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate shapes/dtypes and derive y_pred from y_proba if needed.

    Returns
    -------
    y_true, y_pred, y_proba (possibly clipped), with shapes (N, L)
    """
    y_true = np.asarray(y_true)
    if y_true.ndim != 2:
        raise ValueError("y_true must be a 2D array of shape (N, L) with {0,1} indicators.")
    N, L = y_true.shape

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.shape != (N, L):
            raise ValueError("y_proba must have shape (N, L) and match y_true.")
        # probabilities/scores are per-label and independent in multilabel; do NOT row-normalize
        # clamp to [0,1] if they are probabilities; safe for scores as well
        y_proba = np.clip(y_proba, 0.0, 1.0)

    if y_pred is None:
        if y_proba is None:
            raise ValueError("Provide y_pred or y_proba. If only y_proba is given, y_pred is thresholded from it.")
        y_pred = (y_proba >= float(threshold)).astype(int)
    else:
        y_pred = np.asarray(y_pred)
        if y_pred.shape != (N, L):
            raise ValueError("y_pred must have shape (N, L) and match y_true.")
        # ensure binary
        y_pred = (y_pred > 0).astype(int)

    # ensure binary y_true
    y_true = (y_true > 0).astype(int)

    return y_true, y_pred, y_proba


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[Sequence[Any]] = None,
    threshold: float = 0.5,
) -> MultilabelMetricsResult:
    """
    Compute multilabel metrics given indicator ground truth, predicted labels, and optional per-label probabilities.

    Parameters
    ----------
    y_true : (N, L) array-like of {0,1}
    y_pred : (N, L) array-like of {0,1}, optional (derived from y_proba if None)
    y_proba : (N, L) array-like of floats in [0,1], optional
    label_names : sequence of names for the L labels (for reporting keys)
    threshold : float in [0,1], used to convert y_proba to y_pred if y_pred is None

    Returns
    -------
    MultilabelMetricsResult
    """
    y_true, y_pred, y_proba = _validate_and_prepare(y_true, y_pred, y_proba, threshold)
    N, L = y_true.shape

    # Resolve label names
    if label_names is None:
        label_names = list(range(L))
    else:
        if len(label_names) != L:
            raise ValueError("label_names length must equal number of labels (L).")
        label_names = list(label_names)

    # Subset (exact match) accuracy
    subset_acc = float(accuracy_score(y_true, y_pred))

    # F1 (multilabel-aware averaging)
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_samples = float(f1_score(y_true, y_pred, average="samples", zero_division=0))

    # Jaccard (a.k.a. IoU)
    jacc_micro = float(jaccard_score(y_true, y_pred, average="micro", zero_division=0))
    jacc_macro = float(jaccard_score(y_true, y_pred, average="macro", zero_division=0))
    jacc_weighted = float(jaccard_score(y_true, y_pred, average="weighted", zero_division=0))
    jacc_samples = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))

    # Per-label precision/recall/F1/support
    # classification_report works on multilabel indicator matrices.
    report = classification_report(
        y_true,
        y_pred,
        target_names=[str(x) for x in label_names],
        output_dict=True,
        zero_division=0,
    )

    # Build per-label dict and add per-label Jaccard
    per_label: Dict[Any, Dict[str, float]] = {}
    for j, name in enumerate(label_names):
        key = str(name)
        if key in report:
            # jaccard for the binary problem of label j
            jac_j = float(jaccard_score(y_true[:, j], y_pred[:, j], average="binary", zero_division=0))
            per_label[name] = {
                "precision": float(report[key]["precision"]),
                "recall": float(report[key]["recall"]),
                "f1": float(report[key]["f1-score"]),
                "support": float(report[key]["support"]),
                "jaccard": jac_j,
            }
        else:
            per_label[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0, "jaccard": 0.0}

    # ROC AUC (OvR) and AUPRC — only if y_proba provided; per-label + aggregates
    auc_macro = auc_weighted = auc_micro = None
    auprc_macro = auprc_weighted = auprc_micro = None
    per_label_auc: Optional[Dict[Any, Optional[float]]] = None
    per_label_auprc: Optional[Dict[Any, Optional[float]]] = None
    if y_proba is not None:
        per_label_auc = {}
        per_label_auprc = {}
        for j, name in enumerate(label_names):
            yt = y_true[:, j]
            yp = y_proba[:, j]
            # AUC only defined if both classes present
            if np.unique(yt).size == 2:
                try:
                    auc_j = float(roc_auc_score(yt, yp))
                    auprc_j = float(average_precision_score(yt, yp))
                except ValueError:
                    auc_j = None
                    auprc_j = None
            else:
                auc_j = None
                auprc_j = None
            per_label_auc[name] = auc_j
            per_label_auprc[name] = auprc_j

        # Compute aggregate AUPRC metrics
        auprc_values = np.array([per_label_auprc[name] for name in label_names if per_label_auprc[name] is not None])
        auprc_weights = np.array([y_true[:, j].sum() for j, name in enumerate(label_names) if per_label_auprc[name] is not None])

        if len(auprc_values) > 0:
            auprc_macro = float(np.mean(auprc_values))
            auprc_weighted = float(np.average(auprc_values, weights=auprc_weights)) if auprc_weights.sum() > 0 else None
        else:
            auprc_macro = None
            auprc_weighted = None

        try:
            auprc_micro = float(average_precision_score(y_true.ravel(), y_proba.ravel()))
        except ValueError:
            auprc_micro = None

        # Aggregates
        try:
            auc_macro = float(roc_auc_score(y_true, y_proba, average="macro"))
        except ValueError:
            auc_macro = None
        try:
            auc_weighted = float(roc_auc_score(y_true, y_proba, average="weighted"))
        except ValueError:
            auc_weighted = None
        try:
            auc_micro = float(roc_auc_score(y_true, y_proba, average="micro"))
        except ValueError:
            auc_micro = None

    return MultilabelMetricsResult(
        subset_accuracy=subset_acc,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        f1_weighted=f1_weighted,
        f1_samples=f1_samples,
        jaccard_macro=jacc_macro,
        jaccard_micro=jacc_micro,
        jaccard_weighted=jacc_weighted,
        jaccard_samples=jacc_samples,
        roc_auc_ovr_macro=auc_macro,
        roc_auc_ovr_weighted=auc_weighted,
        roc_auc_ovr_micro=auc_micro,
        per_label=per_label,
        per_label_ovr_auc=per_label_auc,
        per_label_auprc=per_label_auprc,
        auprc_macro=auprc_macro,
        auprc_weighted=auprc_weighted,
        auprc_micro=auprc_micro,
    )


# Optional: tiny demo using random data; only runs if you execute this file directly.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 120
    L = 6
    # Random sparse ground truth
    y_true = (rng.random((N, L)) < 0.2).astype(int)
    # Probabilities with some correlation to truth
    y_proba = rng.random((N, L)) * 0.7 + 0.3 * y_true
    y_proba = np.clip(y_proba, 0.0, 1.0)

    res = compute_multilabel_metrics(y_true, y_proba=y_proba, threshold=0.5,
                                     label_names=[f"label_{i}" for i in range(L)])
    import json
    print(json.dumps(res.to_dict(), indent=2))
