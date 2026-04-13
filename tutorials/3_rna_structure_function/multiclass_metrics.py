#!/usr/bin/env python3
"""
multiclass_metrics.py

Compute multiclass metrics: F1 (macro/micro/weighted + per-class),
ROC AUC (OvR & OvO, macro/weighted + per-class OvR),
Balanced Accuracy, Jaccard (macro/micro/weighted), and Accuracy.

Expected inputs:
- y_true: numpy array of shape (N,) with integer class labels in {0..K-1} or any set listed in `labels`
- y_pred: numpy array of shape (N,) with predicted integer class labels
- y_proba: numpy array of shape (N, K) with class probabilities or confidence scores (rows need not sum to 1; we'll normalize)

Usage (as a library):
    from multiclass_metrics import compute_multiclass_metrics
    metrics = compute_multiclass_metrics(y_true, y_pred, y_proba, labels=None)

Author: ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize


@dataclass
class MetricsResult:
    # Global metrics (non-optional)
    accuracy: float
    balanced_accuracy: float

    f1_macro: float
    f1_micro: float
    f1_weighted: float

    jaccard_macro: float
    jaccard_micro: float
    jaccard_weighted: float

    # Per-class summaries (non-optional)
    per_class: Dict[Any, Dict[str, float]]

    # Optional metrics (require probabilities)
    roc_auc_ovr_macro: Optional[float] = None
    roc_auc_ovr_weighted: Optional[float] = None
    roc_auc_ovo_macro: Optional[float] = None
    roc_auc_ovo_weighted: Optional[float] = None

    auprc_ovr_macro: Optional[float] = None
    auprc_ovr_weighted: Optional[float] = None
    auprc_ovo_macro: Optional[float] = None
    auprc_ovo_weighted: Optional[float] = None

    per_class_ovr_auc: Optional[Dict[Any, float]] = None
    per_class_auprc_ovr: Optional[Dict[Any, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L1-normalize rows of a 2D array; safe for zero rows."""
    if mat.ndim != 2:
        raise ValueError("y_proba must be a 2D array of shape (N, K).")
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < eps, 1.0, row_sums)  # avoid divide-by-zero
    return mat / row_sums


def _ensure_label_space(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence[Any]]) -> List[Any]:
    if labels is not None:
        return list(labels)
    # Use union of labels observed in y_true and y_pred, preserving sorted order if possible
    uniq = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    return list(uniq.tolist())


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
) -> MetricsResult:
    """
    Compute multiclass metrics given true labels, predicted labels, and class probabilities.

    Parameters
    ----------
    y_true : (N,) array-like of ints/labels
    y_pred : (N,) array-like of ints/labels
    y_proba : (N, K) array-like of floats, optional
        Class probabilities or scores for each class. Required for AUC. If given, rows will be normalized to sum to 1.
    labels : sequence, optional
        Full class label set and ordering. If None, inferred from y_true union y_pred.

    Returns
    -------
    MetricsResult
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays of shape (N,).")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    labels_list = _ensure_label_space(y_true, y_pred, labels)
    n_classes = len(labels_list)
    label_to_index = {lab: i for i, lab in enumerate(labels_list)}

    # Map labels to integer indices consistently for metrics that require it
    y_true_idx = np.vectorize(label_to_index.get)(y_true)
    y_pred_idx = np.vectorize(label_to_index.get)(y_pred)

    # ---- Global metrics (not requiring probabilities) ----
    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    bal_acc = float(balanced_accuracy_score(y_true_idx, y_pred_idx))

    f1_macro = float(f1_score(y_true_idx, y_pred_idx, average="macro", labels=list(range(n_classes))))
    f1_micro = float(f1_score(y_true_idx, y_pred_idx, average="micro", labels=list(range(n_classes))))
    f1_weighted = float(f1_score(y_true_idx, y_pred_idx, average="weighted", labels=list(range(n_classes))))

    jacc_macro = float(jaccard_score(y_true_idx, y_pred_idx, average="macro", labels=list(range(n_classes))))
    jacc_micro = float(jaccard_score(y_true_idx, y_pred_idx, average="micro", labels=list(range(n_classes))))
    jacc_weighted = float(jaccard_score(y_true_idx, y_pred_idx, average="weighted", labels=list(range(n_classes))))

    # Per-class F1 and Jaccard (and support) via classification_report
    report = classification_report(
        y_true_idx, y_pred_idx, labels=list(range(n_classes)), output_dict=True, zero_division=0
    )
    per_class: Dict[Any, Dict[str, float]] = {}
    for i, lab in enumerate(labels_list):
        if str(i) in report:
            per_class[lab] = {
                "precision": float(report[str(i)]["precision"]),
                "recall": float(report[str(i)]["recall"]),  # per-class recall == TPR for that class
                "f1": float(report[str(i)]["f1-score"]),
                "support": float(report[str(i)]["support"]),
                "jaccard": float(jaccard_score(y_true_idx == i, y_pred_idx == i))
                if np.any(y_true_idx == i) or np.any(y_pred_idx == i)
                else 0.0,
            }
        else:
            per_class[lab] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0, "jaccard": 0.0}

    # ---- AUC computations (require probabilities) ----
    roc_auc_ovr_macro = roc_auc_ovr_weighted = roc_auc_ovo_macro = roc_auc_ovo_weighted = None
    per_class_auc_ovr: Optional[Dict[Any, float]] = None

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim != 2 or y_proba.shape[0] != y_true.shape[0]:
            raise ValueError("y_proba must have shape (N, K).")
        if y_proba.shape[1] != n_classes:
            raise ValueError(
                f"y_proba second dimension ({y_proba.shape[1]}) must match number of classes ({n_classes}). "
                "If your probabilities are in a different class order, pass the correct `labels` argument."
            )

        y_proba = _normalize_rows(y_proba)

        # Binarize ground truth for AUC
        Y_true_bin = label_binarize(y_true_idx, classes=list(range(n_classes)))

        # Per-class OvR AUCs
        per_class_auc_ovr = {}
        per_class_auprc_ovr = {}
        for i, lab in enumerate(labels_list):
            # Only compute if both positive and negative examples exist; otherwise roc_auc_score raises
            if np.unique(Y_true_bin[:, i]).size == 2:
                try:
                    auc_i = roc_auc_score(Y_true_bin[:, i], y_proba[:, i])
                    auprc_i = average_precision_score(Y_true_bin[:, i], y_proba[:, i])
                except ValueError:
                    auc_i = np.nan
                    auprc_i = np.nan
            else:
                auc_i = np.nan
                auprc_i = np.nan
            per_class_auc_ovr[lab] = float(auc_i) if auc_i == auc_i else None  # NaN -> None
            per_class_auprc_ovr[lab] = float(auprc_i) if auprc_i == auprc_i else None  # NaN -> None

        # Aggregate AUCs using sklearn's multiclass handling
        try:
            roc_auc_ovr_macro = float(
                roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
            )
            auprc_ovr_macro = float(average_precision_score(y_true_idx, y_proba, average="macro"))
        except ValueError:
            roc_auc_ovr_macro = None
            auprc_ovr_macro = None

        try:
            roc_auc_ovr_weighted = float(
                roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="weighted", labels=list(range(n_classes)))
            )
            auprc_ovr_weighted = float(average_precision_score(y_true_idx, y_proba, average="weighted"))
        except ValueError:
            roc_auc_ovr_weighted = None
            auprc_ovr_weighted = None

        try:
            roc_auc_ovo_macro = float(
                roc_auc_score(y_true_idx, y_proba, multi_class="ovo", average="macro", labels=list(range(n_classes)))
            )
            # OvO is not supported for AUPRC
            auprc_ovo_macro = None
        except ValueError:
            roc_auc_ovo_macro = None
            auprc_ovo_macro = None

        try:
            roc_auc_ovo_weighted = float(
                roc_auc_score(y_true_idx, y_proba, multi_class="ovo", average="weighted", labels=list(range(n_classes)))
            )
            # OvO is not supported for AUPRC
            auprc_ovo_weighted = None
        except ValueError:
            roc_auc_ovo_weighted = None
            auprc_ovo_weighted = None

    return MetricsResult(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        f1_weighted=f1_weighted,
        jaccard_macro=jacc_macro,
        jaccard_micro=jacc_micro,
        jaccard_weighted=jacc_weighted,
        roc_auc_ovr_macro=roc_auc_ovr_macro,
        roc_auc_ovr_weighted=roc_auc_ovr_weighted,
        roc_auc_ovo_macro=roc_auc_ovo_macro,
        roc_auc_ovo_weighted=roc_auc_ovo_weighted,
        per_class=per_class,
        per_class_ovr_auc=per_class_auc_ovr,
        per_class_auprc_ovr=per_class_auprc_ovr,
        auprc_ovr_macro=auprc_ovr_macro,
        auprc_ovr_weighted=auprc_ovr_weighted,
        auprc_ovo_macro=auprc_ovo_macro,
        auprc_ovo_weighted=auprc_ovo_weighted,
    )


# Optional: tiny demo using random data; only runs if you execute this file directly.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 100
    K = 5
    y_true = rng.integers(0, K, size=N)
    # Create somewhat noisy predictions
    y_pred = y_true.copy()
    flip_idx = rng.choice(N, size=N // 5, replace=False)
    y_pred[flip_idx] = rng.integers(0, K, size=flip_idx.size)

    # Make probabilities consistent with y_pred but not perfect
    y_proba = rng.random((N, K))
    # Boost probability of the predicted class a bit
    y_proba[np.arange(N), y_pred] += 1.0
    # Normalize rows
    y_proba = _normalize_rows(y_proba)

    res = compute_multiclass_metrics(y_true, y_pred, y_proba=y_proba, labels=list(range(K)))
    import json
    print(json.dumps(res.to_dict(), indent=2))
