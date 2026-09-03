"""Task-aware metrics, bootstrap confidence intervals, and paired model comparisons.

Two conventions worth knowing:

* Bootstrap stratification is per metric. Accuracy and balanced accuracy are resampled
  unstratified, since stratifying fixes the class counts and shifts the estimand. Macro and
  per-class metrics are resampled stratified within class, so no class can vanish.
* Seed variance and test-set variance are separate. `seed_stats` measures training stochasticity
  across seeds on a fixed test set; `bootstrap_ci` measures sampling uncertainty of the
  seed-ensembled prediction.

Model-vs-model claims use `paired_bootstrap` or `mcnemar` on identical items; two overlapping
marginal intervals are not a significance test.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np
from sklearn.metrics import (auc, average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import label_binarize

# metrics for which a stratified bootstrap is the right choice
_STRATIFIED_BY_DEFAULT = {"accuracy": False, "balanced_acc": False, "f1_micro": False}


def _pr_auc(y_bin: np.ndarray, score: np.ndarray) -> tuple:
    """Both PR-AUC estimators from one precision-recall curve. Returns (trapz, ap).

    `trapz` is `auc(recall, precision)`, the estimator the reported numbers use. `ap` is average
    precision, the step-wise sum, since interpolating between PR points can be optimistic. Both
    come from one curve, so the bootstrap does not pay twice.
    """
    y_bin = np.asarray(y_bin).astype(int)
    if y_bin.min() == y_bin.max():          # single-class slice -> both undefined
        return float("nan"), float("nan")
    precision, recall, _ = precision_recall_curve(y_bin, score)
    trapz = float(auc(recall, precision))
    # average_precision_score's definition, computed off the same curve
    ap = float(-np.sum(np.diff(recall) * np.asarray(precision)[:-1]))
    return trapz, ap


def _macro(vals: Sequence[float]) -> float:
    """Unweighted mean over classes, skipping NaN."""
    good = [v for v in vals if not np.isnan(v)]
    return float(np.mean(good)) if good else float("nan")


class SeedStats(NamedTuple):
    mean: float
    std: float
    ci95: float          # half-width, 1.96 * SE


def seed_stats(vals: Sequence[float]) -> SeedStats:
    a = np.asarray(list(vals), dtype=float)
    if a.size < 2:
        return SeedStats(float(a.mean()) if a.size else float("nan"), 0.0, 0.0)
    sd = float(a.std(ddof=1))
    return SeedStats(float(a.mean()), sd, float(1.96 * sd / np.sqrt(a.size)))


def probabilities_from_logits(task_type: str, logits: np.ndarray) -> np.ndarray:
    """Logits -> the probability form that gets averaged across seeds."""
    z = np.asarray(logits, dtype=float)
    if task_type == "binary":
        return 1.0 / (1.0 + np.exp(-z.reshape(-1)))
    if task_type == "multilabel":
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _binary_auroc(y_bin: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(np.asarray(y_bin).astype(int), score))
    except ValueError:                       # single-class slice
        return float("nan")


def metrics_from_prob(task_type: str, y: np.ndarray, prob: np.ndarray,
                      class_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Task-aware metric dict. Per-class entries are added when `class_names` is given.

    * `auprc*` keys are the trapezoidal estimator; the `*_ap` twin is average precision.
    * Macro AUROC and AUPRC are one-vs-rest per class, then an unweighted mean skipping NaN.
    * F1 comes from hard labels: argmax for multiclass, threshold 0.5 otherwise.
    * For single-label multiclass `f1_micro` equals accuracy.
    """
    y = np.asarray(y)
    out: Dict[str, float] = {}
    if task_type == "binary":
        p = np.asarray(prob).reshape(-1)
        out["auprc"], out["auprc_ap"] = _pr_auc(y, p)
        out["auroc"] = _binary_auroc(y, p)
        return out

    if task_type == "multilabel":
        pred = (prob >= 0.5).astype(int)
        out["f1_macro"] = float(f1_score(y, pred, average="macro", zero_division=0))
        out["f1_micro"] = float(f1_score(y, pred, average="micro", zero_division=0))
        per = [_pr_auc(y[:, c], prob[:, c]) for c in range(prob.shape[1])]
        out["auprc_macro"] = _macro([t for t, _ in per])
        out["auprc_macro_ap"] = _macro([a for _, a in per])
        out["auroc_macro"] = _macro([_binary_auroc(y[:, c], prob[:, c])
                                     for c in range(prob.shape[1])])
        if class_names is not None:
            f1s = f1_score(y, pred, average=None, zero_division=0)
            for c, name in enumerate(class_names):
                out[f"F1_{name}"] = float(f1s[c])
                out[f"AUPRC_{name}"] = per[c][0]
                out[f"AUROC_{name}"] = _binary_auroc(y[:, c], prob[:, c])
        return out

    # multiclass
    n_classes = prob.shape[1]
    pred = prob.argmax(1)
    # label_binarize collapses two classes to one column; expand it back so the per-class loop
    # below does not index off the end
    Y = label_binarize(y, classes=list(range(n_classes)))
    if Y.shape[1] == 1 and n_classes == 2:
        Y = np.hstack([1 - Y, Y])
    out["accuracy"] = float((pred == y).mean())
    out["balanced_acc"] = float(balanced_accuracy_score(y, pred))
    out["f1_macro"] = float(f1_score(y, pred, average="macro", zero_division=0))
    out["f1_micro"] = float(f1_score(y, pred, average="micro", zero_division=0))
    per = [_pr_auc(Y[:, c], prob[:, c]) for c in range(n_classes)]
    out["auprc_macro"] = _macro([t for t, _ in per])
    out["auprc_macro_ap"] = _macro([a for _, a in per])
    out["auroc_macro"] = _macro([_binary_auroc(Y[:, c], prob[:, c]) for c in range(n_classes)])
    if class_names is not None:
        f1s = f1_score(y, pred, average=None, labels=list(range(n_classes)), zero_division=0)
        for c, name in enumerate(class_names):
            out[f"F1_{name}"] = float(f1s[c])
            out[f"AUPRC_{name}"] = per[c][0]
            out[f"AUROC_{name}"] = _binary_auroc(Y[:, c], prob[:, c])
    return out


def _classes_for_stratify(task_type: str, y: np.ndarray) -> Optional[List[np.ndarray]]:
    """Index groups to resample within. Multilabel has no single class per item -> unstratified."""
    if task_type == "multilabel":
        return None
    y = np.asarray(y)
    vals = np.unique(y.astype(int))
    return [np.where(y.astype(int) == v)[0] for v in vals]


def _resample(rng, y, task_type, stratified):
    n = len(y)
    if not stratified:
        return rng.integers(0, n, n)
    groups = _classes_for_stratify(task_type, y)
    if groups is None:
        return rng.integers(0, n, n)
    return np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if len(g)])


def bootstrap_ci(task_type: str, y: np.ndarray, prob: np.ndarray, metric_key: str,
                 n_boot: int = 2000, seed: int = 0, stratified: Optional[bool] = None,
                 class_names: Optional[Sequence[str]] = None) -> tuple:
    """Percentile bootstrap on the (seed-ensembled) prediction. Returns (value, lo, hi).

    `stratified` defaults per metric: accuracy/balanced-acc unstratified, everything else stratified.
    """
    if stratified is None:
        stratified = _STRATIFIED_BY_DEFAULT.get(metric_key, True)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _resample(rng, y, task_type, stratified)
        try:
            vals.append(metrics_from_prob(task_type, y[idx], prob[idx], class_names)[metric_key])
        except (ValueError, KeyError, IndexError):
            continue
    point = metrics_from_prob(task_type, y, prob, class_names)[metric_key]
    if not vals:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_bootstrap(task_type: str, y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                     metric_key: str, n_boot: int = 2000, seed: int = 0,
                     stratified: Optional[bool] = None,
                     class_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Bootstrap the DIFFERENCE (A - B) on shared resample indices. Same items, so this is paired."""
    if stratified is None:
        stratified = _STRATIFIED_BY_DEFAULT.get(metric_key, True)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample(rng, y, task_type, stratified)
        try:
            ma = metrics_from_prob(task_type, y[idx], prob_a[idx], class_names)[metric_key]
            mb = metrics_from_prob(task_type, y[idx], prob_b[idx], class_names)[metric_key]
        except (ValueError, KeyError, IndexError):
            continue
        diffs.append(ma - mb)
    pa = metrics_from_prob(task_type, y, prob_a, class_names)[metric_key]
    pb = metrics_from_prob(task_type, y, prob_b, class_names)[metric_key]
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"),) * 2)
    return {"delta": float(pa - pb), "lo": float(lo), "hi": float(hi),
            "significant": bool(diffs) and (lo > 0 or hi < 0)}


def mcnemar(y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> Dict[str, float]:
    """Exact McNemar on hard multiclass predictions (binomial, not the chi-square approximation --
    discordant counts here are small)."""
    from scipy.stats import binomtest
    y = np.asarray(y)
    a_ok = prob_a.argmax(1) == y
    b_ok = prob_b.argmax(1) == y
    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    p = float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if (n01 + n10) else 1.0
    return {"only_a": n10, "only_b": n01, "p_value": p}


# ------------------------------------------------------- hard labels, and clustered resampling
#: Classification metrics defined for a hard prediction. Balanced accuracy leads because these
#: label sets are imbalanced enough that plain accuracy mostly reports the class prior.
HARD_LABEL_METRICS = ("balanced_acc", "accuracy", "f1_macro")


def hard_label_metrics(y, pred) -> Dict[str, float]:
    """Balanced accuracy, accuracy and macro-F1 from hard labels.

    Separate from `metrics_from_prob` because it takes labels rather than probabilities and so
    works for any classifier, and because it is cheap: `metrics_from_prob` also builds a
    precision-recall curve per class, which dominates the cost of a 2,000-resample bootstrap and
    is not defined for a majority-class baseline. Labels may be strings; `y` and `pred` only have
    to be comparable. Metric names match `metrics_from_prob` so one results table can hold both.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    return {"balanced_acc": float(balanced_accuracy_score(y, pred)),
            "accuracy": float((pred == y).mean()),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0))}


class _Clusters:
    """Point indices grouped by cluster, in a layout that makes resampling one array operation.

    The obvious implementation -- a list of index arrays, concatenated per resample -- spends all
    its time in Python: 2,000 resamples of 1,500 PDB entries is three million list operations. This
    stores the points sorted by cluster with an offset per cluster, so drawing a resample is a
    `repeat` and an `arange`. The points selected are identical either way; only their order within
    a resample differs, and every metric here is order-invariant.
    """

    def __init__(self, groups):
        codes = np.unique(np.asarray(groups), return_inverse=True)[1]
        self.order = np.argsort(codes, kind="stable")
        self.sizes = np.bincount(codes)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.n = len(self.sizes)

    def resample(self, rng) -> np.ndarray:
        pick = rng.integers(0, self.n, self.n)
        sizes = self.sizes[pick]
        out_starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        offsets = np.repeat(self.starts[pick] - out_starts, sizes)
        return self.order[offsets + np.arange(sizes.sum())]


def cluster_bootstrap_ci(y, pred, groups, metric_key: str = "balanced_acc",
                         n_boot: int = 2000, seed: int = 0) -> tuple:
    """Percentile bootstrap that resamples CLUSTERS with replacement. Returns (value, lo, hi).

    Use this, not `bootstrap_ci`, whenever the evaluation points are not independent -- several
    residues from one structure, several metal sites from one PDB entry, several pockets from one
    protein. Resampling points inside a cluster treats correlated observations as independent
    evidence and returns an interval that is too narrow. `groups` names the cluster of each point,
    a whole cluster is taken or left, and the number of clusters is held fixed.

    A resample in which only one class survives is skipped rather than scored, since balanced
    accuracy is undefined there.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    clusters = _Clusters(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = clusters.resample(rng)
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(hard_label_metrics(y[idx], pred[idx])[metric_key])
    point = hard_label_metrics(y, pred)[metric_key]
    if not values:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_cluster_bootstrap(y, pred_a, pred_b, groups, metric_key: str = "balanced_acc",
                             n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Cluster bootstrap of the DIFFERENCE a - b, both arms scored on the same resampled points.

    Pairing is what makes the comparison sharp: the two arms share every resample, so the variance
    of the points themselves cancels and what is left is the variance of the difference. Two
    overlapping marginal intervals from `cluster_bootstrap_ci` are not a test of anything.
    """
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    clusters = _Clusters(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = clusters.resample(rng)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(hard_label_metrics(y[idx], pred_a[idx])[metric_key]
                     - hard_label_metrics(y[idx], pred_b[idx])[metric_key])
    delta = (hard_label_metrics(y, pred_a)[metric_key]
             - hard_label_metrics(y, pred_b)[metric_key])
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"),) * 2)
    return {"delta": float(delta), "lo": float(lo), "hi": float(hi),
            "significant": bool(diffs) and (lo > 0 or hi < 0)}
