"""
Fusing a frozen ATOMICA embedding with a frozen sequence-model embedding.

Train the two probes independently, then average their predicted probabilities.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .metrics import metrics_from_prob


def late_fusion(prob_a: np.ndarray, prob_b: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Committed operator: convex combination of two probability arrays. `weight` is A's share."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return weight * prob_a + (1.0 - weight) * prob_b


def select_weight_on_validation(task_type: str, y_val: np.ndarray, val_a: np.ndarray, val_b: np.ndarray,
                                primary: str, grid: Optional[Sequence[float]] = None) -> float:
    """Optional: pick the mixing weight on VALIDATION.
    """
    grid = grid if grid is not None else np.arange(0.0, 1.01, 0.05)
    best_w, best_v = 0.5, -np.inf
    for w in grid:
        v = metrics_from_prob(task_type, y_val, late_fusion(val_a, val_b, float(w)))[primary]
        if v > best_v:
            best_v, best_w = v, float(w)
    return best_w


def fuse_probe_outputs(task_type: str, y_test: np.ndarray, run_a: Dict, run_b: Dict,
                       primary: str, weight: float = 0.5,
                       class_names: Optional[Sequence[str]] = None) -> Dict:
    """Late-fuse two completed `train_probe` runs.
    """
    ens_a = run_a["test_probs"].mean(0)
    ens_b = run_b["test_probs"].mean(0)
    fused = late_fusion(ens_a, ens_b, weight)
    return {
        "operator": "late_fusion", "weight": weight,
        "ensemble": metrics_from_prob(task_type, y_test, fused, class_names),
        "branch_a": metrics_from_prob(task_type, y_test, ens_a, class_names),
        "branch_b": metrics_from_prob(task_type, y_test, ens_b, class_names),
        "fused_probs": fused,
    }
