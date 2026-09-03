"""
The standard frozen-embedding probe training loop.

Protocol is matched to the frozen sequence-model baselines so the comparison is a representation
comparison, not a training-recipe comparison: Adam, plain cross-entropy / BCE, fixed epoch budget with
early stopping on the task's validation metric, N seeds, and **probability ensembling** across seeds
(which is how the baselines are scored).

Per-benchmark hyperparameters (epochs / patience / batch size / weight decay) are passed in via
`ProbeConfig` rather than hard-coded, because they legitimately differ: the MaSIF settings were chosen to
match the PLM baselines exactly, and forcing a single global constant would break the protocol-matching
the fairness argument rests on. `early_stopping=False` is part of that: the ATP/ADP protocol trains a
fixed 60-epoch budget and selects only the hyperparameter configuration on validation, so restoring a
best-validation checkpoint there would add a selection step the Methods do not describe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .head import AtomicaProbeHead, num_outputs
from .metrics import metrics_from_prob, probabilities_from_logits, seed_stats


@dataclass
class ProbeConfig:
    """Training hyperparameters. Defaults follow the frozen-PLM-baseline protocol."""
    hidden_dim: int = 512
    final_hidden_dim: int = 32
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    patience: int = 15
    batch_size: int = 64
    seeds: Sequence[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    min_delta: float = 1e-4
    # False reproduces the baselines' MLPClassifier exactly. Swept, not assumed -- see head.py.
    use_batchnorm: bool = True
    # False trains the full `epochs` budget and keeps the FINAL weights, rather than restoring the
    # best-validation checkpoint. Some benchmarks are specified that way -- the ATP/ADP protocol
    # fits a fixed 60 epochs and selects only the hyperparameter configuration on validation -- and
    # restoring a checkpoint there would add a selection step the Methods do not describe.
    # `patience` is ignored when this is False.
    early_stopping: bool = True


def make_loss(task_type: str, y_train: np.ndarray, loss: str, device) -> nn.Module:
    """`loss` is one of: ce | weighted_ce | bce | focal_bce. The committed recipe uses the plain forms,
    matching the baselines; the weighted/focal variants exist for the validation-selected sweep."""
    if task_type == "binary":
        return nn.BCEWithLogitsLoss()
    if task_type == "multilabel":
        if loss == "focal_bce":
            return _FocalBCE(gamma=2.0)
        return nn.BCEWithLogitsLoss()
    if loss == "weighted_ce":
        y = np.asarray(y_train).astype(int)
        n_classes = int(y.max()) + 1
        counts = np.bincount(y, minlength=n_classes).astype(float)
        w = len(y) / (n_classes * np.maximum(counts, 1))
        return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    return nn.CrossEntropyLoss()


class _FocalBCE(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        return ((1 - pt) ** self.gamma * bce).mean()


def _targets(task_type: str, y, device):
    if task_type == "multiclass":
        return torch.tensor(np.asarray(y), dtype=torch.long, device=device)
    t = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
    return t.view(-1, 1) if task_type == "binary" else t


def train_one_seed(X_tr, y_tr, X_va, y_va, X_te, task_type: str, primary: str, loss: str,
                   cfg: ProbeConfig, seed: int, device) -> Dict:
    """Train one head; return test/val probabilities of the best-validation checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_out = num_outputs(task_type, y_tr)
    model = AtomicaProbeHead(X_tr.shape[1], n_out, task_type, cfg.hidden_dim,
                             cfg.final_hidden_dim, cfg.dropout,
                             use_batchnorm=cfg.use_batchnorm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = make_loss(task_type, y_tr, loss, device)

    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
    xtr, xva, xte = T(X_tr), T(X_va), T(X_te)
    ytr = _targets(task_type, y_tr, device)

    best, best_state, patience, n = -np.inf, None, 0, len(y_tr)
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if idx.numel() < 2:          # BatchNorm needs >= 2 samples
                continue
            opt.zero_grad()
            crit(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_logits = model(xva).cpu().numpy()
        score = metrics_from_prob(task_type, y_va,
                                  probabilities_from_logits(task_type, va_logits))[primary]
        if not cfg.early_stopping:
            # Fixed budget: run every epoch and keep the last weights. `best` still tracks the
            # validation score, which is what the caller selects hyperparameters on, but it now
            # reports the score of the model that is actually returned.
            best = score
            continue
        if score > best + cfg.min_delta:
            best, patience = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(xte).cpu().numpy()
        va_logits = model(xva).cpu().numpy()
    return {"test_prob": probabilities_from_logits(task_type, te_logits),
            "val_prob": probabilities_from_logits(task_type, va_logits),
            "val_primary": float(best),
            # the best-validation checkpoint, so a trained probe can be saved and reloaded
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def save_probe(save_dir, runs, cfg, task_type, primary, loss, input_dim, n_out, class_names=None,
               standardizer=None):
    """Persist a trained probe so it can be reloaded and re-scored without retraining.

    Writes, per seed, the best-validation state_dict, plus ONE meta.json holding everything needed
    to rebuild the head (dims, task type, and the full ProbeConfig) and the train-fit z-score
    statistics. Weights alone are not enough -- the head expects standardized inputs.
    """
    import dataclasses, json as _json, os as _os
    _os.makedirs(save_dir, exist_ok=True)
    for seed, r in zip(cfg.seeds, runs):
        torch.save(r["state_dict"], _os.path.join(save_dir, f"weights_seed{seed}.pt"))
    if standardizer is not None:
        mu, sd = standardizer
        np.savez(_os.path.join(save_dir, "standardizer.npz"), mu=mu, sd=sd)
    with open(_os.path.join(save_dir, "meta.json"), "w") as fh:
        _json.dump({"task_type": task_type, "primary": primary, "loss": loss,
                    "input_dim": int(input_dim), "num_classes": int(n_out),
                    "class_names": list(class_names) if class_names is not None else None,
                    "standardized": standardizer is not None,
                    "config": dataclasses.asdict(cfg)}, fh, indent=1)


def load_probe(save_dir, seed):
    """Rebuild the head for one seed from disk -> (model, standardizer or None, meta)."""
    import json as _json, os as _os
    meta = _json.load(open(_os.path.join(save_dir, "meta.json")))
    c = meta["config"]
    model = AtomicaProbeHead(meta["input_dim"], meta["num_classes"], meta["task_type"],
                             c["hidden_dim"], c["final_hidden_dim"], c["dropout"],
                             use_batchnorm=c.get("use_batchnorm", True))
    model.load_state_dict(torch.load(_os.path.join(save_dir, f"weights_seed{seed}.pt"),
                                     map_location="cpu"))
    model.eval()
    std = None
    if meta.get("standardized"):
        z = np.load(_os.path.join(save_dir, "standardizer.npz"))
        std = (z["mu"], z["sd"])
    return model, std, meta


def train_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, task_type: str, primary: str,
                loss: str = "ce", cfg: Optional[ProbeConfig] = None,
                class_names: Optional[Sequence[str]] = None, device: Optional[str] = None,
                save_dir: Optional[str] = None,
                standardizer: Optional[tuple] = None) -> Dict:
    """Run the full multi-seed probe and ensemble the per-seed probabilities.

    Returns per-seed metrics (mean +/- CI over seeds) AND the ensemble metrics, plus the raw probability
    arrays so downstream code can bootstrap or fuse without retraining.
    """
    cfg = cfg or ProbeConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs = [train_one_seed(X_tr, y_tr, X_va, y_va, X_te, task_type, primary, loss, cfg, s, device)
            for s in cfg.seeds]

    test_probs = np.stack([r["test_prob"] for r in runs])
    val_probs = np.stack([r["val_prob"] for r in runs])
    per_seed = [metrics_from_prob(task_type, y_te, p, class_names) for p in test_probs]
    ensemble = metrics_from_prob(task_type, y_te, test_probs.mean(0), class_names)
    val_ens = metrics_from_prob(task_type, y_va, val_probs.mean(0), class_names)

    if save_dir is not None:
        save_probe(save_dir, runs, cfg, task_type, primary, loss,
                   input_dim=X_tr.shape[1], n_out=num_outputs(task_type, y_tr),
                   class_names=class_names, standardizer=standardizer)

    keys = list(ensemble)
    return {
        "task_type": task_type, "primary": primary, "loss": loss,
        "dim": int(X_tr.shape[1]), "n_test": int(len(y_te)), "n_seeds": len(cfg.seeds),
        "val_primary_per_seed": float(np.mean([r["val_primary"] for r in runs])),
        "val_primary_ensemble": float(val_ens[primary]),
        "per_seed_mean": {k: seed_stats([m[k] for m in per_seed]).mean for k in keys},
        "per_seed_ci95": {k: seed_stats([m[k] for m in per_seed]).ci95 for k in keys},
        "ensemble": ensemble,
        "test_probs": test_probs, "val_probs": val_probs,
    }
