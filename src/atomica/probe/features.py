"""Turning ATOMICA's ``z`` representations into a feature matrix for the probe head.

The three components of ``z_block``, in concatenation order, are the keys that
``PredictionModel.invariant_component_dims()`` returns:

    h_block   the l=0 block readout, ``ns`` wide (32 for the released checkpoint)
    gram      the block's own within-degree Gram entries (544)
    atom      the mean and standard deviation of ``z_atom`` over the block's atoms (1216)

:data:`FEATURE_SETS` names the full descriptor and the nested slices beneath it, so a narrower
feature set is a slice of one extraction rather than a second run of the model.

Standardization is fit on train only and applied to validation and test. It complements the head's
BatchNorm rather than repeating it: BatchNorm sits after the first Linear and never sees the raw
input, whose components span orders of magnitude.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..representations import POOLING, component_normalize

#: The parts of ``z_block``, in concatenation order. Matches ``invariant_component_dims()``.
Z_BLOCK_COMPONENTS: Tuple[str, ...] = ("h_block", "gram", "atom")

#: The full descriptor and the nested slices beneath it.
FEATURE_SETS: Dict[str, Sequence[str]] = {
    # the l=0 readout on its own; this is the h family, not a z representation
    "h_block": ("h_block",),
    # + the block's own Gram entries
    "z_block_gram": ("h_block", "gram"),
    # + the atom-level descriptor pooled into the block: the full z_block
    "z_block": ("h_block", "gram", "atom"),
}
__all__ = [
    "Z_BLOCK_COMPONENTS", "FEATURE_SETS",
    "build_features", "split_z_block", "pool_saved_blocks",
    "fit_standardizer", "apply_standardizer", "standardize", "l2_normalize",
]


def split_z_block(z_block: np.ndarray, component_dims: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Slice a ``z_block`` array back into its three named parts.

    ``component_dims`` is what ``model.invariant_component_dims()`` returns, so the split can never
    drift from how the model built the vector.
    """
    total = int(sum(component_dims.values()))
    if z_block.shape[-1] != total:
        raise ValueError(f"z_block is {z_block.shape[-1]} wide but component_dims sum to {total}; "
                         f"these came from different models or a different pooling")
    out, off = {}, 0
    for name, width in component_dims.items():
        out[name] = z_block[..., off:off + int(width)]
        off += int(width)
    return out


def build_features(arrays: Dict[str, np.ndarray], feature_set: str = "z_block") -> np.ndarray:
    """Concatenate the named components of a feature set into one matrix.

    ``arrays`` maps component name to array, as produced by :func:`split_z_block`. The default is
    the full descriptor; pick a narrower key of :data:`FEATURE_SETS` to use a slice of it.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {list(FEATURE_SETS)}")
    missing = [k for k in FEATURE_SETS[feature_set] if k not in arrays]
    if missing:
        raise KeyError(f"feature set {feature_set!r} needs components {missing}, which are not in "
                       f"the extraction output (found: {sorted(arrays)}). If your arrays are keyed "
                       f"by the old names, block_repr is now h_block.")
    return np.concatenate([arrays[k] for k in FEATURE_SETS[feature_set]], axis=1).astype(np.float32)


def pool_saved_blocks(X: np.ndarray, keys: Sequence, is_global: np.ndarray,
                      mode: str = "mean_std_global",
                      component_dims: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Pool saved per-block ``z_block`` arrays into one vector per graph. Numpy path.

    The torch path lives in :func:`atomica.representations.pool_blocks` and runs inside the model.
    This is its counterpart for arrays already written to disk, and it implements the same two
    rules, so the two cannot drift apart.

    Graph order is first appearance, so labels joined on ``keys`` stay aligned.
    """
    if mode not in POOLING:
        raise ValueError(f"unknown pooling {mode!r}; expected one of {list(POOLING)}")
    keys = np.asarray(keys)
    is_global = np.asarray(is_global, dtype=bool)
    order, seen = [], set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)

    pooled = []
    for k in order:
        m = keys == k
        real = X[m & ~is_global]
        if real.shape[0] == 0:          # degenerate: the graph is only its global node
            real = X[m]
        mean = real.mean(0)
        if mode == "mean_component_normalized":
            if not component_dims:
                raise ValueError("mean_component_normalized needs component_dims; pass the "
                                 "model's invariant_component_dims()")
            import torch
            pooled.append(component_normalize(torch.from_numpy(np.asarray(mean)),
                                              list(component_dims.values())).numpy())
            continue
        std = real.std(0)
        glob = X[m & is_global]
        glob = glob[0] if len(glob) else np.zeros(X.shape[1], dtype=X.dtype)
        pooled.append(np.concatenate([mean, std, glob]))
    return np.asarray(order), np.stack(pooled).astype(np.float32)


# ------------------------------------------------------------------------------- standardization
def fit_standardizer(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Train-fit z-score statistics. Returned explicitly so they can be persisted.

    A saved probe is only reloadable if these travel with the weights: the head was trained on
    z-scored inputs, so applying it to raw features produces garbage silently rather than raising.
    """
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True) + 1e-6
    return mu, sd


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def standardize(train: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Train-fit z-score applied to every split. Returns (train, *others) transformed."""
    mu, sd = fit_standardizer(train)
    return (apply_standardizer(train, mu, sd), *[apply_standardizer(o, mu, sd) for o in others])


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
