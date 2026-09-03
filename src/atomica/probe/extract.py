"""Model and data in, probe-ready feature matrix out.

The seam between :mod:`atomica.representations` and the rest of the probe, so that no benchmark
re-implements pooling. It uses the ``z`` family pooled by ``mean_std_global``, since a head is
trained on top: ``z_graph`` for graph- and pocket-level tasks, ``z_block`` for residue-level ones.

Both functions take already-collated batches, so how those batches were built is the caller's
choice and it changes the output: the per-block attention pads every block out to the largest
block in the batch. :func:`atomica.representations.group_batches` builds batches that keep each
structure at the padding width it would have alone. Fix the batch size and item order across
train, validation and test, and record them with the features.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import representations as R

__all__ = ["extract_graph_features", "extract_block_features", "GRAPH_LEVEL_TASKS",
           "RESIDUE_LEVEL_TASKS"]

#: Tasks whose unit of prediction is a whole graph or pocket. These use ``z_graph``.
GRAPH_LEVEL_TASKS = ("rna_go", "rna_ligand", "masif_ligand", "atp_adp", "pocket_retrieval")

#: Tasks whose unit of prediction is one residue. These use ``z_block``.
RESIDUE_LEVEL_TASKS = ("rna_protein", "rna_site")


@torch.no_grad()
def extract_graph_features(model, batches: Iterable[dict], *,
                           pool: str = "mean_std_global",
                           ids_key: Optional[str] = None,
                           device: Optional[str] = None) -> Tuple[np.ndarray, List]:
    """One ``z_graph`` row per graph, over an iterable of collated batches.

    Parameters
    ----------
    pool : str
        ``mean_std_global`` when a head is trained on these features, which is the probe's case and
        the default. ``mean_component_normalized`` only if you are going to compare the vectors
        directly with a cosine and train nothing. The two are not interchangeable.
    ids_key : str, optional
        A key in each batch holding one id per graph, carried through so labels can be joined.

    Returns ``(X, ids)`` with ``X`` of shape ``[n_graphs, pooled width]``.
    """
    model.eval()
    if device is not None:
        model = model.to(device)
    feats, ids = [], []
    for batch in batches:
        if device is not None:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        z = R.get(model, batch, "z_graph", pool=pool)
        feats.append(z.detach().cpu().numpy())
        if ids_key is not None:
            ids.extend(list(batch[ids_key]))
    if not feats:
        raise ValueError("no batches were supplied")
    return np.concatenate(feats, axis=0).astype(np.float32), ids


@torch.no_grad()
def extract_block_features(model, batches: Iterable[dict], *,
                           drop_global: bool = True,
                           device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """One ``z_block`` row per block, for residue-level tasks.

    Returns ``(X, graph_index)`` where ``graph_index`` says which graph each row came from, counted
    across the whole iterable rather than restarting per batch. The global block node is dropped by
    default, since it is not a residue and carries no label.
    """
    model.eval()
    if device is not None:
        model = model.to(device)
    feats, index, offset = [], [], 0
    for batch in batches:
        if device is not None:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rv = model.infer(batch, return_invariant_repr=True, invariant_pool=None)
        z = rv.block_invariant_repr
        gid = rv.batch_id
        if drop_global:
            keep = batch["B"] != model.global_block_id
            z, gid = z[keep], gid[keep]
        feats.append(z.detach().cpu().numpy())
        index.append(gid.detach().cpu().numpy() + offset)
        offset += int(batch["lengths"].shape[0])
    if not feats:
        raise ValueError("no batches were supplied")
    return (np.concatenate(feats, axis=0).astype(np.float32),
            np.concatenate(index, axis=0))
