"""Model and data in, probe-ready feature matrix out.

This is the seam between :mod:`atomica.representations` and the rest of the probe. It exists so
that "which representation does the probe train on?" has an answer you can read in one function
rather than infer from a pipeline, and so that no benchmark re-implements pooling.

The answer is: **the ``z`` family, pooled by ``mean_std_global``**, because a head is trained on
top and the Methods fix the pooling by how the representation is consumed. Graph- and pocket-level
tasks use ``z_graph``; residue-level tasks use ``z_block``.

ONE THING THE CALLER OWNS, AND MUST NOT VARY
--------------------------------------------
Both functions here take an iterable of already-collated batches, so **how the batches were built
is the caller's choice and it changes the output.** The per-block cross-attention pads every block
out to the largest block in the batch, counted in atoms, under an unmasked softmax, so a structure
matches its batch-of-one value only when nothing else in its batch has a larger block. See
:func:`atomica.representations.describe_batch_sensitivity`.

So a probe run must fix the batch size and the item order, use the same ones for train, validation
and test, and record them next to the features. Comparing a frozen number against a published one
means matching the batch size that produced the published one, not only the checkpoint and the
pooling rule. Measured on MaSIF-ligand, where the published extraction used batch size 16: the
pockets whose largest block is smaller than their batch's largest move by up to 0.128 between batch
size 1 and 16, on a scale where the largest vector entry is 7.1.
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
