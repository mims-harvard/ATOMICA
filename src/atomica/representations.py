"""Named ATOMICA representations, using the symbols the paper uses.

Two families at four levels:

    name           paper symbol     what it is
    ------------   --------------   -------------------------------------------------------
    h_atom         h_a^atom         per-atom scalar readout
    h_block        h_b^block        per-block scalar readout
    h_graph        h_i^graph        pooled over a graph's blocks
    h_interface    h_A^interface    pooled over one molecule's blocks
    z_atom         z_a^atom         atom channels plus rotation invariants of the l>0 channels
    z_block        z_b^block        h_block, its own invariants, and z_atom pooled over the block
    z_graph        z_i^graph        z_block pooled over a graph's blocks
    z_interface    z_i^interface    z_block pooled over one molecule's blocks

The h family keeps the l=0 channels only. The z family adds rotation invariants built from the
l>0 channels, so an ordinary head can use geometry that h drops, and it is what the paper's frozen
benchmarks train on. The interface level pools one molecule's blocks after message passing has run
over the whole complex, so it stays aware of the partner. Widths are read off the model.

z_graph and z_interface take a pooling rule and have no default, because the two are not
comparable:

    mean_std_global             for training a head on top: mean, standard deviation, global node
    mean_component_normalized   for comparing frozen vectors: mean, each part L2-normalized

Component normalization matters because the three parts of z_block differ by about an order of
magnitude in norm, and a cosine weights each part by the product of its norms; over the 21 pairs
of the seven example complexes the atom-pooled part alone carries a median 96.9 percent of an
unnormalized dot product.

For reproducibility, batch composition affects the result, since the per-block attention runs over
a batch-sized padded array. :func:`embed_dataset` batches structures that share a largest block,
which reproduces batch-size-1 values. See :func:`describe_batch_sensitivity`.

One asymmetry to know: z_atom is built from the raw 0e/0o channels while h_atom is their narrower
projection, so z_atom is not h_atom with invariants appended, although z_block does begin with
h_block.

Start with :func:`guidance` for which name to use when, and :func:`describe` for widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

__all__ = [
    # the vocabulary
    "REPRESENTATIONS",
    "POOLING",
    "LEGACY_FIELD_NAMES",
    "RepresentationSpec",
    "Usage",
    "USAGE",
    # asking the model for one
    "get",
    "get_many",
    "available",
    # pooling
    "pool_blocks",
    "pooled_width",
    "component_normalize",
    # documentation you can print
    "describe",
    "describe_batch_sensitivity",
    "guidance",
    # extraction to a file, and the command line behind `python -m atomica.representations`
    "load_model",
    "embed_items",
    "embed_dataset",
    "group_batches",
    "write_embeddings",
    "main",
    "cli",
]


# --------------------------------------------------------------------------------- the vocabulary
@dataclass(frozen=True)
class RepresentationSpec:
    """One row of the table in the module docstring."""

    name: str
    paper_symbol: str
    level: str          # atom | block | graph | interface
    family: str         # h | z
    needs_pool: bool    # graph- and interface-level z only
    needs_segment: bool  # interface level only
    description: str


def _spec(name, paper_symbol, level, family, needs_pool, needs_segment, description):
    return RepresentationSpec(name, paper_symbol, level, family, needs_pool, needs_segment,
                              description)


REPRESENTATIONS: Dict[str, RepresentationSpec] = {
    s.name: s for s in (
        _spec("h_atom", "h_a^atom", "atom", "h", False, False,
              "Atom-level scalar readout after out_ffn."),
        _spec("h_block", "h_b^block", "block", "h", False, False,
              "Block-level scalar readout after out_ffn. The per-residue vector used for "
              "block-to-block comparisons in Fig. 4a,d."),
        _spec("h_graph", "h_i^graph", "graph", "h", False, False,
              "Learned attention pooling over the graph's blocks."),
        _spec("h_interface", "h_A^interface", "interface", "h", False, True,
              "The same learned attention pooling restricted to one molecule's blocks. The "
              "representation used for the cross-modality retrieval in Fig. 4e,f."),
        _spec("z_atom", "z_a^atom", "atom", "z", False, False,
              "Atom-level invariants: the 0e and 0o channels concatenated with the within-degree "
              "Gram entries of the l>0 channels."),
        _spec("z_block", "z_b^block", "block", "z", False, False,
              "Block-level invariants: h_block, the block's own Gram entries, and the mean and "
              "standard deviation of z_atom over the block's atoms."),
        _spec("z_graph", "z_i^graph", "graph", "z", True, False,
              "z_block pooled over the graph's blocks by one of the two fixed rules."),
        _spec("z_interface", "z_i^interface", "interface", "z", True, True,
              "z_block pooled over one molecule's blocks by one of the two fixed rules."),
    )
}

POOLING = ("mean_std_global", "mean_component_normalized")

#: Model attribute and older column names, mapped to the name to use instead.
LEGACY_FIELD_NAMES = {
    "unit_repr": "h_atom",
    "block_repr": "h_block",
    "graph_repr": "h_graph",
    "atom_invariant_repr": "z_atom",
    "block_invariant_repr": "z_block",
    "graph_invariant_repr": "z_graph",
}


# ------------------------------------------------------------------------------------- pooling
def component_normalize(x: torch.Tensor, component_dims: Sequence[int]) -> torch.Tensor:
    """L2-normalize each part of the last axis separately, then re-concatenate.

    ``component_dims`` are the part widths, in order, summing to ``x.shape[-1]``. Normalizing per
    part gives each one equal weight in a cosine; it uses the same columns either way.
    """
    total = int(sum(component_dims))
    if total != x.shape[-1]:
        raise ValueError(f"component_dims sum to {total} but the vector is {x.shape[-1]} wide")
    out, off = [], 0
    for width in component_dims:
        part = x[..., off:off + int(width)]
        out.append(part / part.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        off += int(width)
    return torch.cat(out, dim=-1)


def _segment_stats(x, index, n, want_std):
    """Mean, and optionally population standard deviation, of ``x`` grouped by ``index``."""
    from .utils.scatter import scatter_mean
    mean = scatter_mean(x, index, dim=0, dim_size=n)
    if not want_std:
        return mean, None
    mean_sq = scatter_mean(x * x, index, dim=0, dim_size=n)
    return mean, (mean_sq - mean * mean).clamp_min(0).sqrt()


def pool_blocks(x: torch.Tensor, batch_id: torch.Tensor, is_global: torch.Tensor,
                mode: str, component_dims: Optional[Dict[str, int]] = None,
                keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pool per-block features into one vector per graph by one of the two rules.

    ``is_global`` marks the global block node, excluded from the mean and standard deviation and
    supplied separately by ``mean_std_global``. ``keep`` restricts the pool to a subset of blocks,
    which is how the interface level is built.
    """
    if mode not in POOLING:
        raise ValueError(f"unknown pooling {mode!r}; expected one of {list(POOLING)}")
    n = int(batch_id.max().item()) + 1 if batch_id.numel() else 0
    ordinary = ~is_global
    if keep is not None:
        ordinary = ordinary & keep
    if not bool(ordinary.any()):
        raise ValueError("no blocks left to pool; check the segment selection")

    mean, std = _segment_stats(x[ordinary], batch_id[ordinary], n,
                               want_std=(mode == "mean_std_global"))
    if mode == "mean_component_normalized":
        if not component_dims:
            raise ValueError("mean_component_normalized needs component_dims; pass the model's "
                             "invariant_component_dims()")
        return component_normalize(mean, list(component_dims.values()))

    # A multi-segment complex has one global node per segment; the term is the first of them, and
    # an interface reuses its graph's.
    glob = _first_global(x, batch_id, is_global, n, fallback=mean)
    return torch.cat([mean, std, glob], dim=-1)


def _first_global(x, batch_id, is_global, n, fallback):
    """The first global block node per graph, falling back for graphs that have none."""
    if not bool(is_global.any()):
        return fallback
    rows = torch.arange(x.shape[0], device=x.device)[is_global]
    which = batch_id[is_global]
    # batch_id is non-decreasing, so scattering in reverse leaves the FIRST row per graph in place
    picked = torch.full((n,), -1, dtype=torch.long, device=x.device)
    picked.scatter_(0, which.flip(0), rows.flip(0))
    out = fallback.clone()
    found = picked >= 0
    if bool(found.any()):
        out[found] = x[picked[found]]
    return out


def pooled_width(block_width: int, mode: str) -> int:
    """Width of the pooled vector for a given block width and pooling rule."""
    if mode not in POOLING:
        raise ValueError(f"unknown pooling {mode!r}; expected one of {list(POOLING)}")
    return block_width * (3 if mode == "mean_std_global" else 1)


# ---------------------------------------------------------------------------------- the accessor
def _batch_size(batch) -> int:
    return int(batch["lengths"].shape[0])


def _segment_mask(batch, segment: int) -> torch.Tensor:
    seg = batch["segment_ids"]
    mask = seg == segment
    if not bool(mask.any()):
        present = sorted(set(seg.tolist()))
        raise ValueError(f"segment {segment} is empty; this batch has segments {present}")
    return mask


def get(model, batch, name: str, *, pool: Optional[str] = None, segment: Optional[int] = None,
        return_value=None, allow_batched_attention: bool = False) -> torch.Tensor:
    """Return one named representation for one collated batch.

    ``pool`` is required for z_graph and z_interface, ``segment`` for h_interface and z_interface.
    Pass ``return_value`` to reuse a forward pass you have already run, or use :func:`get_many`.
    ``allow_batched_attention`` opts h_graph and h_interface into batch sizes above one.
    """
    if name not in REPRESENTATIONS:
        hint = LEGACY_FIELD_NAMES.get(name)
        extra = f" Did you mean {hint!r}?" if hint else ""
        raise KeyError(f"unknown representation {name!r}; expected one of "
                       f"{list(REPRESENTATIONS)}.{extra}")
    spec = REPRESENTATIONS[name]

    if spec.needs_pool and pool is None:
        raise ValueError(
            f"{name} needs an explicit pool. Use 'mean_std_global' when a head will be trained on "
            f"top, or 'mean_component_normalized' to compare frozen representations directly. The "
            f"two are not comparable, so there is deliberately no default.")
    if pool is not None and not spec.needs_pool:
        raise ValueError(f"{name} takes no pooling argument")
    if spec.needs_segment and segment is None:
        raise ValueError(f"{name} needs segment=0 or segment=1 to say which molecule to pool")
    if segment is not None and not spec.needs_segment:
        raise ValueError(f"{name} is not an interface-level representation, so it takes no segment")

    uses_attention = spec.family == "h" and spec.level in ("graph", "interface")
    if uses_attention and _batch_size(batch) > 1 and not allow_batched_attention:
        raise ValueError(
            f"{name} pools over an array padded to the batch, so its value depends on batch "
            f"composition. Use batch size 1, or pass allow_batched_attention=True and pin the "
            f"batch size and item order for every comparison.")

    if return_value is None:
        want_z = spec.family == "z"
        return_value = model.infer(batch, return_invariant_repr=want_z,
                                   invariant_pool=pool) if want_z else model.infer(batch)

    if spec.family == "h":
        if spec.level == "atom":
            return return_value.unit_repr
        if spec.level == "block":
            return return_value.block_repr
        if spec.level == "graph":
            return return_value.graph_repr
        mask = _segment_mask(batch, segment)
        return model.attention_pooling(return_value.block_repr[mask],
                                       return_value.batch_id[mask])

    if return_value.block_invariant_repr is None:
        raise ValueError("this forward pass did not build the z family; call "
                         "model.infer(batch, return_invariant_repr=True)")
    if spec.level == "atom":
        return return_value.atom_invariant_repr
    if spec.level == "block":
        return return_value.block_invariant_repr

    is_global = batch["B"] == model.global_block_id
    keep = _segment_mask(batch, segment) if spec.level == "interface" else None
    return pool_blocks(return_value.block_invariant_repr, return_value.batch_id, is_global,
                       mode=pool, component_dims=model.invariant_component_dims(), keep=keep)


def get_many(model, batch, names: Sequence[str], *, pool: Optional[str] = None,
             segment: Optional[int] = None,
             allow_batched_attention: bool = False) -> Dict[str, torch.Tensor]:
    """Several representations from as few forward passes as possible.

    One pass covers the h family; the z family needs a second, since it reads the raw irreps.
    """
    names = list(names)
    unknown = [n for n in names if n not in REPRESENTATIONS]
    if unknown:
        raise KeyError(f"unknown representations {unknown}; expected keys of REPRESENTATIONS")
    out, cache = {}, {}
    for name in names:
        spec = REPRESENTATIONS[name]
        want_z = spec.family == "z"
        key = ("z", pool) if want_z else ("h", None)
        if key not in cache:
            cache[key] = (model.infer(batch, return_invariant_repr=True, invariant_pool=pool)
                          if want_z else model.infer(batch))
        # pool and segment go only to the names that take them, so one call can mix levels
        out[name] = get(model, batch, name,
                        pool=pool if spec.needs_pool else None,
                        segment=segment if spec.needs_segment else None,
                        return_value=cache[key],
                        allow_batched_attention=allow_batched_attention)
    return out


# ------------------------------------------------------------------------------------ description
def available() -> Sequence[str]:
    """The representation names, in table order."""
    return tuple(REPRESENTATIONS)


def describe(model=None) -> str:
    """A printable table of what can be extracted, with widths if a model is supplied."""
    rows = [("name", "paper", "level", "family", "width")]
    widths = {}
    if model is not None:
        z_block = int(sum(model.invariant_component_dims().values()))
        h_width = int(model.top_encoder.encoder.ns)
        pooled = (f"{pooled_width(z_block, 'mean_std_global')} or "
                  f"{pooled_width(z_block, 'mean_component_normalized')}")
        widths = {
            "h_atom": str(h_width), "h_block": str(h_width),
            "h_graph": str(h_width), "h_interface": str(h_width),
            "z_atom": str(int(model._irrep_invariants().n_descriptor)),
            "z_block": str(z_block),
            "z_graph": pooled, "z_interface": pooled,
        }
    for spec in REPRESENTATIONS.values():
        rows.append((spec.name, spec.paper_symbol, spec.level, spec.family,
                     widths.get(spec.name, "?")))
    pad = [max(len(r[i]) for r in rows) for i in range(5)]
    lines = ["  ".join(cell.ljust(pad[i]) for i, cell in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * p for p in pad))
    if model is not None:
        gmp = bool(getattr(model, "global_message_passing", False))
        lines.append("")
        lines.append(f"global_message_passing={gmp}: h_graph "
                     f"{'includes' if gmp else 'excludes'} the global block node.")
    return "\n".join(lines)


def describe_batch_sensitivity() -> str:
    """How batch composition affects the vectors, and how to keep runs reproducible."""
    return (
        "The per-block attention runs over an array padded to the largest block in the batch, "
        "counted in atoms, so a structure's vectors depend on what shares its batch. A structure "
        "matches its batch-size-1 value whenever nothing else in the batch has a larger block.\n\n"
        "embed_dataset() therefore batches only structures that share a largest block, which "
        "reproduces batch-size-1 values: over the seven example complexes h_atom, h_block, z_block "
        "and z_graph come back bit-identical and h_graph to 9e-8, where one ungrouped batch of "
        "seven moves z_block by up to 3.6. Splitting such a batch changes nothing, so the "
        "out-of-memory retry is safe.\n\n"
        "h_graph and h_interface pool over a second padded array, sized in blocks, which the "
        "grouping does not control; on some checkpoints that moves them by up to 4.2e-2. Use batch "
        "size 1 for those two names, or pin the batch size and the item order."
    )


# ------------------------------------------------------------------- which one do I want, and why
@dataclass(frozen=True)
class Usage:
    """One row of the guidance table; ``choice`` names the pooling rule when it matters."""

    choice: str
    when: str
    paper_use: str


#: When each choice is right, and where the paper uses it.
USAGE: Sequence[Usage] = (
    Usage("h_atom",
          "You want one vector per atom and the scalar channels are enough.",
          "PCA of the mean embedding per chemical element, Fig. 2."),
    Usage("h_block",
          "You are comparing single residues or single chemical fragments to each other with a "
          "cosine.",
          "PCA of the mean embedding per block type, Fig. 2. Distances between inhibitor blocks "
          "and peptide blocks, Fig. 4a,d."),
    Usage("h_graph",
          "You want one small vector per complex to visualize or to browse by nearest neighbour, "
          "and every vector in the comparison was produced at the same batch size.",
          "UMAP of 2,105,459 complexes, Fig. 2."),
    Usage("h_interface",
          "You are comparing one molecule's side of a complex against another molecule's side, "
          "including across modalities.",
          "Inhibitor queries ranked against protein B surface patches, Fig. 4e,f."),
    Usage("z_atom",
          "Rarely on its own. It is a component of z_block, and it is the level to use when you "
          "need per-atom geometry rather than per-atom chemistry.",
          "Not reported on its own."),
    Usage("z_block",
          "You are training a head that predicts something for every residue, or probing what a "
          "single block's environment encodes.",
          "Residue-level RNA-Protein and RNA-Site tasks, Fig. 3. Metal coordination-number and "
          "geometry probes, Table S1."),
    Usage("z_graph, pool='mean_std_global'",
          "You are training any head on top of a whole complex or pocket. The default for "
          "downstream tasks.",
          "RNA-GO, RNA-Ligand and MaSIF-ligand, Fig. 3. ATP versus ADP pocket discrimination, "
          "Fig. 3i."),
    Usage("z_graph, pool='mean_component_normalized'",
          "You are comparing frozen complexes directly with a cosine or a distance and training "
          "nothing.",
          "Same-ligand pocket retrieval, Fig. 3j and Table S2."),
    Usage("z_interface, pool='mean_std_global'",
          "The head-training case above, restricted to the blocks of one molecule.",
          "Alternative interface representation, Table S7."),
    Usage("z_interface, pool='mean_component_normalized'",
          "The frozen-comparison case above, restricted to the blocks of one molecule.",
          "ATOMICAScore: the cosine between this readout over the ligand blocks and the same "
          "readout with one interface block masked, Fig. 2."),
)

_TWO_QUESTIONS = """\
Two questions decide the answer.

1. Will anything be trained on top of the vector?
   Yes -> the z family with mean_std_global. The pooling is parameter-free, so the head stays the
          only fitted part.
   No  -> the z family with mean_component_normalized to compare whole complexes, or the h family
          to compare single blocks or to visualize.

2. Are you describing the whole complex, one molecule in it, one residue, or one atom?
   That picks the level: graph, interface, block, atom. An interface vector is pooled after
   message passing over the whole complex, so it stays aware of the partner.

h keeps only the 32 lambda = 0 numbers; z turns the higher-degree channels into invariants a plain
head can read. In the MaSIF-ligand setup the same head reaches 0.589 accuracy on h_block and 0.837
on the 1792-wide z_block."""


def guidance(width: int = 98) -> str:
    """The 'which representation do I want' table, as printable text."""
    import textwrap

    lines = ["Which ATOMICA representation to ask for", "=" * 39, "", _TWO_QUESTIONS, "",
             "-" * width, ""]
    for use in USAGE:
        lines.append(use.choice)
        for label, body in (("use when", use.when), ("in the paper", use.paper_use)):
            wrapped = textwrap.wrap(body, width=max(30, width - 20))
            lines.append(f"    {label:<14}{wrapped[0]}")
            lines.extend(" " * 18 + piece for piece in wrapped[1:])
        lines.append("")
    lines.append("Any comparison must use one name, one pooling rule and one checkpoint "
                 "throughout.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- extraction
def load_model(model_config: str, model_weights: str):
    """Load a checkpoint, returning ``(model, dataset_class)``.

    A ProteinInterfaceModel checkpoint is unwrapped to the encoder inside it.
    """
    import json

    with open(model_config, "r") as handle:
        model_type = json.load(handle).get("model_type")
    if model_type in ("PredictionModel", "DenoisePretrainModel"):
        from .data.dataset import PDBDataset
        from .models.prediction_model import PredictionModel
        model = PredictionModel.load_from_config_and_weights(model_config, model_weights)
        return model, PDBDataset
    if model_type == "ProteinInterfaceModel":
        from .data.dataset import ProtInterfaceDataset
        from .models.prot_interface_model import ProteinInterfaceModel
        wrapper = ProteinInterfaceModel.load_from_config_and_weights(model_config, model_weights)
        return wrapper.prot_model, ProtInterfaceDataset
    raise NotImplementedError(f"no representation extractor for model_type {model_type!r}")


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(value, device) for value in obj)
    return obj


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. Check that the "
            "installed torch build matches your CUDA driver (see setup/README.md), or pass "
            "--device cpu.")
    return device


@torch.no_grad()
def embed_items(model, items: Sequence[dict], names: Sequence[str], *,
                pool: Optional[str] = None, segment: Optional[int] = None,
                device: str = "cpu", allow_batched_attention: bool = False,
                drop_global_block: bool = False, data_key: str = "data") -> list:
    """One dict per input structure, holding every requested representation.

    ``items`` are dataset entries with an ``id`` and a processed structure under ``data_key``.
    Atom- and block-level rows are sliced back out of the batch per structure.
    """
    import numpy as np

    from .data.dataset import PDBDataset

    structures = [item[data_key] for item in items]
    batch = _to_device(PDBDataset.collate_fn(structures), device)
    vectors = get_many(model, batch, names, pool=pool, segment=segment,
                       allow_batched_attention=allow_batched_attention)
    vectors = {name: value.detach().cpu().numpy() for name, value in vectors.items()}

    rows, first_atom, first_block = [], 0, 0
    for i, (item, structure) in enumerate(zip(items, structures)):
        n_atoms, n_blocks = len(structure["A"]), len(structure["B"])
        block_ids = np.asarray(structure["B"])
        keep = block_ids != model.global_block_id if drop_global_block else None
        row = {
            "id": item["id"],
            "block_id": (block_ids[keep] if keep is not None else block_ids).tolist(),
            "atom_id": list(structure["A"]),
        }
        for name in names:
            level = REPRESENTATIONS[name].level
            values = vectors[name]
            if level == "atom":
                row[name] = values[first_atom:first_atom + n_atoms]
            elif level == "block":
                block_rows = values[first_block:first_block + n_blocks]
                row[name] = block_rows[keep] if keep is not None else block_rows
            else:
                row[name] = values[i]
        rows.append(row)
        first_atom += n_atoms
        first_block += n_blocks
    return rows


def group_batches(items: Sequence[dict], batch_size: int, *, data_key: str = "data",
                  group_by_max_block: bool = True, atom_budget: Optional[int] = None) -> list:
    """Split items into batches, by default grouping structures that share a largest block.

    That keeps each structure's padding width the same as it would be alone, so a batched run
    reproduces a batch-size-1 run and splitting a batch changes nothing. ``atom_budget`` caps the
    atoms per batch. Returns ``[(indices, items), ...]``; the indices restore the input order.
    """
    n = len(items)
    if batch_size <= 1:
        return [([i], [items[i]]) for i in range(n)]
    if not group_by_max_block:
        return [(list(range(s, min(s + batch_size, n))), list(items[s:s + batch_size]))
                for s in range(0, n, batch_size)]

    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[int(max(items[i][data_key]["block_lengths"]))].append(i)

    def atoms(i):
        return len(items[i][data_key]["A"])

    out = []
    for width in sorted(groups):
        current, used = [], 0
        for i in groups[width]:
            over_budget = atom_budget is not None and current and used + atoms(i) > atom_budget
            if len(current) == batch_size or over_budget:
                out.append((current, [items[j] for j in current]))
                current, used = [], 0
            current.append(i)
            used += atoms(i)
        if current:
            out.append((current, [items[j] for j in current]))
    return out


def embed_dataset(model, dataset, names: Sequence[str], *, pool: Optional[str] = None,
                  segment: Optional[int] = None, batch_size: int = 1, device: str = "cpu",
                  allow_batched_attention: bool = False, drop_global_block: bool = False,
                  data_key: str = "data", group_by_max_block: bool = True,
                  atom_budget: Optional[int] = None, progress: bool = True,
                  strict: bool = False) -> list:
    """Embed a whole dataset, returning rows in the dataset's own order.

    ``group_by_max_block`` batches only structures that share a largest block, so the result
    matches a batch-size-1 run; turn it off only to reproduce a file made without it, and then
    hold the batch size and item order fixed. ``atom_budget`` caps the atoms per batch, which only
    splits a batch and changes no vector.

    On failure the default retries a batch one structure at a time and skips a structure that
    still fails, which returns fewer rows than the dataset has. ``strict=True`` raises instead.
    """
    items = dataset.data
    batches = group_batches(items, batch_size, data_key=data_key,
                            group_by_max_block=group_by_max_block, atom_budget=atom_budget)
    if progress:
        try:
            from tqdm import tqdm
            batches = tqdm(batches, desc=f"embedding {len(items)} structures")
        except ImportError:
            pass

    by_index = {}
    for indices, chunk in batches:
        try:
            by_index.update(zip(indices, embed_items(
                model, chunk, names, pool=pool, segment=segment, device=device,
                allow_batched_attention=allow_batched_attention,
                drop_global_block=drop_global_block, data_key=data_key)))
        except Exception as error:  # noqa: BLE001 - re-raised below unless it is a recoverable OOM
            if strict:
                raise RuntimeError(
                    f"failed on a batch of {len(chunk)} structures, first id "
                    f"{chunk[0].get('id', '?')!r}: {error}\n"
                    f"strict=True, so nothing is retried or skipped. Lower batch_size or "
                    f"atom_budget and re-run.") from error
            if "out of memory" not in str(error).lower() or len(chunk) == 1:
                raise
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            note = ("same largest block, so one at a time gives the same vectors"
                    if group_by_max_block else
                    "without grouping these get a different pad width from the rest of the run")
            print(f"out of memory on a batch of {len(chunk)}; retrying one at a time: {note}")
            for index, item in zip(indices, chunk):
                try:
                    by_index[index] = embed_items(
                        model, [item], names, pool=pool, segment=segment, device=device,
                        allow_batched_attention=True, drop_global_block=drop_global_block,
                        data_key=data_key)[0]
                except Exception as item_error:  # noqa: BLE001 - one bad structure must not stop
                    print(f"skipping {item['id']}: {item_error}")
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
    return [by_index[i] for i in sorted(by_index)]


def write_embeddings(rows: Sequence[dict], output_path: str) -> None:
    """Write to ``.parquet`` or ``.pkl``. Parquet needs plain lists, pickle keeps numpy arrays."""
    import pickle

    import numpy as np

    if output_path.endswith(".parquet"):
        import pandas as pd

        frame = pd.DataFrame([
            {key: (value.tolist() if isinstance(value, np.ndarray) else value)
             for key, value in row.items()}
            for row in rows
        ])
        frame.to_parquet(output_path, index=False)
    else:
        with open(output_path, "wb") as handle:
            pickle.dump(list(rows), handle)
    print(f"wrote {len(rows)} structures to {output_path}")


# ------------------------------------------------------------------------------------ command line
def _parse_names(raw: str) -> list:
    names = [name.strip() for name in raw.replace(" ", ",").split(",") if name.strip()]
    unknown = [name for name in names if name not in REPRESENTATIONS]
    if unknown:
        raise SystemExit(f"unknown representation(s) {unknown}. Choose from "
                         f"{list(REPRESENTATIONS)}, or run --guidance.")
    if not names:
        raise SystemExit("--representations was empty")
    return names


def main(args) -> None:
    """Extract named representations for a processed dataset and write them to one file."""
    if getattr(args, "model_ckpt", None):
        from .utils import pickled_checkpoint_error
        raise pickled_checkpoint_error(args.model_ckpt, "--model_config", "--model_weights")

    # --guidance and --describe work with or without a checkpoint.
    if getattr(args, "guidance", False) or getattr(args, "describe", False):
        if getattr(args, "guidance", False):
            print(guidance())
            print()
        model = None
        if args.model_config and args.model_weights:
            model, _ = load_model(args.model_config, args.model_weights)
        else:
            print("no checkpoint given; pass --model_config and --model_weights for widths\n")
        print(describe(model))
        return

    # Validate before loading a checkpoint, so a wrong flag fails immediately.
    names = _parse_names(args.representations)
    specs = [REPRESENTATIONS[name] for name in names]
    if any(spec.needs_pool for spec in specs) and not args.pool:
        raise SystemExit(
            "--pool is required for z_graph and z_interface: mean_std_global to train a head on "
            "these vectors, mean_component_normalized to compare them with a cosine.")
    if any(spec.needs_segment for spec in specs) and args.segment is None:
        raise SystemExit("--segment is required for h_interface and z_interface: 0 is the "
                         "receptor, 1 the ligand or partner chain.")
    if not (args.model_config and args.model_weights):
        raise SystemExit("both --model_config and --model_weights are required")
    model, dataset_class = load_model(args.model_config, args.model_weights)

    batch_size = args.batch_size
    uses_attention = [spec.name for spec in specs
                      if spec.family == "h" and spec.level in ("graph", "interface")]
    # getattr so a caller that builds its own namespace keeps working when an option is added
    group_by_max_block = getattr(args, "group_by_max_block", True)
    atom_budget = getattr(args, "atom_budget", None)
    strict = getattr(args, "strict", False)
    if batch_size > 1 and group_by_max_block:
        print(f"--batch_size {batch_size}, grouping structures that share a largest block, so "
              f"these vectors match a batch-size-1 run")
    elif batch_size > 1:
        print(f"note: --batch_size {batch_size} without grouping, so the vectors depend on the "
              f"batch size and the file order; hold both fixed across runs you will compare")
    if uses_attention and batch_size > 1 and not args.allow_batched_attention:
        print(f"{', '.join(uses_attention)} pools over a second padded array that the grouping "
              f"does not cover, so batch size is set to 1; pass --allow_batched_attention to keep "
              f"--batch_size {batch_size}")
        batch_size = 1

    device = _resolve_device(getattr(args, "device", "auto"))
    model = model.to(device)
    dataset = dataset_class(args.data_path)
    data_key = "prot_data" if dataset_class.__name__ == "ProtInterfaceDataset" else "data"

    print(f"device {device}, {len(dataset.data)} structures, batch size {batch_size}")
    for spec in specs:
        line = f"  {spec.name} ({spec.paper_symbol})"
        if spec.needs_pool:
            line += f", pool {args.pool}"
        if spec.needs_segment:
            line += f", segment {args.segment}"
        print(f"{line}: {spec.description}")

    rows = embed_dataset(model, dataset, names, pool=args.pool, segment=args.segment,
                         batch_size=batch_size, device=device,
                         allow_batched_attention=args.allow_batched_attention,
                         drop_global_block=args.drop_global_block, data_key=data_key,
                         group_by_max_block=group_by_max_block,
                         atom_budget=atom_budget, strict=strict)
    write_embeddings(rows, args.output_path)


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m atomica.representations",
        description="Extract named ATOMICA representations for processed structures.",
        epilog="Run --guidance for what each representation is for, or --describe for widths.")
    parser.add_argument("--model_config", type=str, default=None,
                        help="path of the model config json")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="path of the model weights .pt")
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="no longer loadable; use --model_config and --model_weights")
    parser.add_argument("--data_path", type=str, default=None,
                        help="processed structures from atomica.data.process_pdbs")
    parser.add_argument("--output_path", type=str, default=None,
                        help="where to write the representations (.parquet or .pkl)")
    parser.add_argument("--representations", type=str, default="h_block,h_graph",
                        help="comma-separated names: " + ", ".join(REPRESENTATIONS))
    parser.add_argument("--pool", type=str, default=None, choices=list(POOLING),
                        help="required for z_graph and z_interface")
    parser.add_argument("--segment", type=int, default=None,
                        help="required for h_interface and z_interface; 0 is the receptor and 1 "
                             "the ligand or partner chain")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="above 1, batches group structures that share a largest block so "
                             "the vectors still match a batch-size-1 run")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no_group_by_max_block", dest="group_by_max_block", action="store_false",
                        help="batch in file order instead; the vectors then depend on the batch "
                             "size and the file order")
    parser.add_argument("--atom_budget", type=int, default=None,
                        help="cap the atoms per batch when a batch does not fit in memory; "
                             "changes no vector")
    parser.add_argument("--strict", action="store_true",
                        help="raise on the first failed batch instead of skipping structures")
    parser.add_argument("--drop_global_block", action="store_true",
                        help="drop the global block node from block-level output")
    parser.add_argument("--allow_batched_attention", action="store_true",
                        help="keep --batch_size for h_graph and h_interface")
    parser.add_argument("--guidance", action="store_true",
                        help="print which representation to use when, then exit")
    parser.add_argument("--describe", action="store_true",
                        help="print the representation table with this checkpoint's widths")
    args = parser.parse_args(argv)
    if not (args.guidance or args.describe):
        missing = [flag for flag, value in (("--data_path", args.data_path),
                                            ("--output_path", args.output_path)) if not value]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def cli():
    """Console entry point for ``atomica-representations``."""
    main(parse_args())


if __name__ == "__main__":
    cli()
