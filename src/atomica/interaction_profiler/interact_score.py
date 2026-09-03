"""ATOMICAScore: how much the ligand's representation depends on each interface residue.

For block ``i`` of an interaction graph ``G``, build ``G \\ i`` by replacing that block with the
mask block and its atoms with a single mask atom, then take

    a_i = cos( r(G), r(G \\ i) )

A low ``a_i`` means masking the residue moved the readout a lot, so the residue matters more. The
readout ``r`` is the component-normalized mean of ``z_block`` over the ligand's blocks, taken from
:mod:`atomica.representations` so that the vector scored here is the one the paper names::

    R.get(model, batch, "z_interface", pool="mean_component_normalized", segment=ligand_segment)

Three choices, each of which changes the number:

* Pool over the ligand, not the whole complex. Masking a block moves a whole-complex pooled vector
  partly by removing that block from the pool; the ligand blocks are never masked, so that
  self-contribution disappears.
* Component-normalize before the cosine. ``z_block`` is three parts whose norms differ several
  fold, and a cosine weights each part by the product of its norms, so without this the
  atom-pooled part takes about 95 percent of the similarity.
* Score amino-acid residue blocks only. A protein block is one residue but a small-molecule block
  is a chemical fragment, so ranking both in one list would not compare like with like. A complex
  with no amino-acid block returns an empty result.

Batching. The intact graph sits in slot 0 of every forward pass and the pass is padded to a fixed
:data:`BATCH_SIZE`. Both matter: the atom-to-block cross-attention pads every block out to the
largest block in the batch, so keeping the intact graph present gives every vector entering a
cosine the same pad width. See ``representations.describe_batch_sensitivity``.

    python -m atomica.interaction_profiler.interact_score \\
        --data_path processed.pkl --output_path scores.jsonl \\
        --model_config config.json --model_weights weights.pt
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Sequence
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .. import representations as R
from ..data.dataset import PDBDataset
from ..data.pdb_utils import VOCAB
from ..models.prediction_model import PredictionModel

__all__ = [
    "READOUT",
    "POOLING",
    "BATCH_SIZE",
    "amino_acid_block_ids",
    "find_ligand_segment",
    "scorable_blocks",
    "mask_block",
    "ATOMICAScoreResult",
    "atomica_score",
    "precision_at_k",
    "auroc",
]

#: The representation the score is computed on, named as :mod:`atomica.representations` names it.
READOUT = "z_interface"
POOLING = "mean_component_normalized"

#: Graphs per forward pass. Fixed and recorded on every result, not sized to fit memory.
BATCH_SIZE = 8


# ------------------------------------------------------------------------------ which blocks
def amino_acid_block_ids() -> frozenset:
    """Vocabulary indices of the 20 standard amino acids.

    These indices sit ahead of the tokenizer-dependent fragment entries in the vocabulary, so they
    do not move when a different fragmentation method is loaded.
    """
    return frozenset(VOCAB.symbol_to_idx(symbol) for symbol, _ in VOCAB.aas)


def find_ligand_segment(data) -> int:
    """Which segment holds the ligand, inferred from the block types.

    The ligand is the segment with no amino-acid blocks. When two segments both qualify, or none
    does, the complex is not a protein--ligand complex in the sense ATOMICAScore is defined for
    and the caller has to say which side to pool.
    """
    blocks = np.asarray(data["B"])
    segments = np.asarray(data["segment_ids"])
    amino = amino_acid_block_ids()
    glb = VOCAB.symbol_to_idx(VOCAB.GLB)
    candidates = []
    for segment in sorted(set(int(s) for s in segments)):
        real = blocks[(segments == segment) & (blocks != glb)]
        if len(real) and not any(int(b) in amino for b in real):
            candidates.append(segment)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "every segment contains amino-acid blocks, so none of them is the ligand. "
            "ATOMICAScore is defined for a complex whose two sides are a protein and a "
            "non-protein ligand; for a protein-protein interface, pass ligand_segment= to say "
            "which side to pool the readout over.")
    raise ValueError(
        f"segments {candidates} all lack amino-acid blocks, so the ligand is ambiguous. Pass "
        f"ligand_segment= to say which one to pool the readout over.")


def scorable_blocks(data, ligand_segment: int) -> List[int]:
    """Indices of the blocks ATOMICAScore masks: amino-acid residues outside the ligand segment.

    Global block nodes, ligand fragments, nucleotides and unknown blocks are all excluded. See the
    module docstring for why the score is restricted to amino-acid residue blocks.
    """
    blocks = np.asarray(data["B"])
    segments = np.asarray(data["segment_ids"])
    amino = amino_acid_block_ids()
    return [i for i in range(len(blocks))
            if int(blocks[i]) in amino and int(segments[i]) != ligand_segment]


# -------------------------------------------------------------------------------- the masking
def mask_block(data, block_idx):
    """A copy of ``data`` with block ``block_idx`` replaced by the mask block and one mask atom.

    The mask atom sits at the block's centroid, so the masked graph keeps the residue's position
    while losing its identity, its size and its internal geometry.
    """
    data = deepcopy(data)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    data["B"][block_idx] = VOCAB.symbol_to_idx(VOCAB.MASK)
    block_start = sum(data["block_lengths"][:block_idx])
    block_end = block_start + data["block_lengths"][block_idx]
    data["block_lengths"][block_idx] = 1
    data["X"] = (data["X"][:block_start]
                 + [np.mean(data["X"][block_start:block_end], axis=0).tolist()]
                 + data["X"][block_end:])
    data["A"] = data["A"][:block_start] + [VOCAB.get_atom_mask_idx()] + data["A"][block_end:]
    data["atom_positions"] = (data["atom_positions"][:block_start]
                              + [VOCAB.get_atom_pos_mask_idx()]
                              + data["atom_positions"][block_end:])
    return data


# --------------------------------------------------------------------------------- the score
@dataclass
class ATOMICAScoreResult:
    """One ATOMICAScore per masked residue block.

    ``score`` is the cosine ``a_i``, so **low means important**. ``importance`` flips the sign so
    that higher means more important, which is the direction the ranking and the metrics use.

    ``batch_size`` records how many graphs were in each forward pass, so a saved result says how
    it was produced.
    """

    block_idx: List[int]
    score: np.ndarray
    ligand_segment: int
    batch_size: int = BATCH_SIZE
    readout: str = READOUT
    pooling: str = POOLING

    @property
    def importance(self) -> np.ndarray:
        return -np.asarray(self.score, dtype=float)

    def ranking(self) -> List[int]:
        """Block indices ordered most important first, i.e. by ascending cosine."""
        order = np.argsort(np.asarray(self.score, dtype=float))
        return [self.block_idx[i] for i in order]

    def __len__(self) -> int:
        return len(self.block_idx)


def _model_device(model) -> str:
    """Where the model's parameters live, so callers need not repeat the device."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(value, device) for value in obj)
    return obj


@torch.no_grad()
def _readout(model, graphs, ligand_segment, device):
    """The paper's readout for each graph in ``graphs``, as one tensor of shape [n_graphs, width].

    Callers must hand this a constant number of graphs. See :data:`BATCH_SIZE` for why.
    """
    batch = _to_device(PDBDataset.collate_fn(graphs), device)
    return R.get(model, batch, READOUT, pool=POOLING, segment=ligand_segment)


def _readout_fixed(model, graphs, ligand_segment, device, batch_size):
    """:func:`_readout` on exactly ``batch_size`` graphs, refusing to silently resize the batch.

    An out-of-memory error is raised rather than retried on a smaller batch. Retrying would
    finish the run, but the scores that came back would have been computed at a batch size the
    caller never chose and the result would not record, which is the failure this whole module is
    arranged to avoid.
    """
    if len(graphs) != batch_size:
        raise AssertionError(f"internal error: {len(graphs)} graphs in a batch of {batch_size}")
    try:
        return _readout(model, graphs, ligand_segment, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError(
            f"out of memory with batch_size={batch_size}. Pass a smaller batch_size to "
            f"atomica_score (or --batch_size on the command line) and record the value you "
            f"used.") from None


def atomica_score(model, data, *, ligand_segment: Optional[int] = None,
                  device: Optional[str] = None,
                  batch_size: int = BATCH_SIZE,
                  blocks: Optional[Sequence[int]] = None) -> ATOMICAScoreResult:
    """ATOMICAScore for every amino-acid residue block of one complex.

    Parameters
    ----------
    model : PredictionModel
        A loaded ATOMICA checkpoint, already on ``device``.
    data : dict
        One processed complex, as ``PDBDataset[i]`` returns it.
    ligand_segment : int, optional
        Which segment to pool the readout over. Inferred by :func:`find_ligand_segment` when
        omitted.
    device : str, optional
        Where to run the forward passes. Defaults to wherever the model's parameters are.
    batch_size : int
        Graphs per forward pass, held constant for every pass including the last, and recorded on
        the result. On CPU the scores do not depend on it at all. On GPU they move by about 1e-7,
        which is the same amount two runs at one batch size move, so it is kernel nondeterminism
        rather than a batch effect. See the REPRODUCIBILITY section of the module docstring.
    blocks : sequence of int, optional
        Score these block indices instead of every amino-acid residue block. The caller is then
        responsible for the scope restriction the module docstring describes.

    Returns
    -------
    ATOMICAScoreResult
        Empty when the complex has no amino-acid residue block, which is the correct answer for a
        complex ATOMICAScore is not defined over rather than an error.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2: slot 0 carries the intact graph so that "
                         "the reference and the perturbed vector are encoded together")
    model.eval()
    device = device or _model_device(model)
    if ligand_segment is None:
        ligand_segment = find_ligand_segment(data)
    targets = list(blocks) if blocks is not None else scorable_blocks(data, ligand_segment)
    if not targets:
        return ATOMICAScoreResult(block_idx=[], score=np.zeros(0), ligand_segment=ligand_segment,
                                  batch_size=batch_size)

    # [intact] + masked graphs, padded back to batch_size with copies of the intact graph.
    per_pass = batch_size - 1
    cosines = []
    for start in range(0, len(targets), per_pass):
        chunk = [mask_block(data, block) for block in targets[start:start + per_pass]]
        n_real = len(chunk)
        graphs = [data] + chunk + [data] * (per_pass - n_real)
        vectors = _readout_fixed(model, graphs, ligand_segment, device, batch_size)
        reference, perturbed = vectors[0], vectors[1:1 + n_real]
        cosines.extend(torch.nn.functional.cosine_similarity(
            reference.unsqueeze(0).expand_as(perturbed), perturbed, dim=-1).tolist())

    return ATOMICAScoreResult(block_idx=targets, score=np.asarray(cosines, dtype=float),
                              ligand_segment=ligand_segment, batch_size=batch_size)


# -------------------------------------------------------------------------------- the metrics
def precision_at_k(importance, labels, k: int = 10) -> float:
    """Fraction of the ``k`` top-ranked residues that carry a positive label.

    ``importance`` is higher-is-more-important, so pass ``result.importance`` rather than
    ``result.score``. The denominator is ``k``, not the number of positives and not the number of
    residues, which makes this a fraction of ten at ``k=10``. An earlier version of this figure
    was reported as a count out of ten; multiply by ten to recover that convention.
    """
    importance = np.asarray(importance, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if len(importance) != len(labels):
        raise ValueError(f"{len(importance)} scores against {len(labels)} labels")
    if len(importance) == 0:
        return float("nan")
    top = np.argsort(-importance)[:k]
    return float(labels[top].sum()) / float(k)


def auroc(importance, labels) -> float:
    """Area under the ROC curve, computed from ranks so that ties are handled by mid-rank.

    Returns NaN when one of the two classes is absent, which is why the paper's evaluation keeps
    only complexes carrying at least one residue of each label.
    """
    from scipy.stats import rankdata

    importance = np.asarray(importance, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(importance)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------------- the script
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute ATOMICAScore for every amino-acid residue block of every complex in "
                    "a processed dataset, and append one JSON record per complex.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Processed dataset (.pkl or .parquet) from atomica.data.process_pdbs")
    parser.add_argument("--output_path", type=str, required=True, help="Output .jsonl")
    parser.add_argument("--model_config", type=str, required=True, help="Model config JSON")
    parser.add_argument("--model_weights", type=str, required=True, help="Model weights .pt")
    parser.add_argument("--ligand_segment", type=int, default=None,
                        help="Segment to pool the readout over; inferred per complex when omitted")
    parser.add_argument("--device", type=str, default=None,
                        help="Defaults to cuda when available, otherwise cpu")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Graphs per forward pass, held constant (default {BATCH_SIZE}). "
                             f"Recorded in every output record; scores from different batch "
                             f"sizes are not bit-comparable.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PredictionModel.load_from_config_and_weights(args.model_config, args.model_weights)
    model = model.to(device).eval()
    dataset = PDBDataset(args.data_path)

    # Appending, and skipping ids already written, so an interrupted run resumes where it stopped
    # instead of duplicating records or starting over.
    done, seen_batch_sizes = set(), set()
    if os.path.exists(args.output_path):
        with open(args.output_path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    done.add(record["id"])
                    seen_batch_sizes.add(record.get("batch_size"))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"{len(done)} complex(es) already in {args.output_path}; they will be skipped")
        mismatched = {b for b in seen_batch_sizes if b is not None and b != args.batch_size}
        if mismatched:
            raise SystemExit(
                f"{args.output_path} already holds scores computed at batch_size "
                f"{sorted(mismatched)}, and this run would append scores at "
                f"{args.batch_size}. Scores from different batch sizes are not bit-comparable, "
                f"so appending would produce a file that cannot be compared with itself. Re-run "
                f"with --batch_size {sorted(mismatched)[0]}, or write to a new file.")

    todo = [i for i in range(len(dataset)) if dataset.indexes[i] not in done]
    skipped = 0
    with open(args.output_path, "a") as handle:
        for i in tqdm(todo, total=len(todo)):
            try:
                result = atomica_score(model, dataset[i], ligand_segment=args.ligand_segment,
                                       device=device, batch_size=args.batch_size)
            except ValueError as error:
                print(f"skipping {dataset.indexes[i]}: {error}")
                skipped += 1
                continue
            if len(result) == 0:
                skipped += 1
                continue
            handle.write(json.dumps({
                "id": dataset.indexes[i],
                "block_idx": result.block_idx,
                "atomica_score": [float(s) for s in result.score],
                "ligand_segment": int(result.ligand_segment),
                "batch_size": int(result.batch_size),
                "readout": f"{result.readout}:{result.pooling}",
            }) + "\n")
            handle.flush()          # a killed run keeps everything it had already scored
    print(f"finished; {len(todo) - skipped} scored, {skipped} skipped with no amino-acid "
          f"residue block or an ambiguous ligand segment")


if __name__ == "__main__":
    main()
