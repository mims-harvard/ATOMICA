"""ATOMICAScore computes the readout the paper defines, over the blocks the paper scopes it to.

The heavy checks run on 6hrg_A_A_ZN, a 14-block zinc site, so the whole file finishes on CPU in
well under a minute.

Skipped unless the pretrain checkpoint and the processed example data are both present.
"""
import os

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain")
CONFIG = os.path.join(CKPT, "pretrain_model_config.json")
WEIGHTS = os.path.join(CKPT, "pretrain_model_weights.pt")
DATA = os.path.join(REPO, "data", "example", "example_processed_data.parquet")

SMALL = "6hrg_A_A_ZN"       # protein + Zn ion, 9 amino-acid residue blocks
LARGER = "6llw_A_A_UDP"     # protein + UDP, 29 amino-acid residue blocks
NO_PROTEIN = "4yaz_A_A_4BW"  # RNA riboswitch: no amino-acid residue block at all

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CONFIG) and os.path.exists(WEIGHTS) and os.path.exists(DATA)),
    reason="needs the pretrain checkpoint and the processed example data",
)


@pytest.fixture(scope="module")
def model():
    from atomica.models.prediction_model import PredictionModel

    return PredictionModel.load_from_config_and_weights(CONFIG, WEIGHTS).eval()


@pytest.fixture(scope="module")
def dataset():
    from atomica.data.dataset import PDBDataset

    return PDBDataset(DATA)


def item(dataset, name):
    return dataset[dataset.indexes.index(name)]


def _model_device(model):
    from atomica.interaction_profiler.interact_score import _model_device as impl

    return impl(model)


# ------------------------------------------------------------------------------- which blocks
def test_amino_acid_ids_are_the_twenty_standard_residues():
    from atomica.data.pdb_utils import VOCAB
    from atomica.interaction_profiler.interact_score import amino_acid_block_ids

    ids = amino_acid_block_ids()
    assert len(ids) == 20
    assert {VOCAB.idx_to_symbol(i) for i in ids} == {s for s, _ in VOCAB.aas}


def test_ligand_segment_is_the_one_without_amino_acids(dataset):
    from atomica.interaction_profiler.interact_score import find_ligand_segment

    for name in (SMALL, LARGER):
        assert find_ligand_segment(item(dataset, name)) == 1


def test_scorable_blocks_are_amino_acids_outside_the_ligand(dataset):
    from atomica.data.pdb_utils import VOCAB
    from atomica.interaction_profiler.interact_score import (
        amino_acid_block_ids, scorable_blocks)

    data = item(dataset, LARGER)
    blocks = scorable_blocks(data, ligand_segment=1)
    amino = amino_acid_block_ids()
    assert blocks, "the example has amino-acid blocks"
    assert all(int(data["B"][b]) in amino for b in blocks)
    assert all(int(data["segment_ids"][b]) == 0 for b in blocks)
    # nothing scorable is a global node or a ligand fragment
    assert VOCAB.symbol_to_idx(VOCAB.GLB) not in {int(data["B"][b]) for b in blocks}


def test_a_complex_with_no_amino_acid_block_scores_nothing(model, dataset):
    """ATOMICAScore is scoped to amino-acid residue blocks, so an RNA receptor yields no ranking."""
    from atomica.interaction_profiler.interact_score import atomica_score, scorable_blocks

    data = item(dataset, NO_PROTEIN)
    assert scorable_blocks(data, ligand_segment=1) == []
    result = atomica_score(model, data, ligand_segment=1)
    assert len(result) == 0
    assert result.score.shape == (0,)


# ---------------------------------------------------------------------------------- the score
def test_score_shape_direction_and_readout(model, dataset):
    from atomica.interaction_profiler.interact_score import (
        BATCH_SIZE, POOLING, READOUT, atomica_score, scorable_blocks)

    data = item(dataset, SMALL)
    result = atomica_score(model, data)
    assert result.readout == READOUT == "z_interface"
    assert result.pooling == POOLING == "mean_component_normalized"
    assert result.batch_size == BATCH_SIZE
    assert result.block_idx == scorable_blocks(data, ligand_segment=1)
    assert result.score.shape == (len(result.block_idx),)
    assert np.isfinite(result.score).all()
    assert ((result.score >= -1.0) & (result.score <= 1.0)).all()
    # importance flips the sign, and ranking is ascending cosine
    assert np.allclose(result.importance, -result.score)
    assert result.ranking()[0] == result.block_idx[int(np.argmin(result.score))]


def test_same_batch_size_repeats_exactly_on_cpu(model, dataset):
    """Fix the batch size and a CPU run repeats bit for bit.

    The fixture keeps the model on CPU deliberately. On CUDA this assertion would not hold, and
    the reason is not the batch size: some CUDA reductions are not order-deterministic, so two
    runs at one batch size differ by about 1e-7 just as two runs at different batch sizes do.
    ``torch.use_deterministic_algorithms(True)`` removes that, measured exactly 0.
    """
    from atomica.interaction_profiler.interact_score import atomica_score

    assert _model_device(model) == "cpu", "this test characterises the CPU path"
    data = item(dataset, SMALL)
    first = atomica_score(model, data, batch_size=4)
    second = atomica_score(model, data, batch_size=4)
    assert first.block_idx == second.block_idx
    assert np.array_equal(first.score, second.score)
    assert first.batch_size == second.batch_size == 4


def test_batch_size_is_recorded_and_batches_are_uniform(model, dataset):
    """A result must say what batch size produced it, and every pass must be that wide."""
    from atomica.interaction_profiler import interact_score as m

    data = item(dataset, SMALL)
    widths = []
    original = m._readout_fixed

    def spy(model_, graphs, ligand_segment, device, batch_size):
        widths.append(len(graphs))
        return original(model_, graphs, ligand_segment, device, batch_size)

    m._readout_fixed = spy
    try:
        result = m.atomica_score(model, data, batch_size=4)
    finally:
        m._readout_fixed = original

    assert result.batch_size == 4
    # 9 residues at 3 masked graphs per pass is 3 passes, every one padded to exactly 4 graphs
    assert widths == [4, 4, 4], widths


def test_reproduces_the_published_pairwise_construction(model, dataset):
    """The batched score must equal one-pass-per-residue on ``[intact, masked]``.

    This is the regression test for the cross-attention pad-width bug that
    ``representations.group_batches`` addresses for dataset extraction. The pad width is the
    largest block in the batch, so the intact graph is kept in slot 0 of every pass; the largest
    block is then the intact graph's whatever is masked, which is the width the published
    implementation used when it collated exactly ``[intact, masked]``.
    """
    import torch

    from atomica.data.dataset import PDBDataset
    from atomica.interaction_profiler.interact_score import (
        POOLING, READOUT, atomica_score, mask_block)
    from atomica import representations as R

    data = item(dataset, SMALL)
    batched = atomica_score(model, data, batch_size=8)

    pairwise = []
    for block in batched.block_idx:
        collated = PDBDataset.collate_fn([data, mask_block(data, block)])
        with torch.no_grad():
            vectors = R.get(model, collated, READOUT, pool=POOLING, segment=1)
        pairwise.append(float(torch.nn.functional.cosine_similarity(
            vectors[0], vectors[1], dim=-1)))

    assert np.allclose(batched.score, np.array(pairwise), rtol=0, atol=1e-6)


def test_every_pass_sees_the_intact_graphs_largest_block(model, dataset):
    """Whatever is masked, the pad width of a pass is the intact graph's largest block."""
    from atomica.data.dataset import PDBDataset
    from atomica.interaction_profiler import interact_score as m

    data = item(dataset, SMALL)
    intact_max = int(max(data["block_lengths"]))
    widths = []
    original = m._readout_fixed

    def spy(model_, graphs, ligand_segment, device, batch_size):
        widths.append(max(int(max(g["block_lengths"])) for g in graphs))
        return original(model_, graphs, ligand_segment, device, batch_size)

    m._readout_fixed = spy
    try:
        m.atomica_score(model, data, batch_size=4)
    finally:
        m._readout_fixed = original

    assert widths, "no forward pass was made"
    assert set(widths) == {intact_max}, (widths, intact_max)


def test_batch_size_below_two_is_rejected(model, dataset):
    """Slot 0 holds the intact graph, so a batch of one cannot carry a masked graph."""
    from atomica.interaction_profiler.interact_score import atomica_score

    with pytest.raises(ValueError, match="at least 2"):
        atomica_score(model, item(dataset, SMALL), batch_size=1)


def test_batch_size_does_not_change_the_cpu_score(model, dataset):
    """On CPU the readout does not depend on batch composition at all.

    Edges are built per graph and the z pooling is a scatter over real blocks with no padding, so
    batch composition carries no semantic information. Measured across batch sizes 1 to 29 on
    6llw the CPU scores are bit-identical; on GPU they move by at most 2.4e-07, which is the
    hardware's own run-to-run noise rather than a batch effect, and the induced ranking is
    unchanged either way (Spearman 1.00000, zero rank changes).
    """
    from atomica.interaction_profiler.interact_score import atomica_score

    data = item(dataset, SMALL)
    narrow = atomica_score(model, data, batch_size=2)
    wide = atomica_score(model, data, batch_size=10)
    assert narrow.block_idx == wide.block_idx
    if _model_device(model) == "cpu":
        assert np.array_equal(narrow.score, wide.score)
    else:
        assert np.abs(narrow.score - wide.score).max() < 1e-5
    # what actually matters downstream is that the ranking is untouched
    assert narrow.ranking() == wide.ranking()


def test_readout_matches_representations_module(model, dataset):
    """atomica_score must score the same vector representations.py hands out under that name."""
    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    from atomica.interaction_profiler.interact_score import _readout

    data = item(dataset, SMALL)
    batch = PDBDataset.collate_fn([data])
    with torch.no_grad():
        direct = R.get(model, batch, "z_interface", pool="mean_component_normalized", segment=1)
    via_module = _readout(model, [data], 1, "cpu")   # batch of 1 is fine for a direct comparison
    assert torch.allclose(direct, via_module, atol=1e-6)
    # the three parts of z_block, each L2-normalized, so the whole vector has norm sqrt(3)
    assert direct.shape[-1] == sum(model.invariant_component_dims().values())
    assert abs(float(direct.norm()) - np.sqrt(3)) < 1e-4


def test_masking_only_changes_the_masked_block(dataset):
    from atomica.data.pdb_utils import VOCAB
    from atomica.interaction_profiler.interact_score import mask_block

    data = item(dataset, SMALL)
    target = 1
    masked = mask_block(data, target)
    assert masked["B"][target] == VOCAB.symbol_to_idx(VOCAB.MASK)
    assert masked["block_lengths"][target] == 1
    assert list(masked["B"][:target]) == list(np.asarray(data["B"])[:target])
    assert list(masked["B"][target + 1:]) == list(np.asarray(data["B"])[target + 1:])
    # the original is untouched
    assert int(np.asarray(data["B"])[target]) != VOCAB.symbol_to_idx(VOCAB.MASK)


# -------------------------------------------------------------------------------- the metrics
def test_precision_at_k_is_a_fraction_not_a_count():
    """The published convention changed; k is the denominator."""
    from atomica.interaction_profiler.interact_score import precision_at_k

    importance = np.arange(20)[::-1].astype(float)   # index 0 ranked first
    labels = np.zeros(20, bool)
    labels[:4] = True                                 # the 4 top-ranked are positive
    assert precision_at_k(importance, labels, k=10) == pytest.approx(0.4)
    assert precision_at_k(importance, labels, k=4) == pytest.approx(1.0)
    # all ten top-ranked positive is 1.0, never 10
    labels[:10] = True
    assert precision_at_k(importance, labels, k=10) == pytest.approx(1.0)


def test_precision_at_k_rejects_mismatched_lengths():
    from atomica.interaction_profiler.interact_score import precision_at_k

    with pytest.raises(ValueError):
        precision_at_k(np.zeros(5), np.zeros(4, bool))


def test_auroc_matches_sklearn_and_is_nan_for_one_class():
    from atomica.interaction_profiler.interact_score import auroc

    rng = np.random.default_rng(0)
    importance = rng.normal(size=200)
    labels = rng.random(200) < 0.3
    try:
        from sklearn.metrics import roc_auc_score
        assert auroc(importance, labels) == pytest.approx(roc_auc_score(labels, importance))
    except ImportError:
        assert 0.0 <= auroc(importance, labels) <= 1.0
    assert np.isnan(auroc(importance, np.ones(200, bool)))
    assert np.isnan(auroc(importance, np.zeros(200, bool)))
    # a perfect ranking is 1.0
    assert auroc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0], bool)) == pytest.approx(1.0)
