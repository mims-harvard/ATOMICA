"""Regression tests for the MaskedNodeModel options used by the nucleotide-recovery fine-tunes.

Every option added for `tutorials/11_sequence_recovery` claims two things:

  1. it is the identity at initialisation, so a fine-tune that enables it still starts from
     exactly the pretrained weights, and
  2. leaving it off reproduces the released pretrained model bit for bit.

Both are testable, and this file tests them on a small synthetic complex rather than asserting
them in a comment. It also covers `force_masked_blocks`, which is what lets several models be
scored on one shared set of masked positions, and the `kv_mask` argument to `CrossAttention`.
"""
import numpy as np
import pytest
import torch

from atomica.data.pdb_utils import VOCAB
from atomica.models.masking_model import MaskedNodeModel
from atomica.models.tools import CrossAttention


BASE = dict(atom_hidden_size=16, block_hidden_size=16, edge_size=16, k_neighbors=4,
            n_layers=2, num_masked_block_classes=436, global_message_passing=True,
            fragmentation_method="PS_300")


@pytest.fixture(scope="module", autouse=True)
def _tokenizer():
    VOCAB.load_tokenizer("PS_300")


def make_batch(seed=0, n_blocks_per_segment=6, atoms_per_block=4):
    """A minimal two-segment complex in the layout MaskedNodeModel.forward expects."""
    rng = np.random.default_rng(seed)
    n_blocks = 2 * n_blocks_per_segment
    block_lengths = [atoms_per_block] * n_blocks
    n_atoms = sum(block_lengths)

    aa_idx = [VOCAB.symbol_to_idx(x[0]) for x in VOCAB.aas]
    B = torch.tensor([aa_idx[int(rng.integers(len(aa_idx)))] for _ in range(n_blocks)])
    A = torch.tensor([int(rng.integers(1, VOCAB.get_num_atom_type() - 1))
                      for _ in range(n_atoms)])
    X = torch.tensor(rng.normal(scale=6.0, size=(n_atoms, 3)), dtype=torch.float)
    segment_ids = torch.tensor([0] * n_blocks_per_segment + [1] * n_blocks_per_segment)

    masked = torch.zeros(n_blocks, dtype=torch.bool)
    masked[[1, 3, 8]] = True
    return dict(Z=X, B=B, A=A,
                block_lengths=torch.tensor(block_lengths),
                lengths=torch.tensor([n_blocks]),
                segment_ids=segment_ids,
                masked_blocks=masked,
                masked_labels=torch.tensor([0, 5, 11]))


def logits(model, batch):
    model.eval()
    with torch.no_grad():
        _, out = model(**batch, return_logits=True)
    return out


@pytest.mark.parametrize("option", [
    {"masked_affine": True},
    {"bottom_repr_scale": True},
    {"top_pair_geom": True},
    {"top_long_range_edge_length": 24.0},
])
def test_option_is_identity_at_initialisation(option):
    """Enabling an option must not change the output of the weights it starts from.

    Both models are built with the same seed and the flagged one then loads the plain model's
    state dict, so the only difference between them is the new component. Every one of these is
    designed to be a no-op until training moves it, and a fine-tune that started from a perturbed
    model would not be a fine-tune of the released checkpoint.
    """
    batch = make_batch()

    torch.manual_seed(1234)
    plain = MaskedNodeModel(**BASE)
    torch.manual_seed(1234)
    flagged = MaskedNodeModel(**BASE, **option)

    missing, unexpected = flagged.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected, f"plain model carries keys the flagged one lacks: {unexpected}"
    assert missing, "the option added no parameters, so this test proves nothing"

    torch.testing.assert_close(logits(plain, batch), logits(flagged, batch),
                               rtol=1e-5, atol=1e-6)


def test_defaults_match_the_published_configuration():
    """A model built with no options carries exactly the released model's config values."""
    cfg = MaskedNodeModel(**BASE).get_config()
    assert cfg["top_max_edge_length"] == 5
    assert cfg["top_long_range_edge_length"] is None
    assert cfg["na_loss_weight"] == 1.0
    assert cfg["attn_pad_mask"] is False
    assert cfg["masked_affine"] is False
    assert cfg["bottom_repr_scale"] is False
    assert cfg["top_pair_geom"] is False


def test_config_round_trips_through_the_constructor():
    model = MaskedNodeModel(**BASE, attn_pad_mask=True, na_loss_weight=20.0,
                            top_max_edge_length=12.0, masked_affine=True)
    cfg = dict(model.get_config())
    cfg.pop("model_type")
    cfg["block_hidden_size"] = cfg.pop("block_hidden_size")
    rebuilt = MaskedNodeModel(**cfg)
    rebuilt.load_state_dict(model.state_dict())
    batch = make_batch()
    torch.testing.assert_close(logits(model, batch), logits(rebuilt, batch))


def test_inactive_unknown_config_keys_are_dropped_but_active_ones_raise():
    """Configs written by a research branch can carry switches this class does not implement.

    Where such a key is off it describes a component that was never built, so dropping it loads
    the identical model. Where it is on it is a real architectural difference and must not be
    ignored.
    """
    cfg = MaskedNodeModel(**BASE).get_config()
    cfg.pop("model_type")
    kept = MaskedNodeModel._drop_inactive_config_keys({**cfg, "some_ablation": False})
    assert "some_ablation" not in kept
    MaskedNodeModel(**kept)   # still constructible

    with pytest.raises(ValueError, match="some_ablation"):
        MaskedNodeModel._drop_inactive_config_keys({**cfg, "some_ablation": True})


def test_na_loss_weight_only_moves_the_loss():
    """Upweighting nucleotide positions changes the training loss and not the predictions."""
    batch = make_batch()
    batch["masked_labels"] = torch.tensor([0, 24, 26])   # one amino acid, two RNA bases

    torch.manual_seed(7)
    plain = MaskedNodeModel(**BASE)
    torch.manual_seed(7)
    weighted = MaskedNodeModel(**BASE, na_loss_weight=20.0)
    weighted.load_state_dict(plain.state_dict())

    torch.testing.assert_close(logits(plain, batch), logits(weighted, batch))
    with torch.no_grad():
        loss_plain, _ = plain(**batch)
        loss_weighted, _ = weighted(**batch)
    assert not torch.isclose(loss_plain, loss_weighted), \
        "na_loss_weight left the loss unchanged, so it is not being applied"


def test_force_masked_blocks_pins_the_mask(tmp_path):
    """The dataset must mask exactly the blocks it is handed, and resume sampling when reset."""
    from atomica.data.dataset_pretrain import PretrainMaskedDataset
    import pickle

    # one item, enough maskable blocks that a random draw would almost never match our pin
    batch = make_batch(n_blocks_per_segment=10)
    n_blocks = len(batch["B"])
    item = {"id": "synthetic_0", "data": {
        "X": batch["Z"].tolist(), "B": batch["B"].tolist(), "A": batch["A"].tolist(),
        "atom_positions": [0] * len(batch["A"]),
        "block_lengths": batch["block_lengths"].tolist(),
        "segment_ids": batch["segment_ids"].tolist(),
    }}
    path = tmp_path / "synthetic.pkl"
    with open(path, "wb") as fh:
        pickle.dump([item], fh)

    ds = PretrainMaskedDataset(
        data_file=str(path), mask_proportion=0.1,
        mask_token=VOCAB.symbol_to_idx(VOCAB.MASK),
        vocab_to_mask=[VOCAB.symbol_to_idx(x[0])
                       for x in VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags],
        atom_mask_token=VOCAB.get_atom_mask_idx())
    assert len(ds) == 1, "the synthetic item was dropped as having no maskable blocks"

    pinned = [2, 5, 9]
    ds.force_masked_blocks = pinned
    got = [i for i, m in enumerate(ds[0]["masked_blocks"]) if m]
    assert got == pinned

    ds.force_masked_blocks = None
    n_sampled = sum(ds[0]["masked_blocks"])
    assert n_sampled == max(1, int(0.1 * n_blocks)), \
        "clearing force_masked_blocks did not restore the sampled draw"


def test_cross_attention_kv_mask_ignores_padding():
    """Without kv_mask the softmax runs over batchify's zero padding, which is not neutral.

    key_proj and value_proj carry biases, so a zero pad row maps to a nonzero constant and takes
    attention mass away from the real entries. Passing kv_mask must give the same answer as
    running the unpadded sequence on its own.
    """
    torch.manual_seed(0)
    attn = CrossAttention(dim_query=8, dim_kv=8, dim_out=8, num_heads=2, dropout=0.0).eval()
    query = torch.randn(1, 1, 8)
    real = torch.randn(1, 3, 8)
    padded = torch.cat([real, torch.zeros(1, 5, 8)], dim=1)
    mask = torch.tensor([[True, True, True, False, False, False, False, False]])

    with torch.no_grad():
        unpadded_out = attn(query, real)
        masked_out = attn(query, padded, kv_mask=mask)
        unmasked_out = attn(query, padded)

    torch.testing.assert_close(unpadded_out, masked_out, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(unpadded_out, unmasked_out, atol=1e-4), \
        "padding made no difference here, so this test cannot detect the bug it guards"
