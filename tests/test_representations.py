"""Every representation the paper names can be extracted, has the documented width, and is
invariant to rotation. Also checks that the additive model changes did not move the old outputs.

Skipped unless the pretrain checkpoint and the processed example data are both present.
"""
import json
import os

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain")
CONFIG = os.path.join(CKPT, "pretrain_model_config.json")
WEIGHTS = os.path.join(CKPT, "pretrain_model_weights.pt")
DATA = os.path.join(REPO, "data", "example", "example_processed_data.parquet")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CONFIG) and os.path.exists(WEIGHTS) and os.path.exists(DATA)),
    reason="needs the pretrain checkpoint and the processed example data",
)


@pytest.fixture(scope="module")
def model_and_batch():
    from atomica.models.prediction_model import PredictionModel
    from atomica.data.dataset import PDBDataset

    model = PredictionModel.load_from_config_and_weights(CONFIG, WEIGHTS)
    model.eval()
    ds = PDBDataset(DATA)
    batch = ds.collate_fn([ds[0]])
    return model, batch


def _widths(model):
    dims = model.invariant_component_dims()
    ns = int(model.top_encoder.encoder.ns)
    return ns, int(sum(dims.values())), dims


def test_every_name_resolves(model_and_batch):
    from atomica import representations as R
    model, batch = model_and_batch
    ns, z_block, _ = _widths(model)

    with torch.no_grad():
        got = {
            "h_atom": R.get(model, batch, "h_atom"),
            "h_block": R.get(model, batch, "h_block"),
            "h_graph": R.get(model, batch, "h_graph"),
            "h_interface": R.get(model, batch, "h_interface", segment=0),
            "z_atom": R.get(model, batch, "z_atom"),
            "z_block": R.get(model, batch, "z_block"),
            "z_graph_head": R.get(model, batch, "z_graph", pool="mean_std_global"),
            "z_graph_frozen": R.get(model, batch, "z_graph", pool="mean_component_normalized"),
            "z_interface": R.get(model, batch, "z_interface", pool="mean_std_global", segment=0),
        }
    assert got["h_atom"].shape[-1] == ns
    assert got["h_block"].shape[-1] == ns
    assert got["h_graph"].shape[-1] == ns
    assert got["h_interface"].shape[-1] == ns
    assert got["z_block"].shape[-1] == z_block
    assert got["z_graph_head"].shape[-1] == 3 * z_block
    assert got["z_graph_frozen"].shape[-1] == z_block
    assert got["z_interface"].shape[-1] == 3 * z_block
    for name, t in got.items():
        assert torch.isfinite(t).all(), f"{name} has non-finite entries"


def test_documented_widths_for_the_released_checkpoint(model_and_batch):
    """The numbers quoted in the module docstring and in Methods."""
    model, _ = model_and_batch
    ns, z_block, dims = _widths(model)
    assert ns == 32
    assert dims == {"h_block": 32, "gram": 544, "atom": 1216}
    assert z_block == 1792
    assert model._irrep_invariants().n_descriptor == 608


def test_pool_must_be_named(model_and_batch):
    from atomica import representations as R
    model, batch = model_and_batch
    with pytest.raises(ValueError, match="needs an explicit pool"):
        R.get(model, batch, "z_graph")
    with pytest.raises(ValueError, match="segment"):
        R.get(model, batch, "z_interface", pool="mean_std_global")
    with pytest.raises(KeyError):
        R.get(model, batch, "graph_repr")


def test_rotation_invariance(model_and_batch):
    """Both families are invariant to a global rotation. This is the property the z family exists
    to provide, so it is checked rather than assumed."""
    from atomica import representations as R
    model, batch = model_and_batch

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=batch["X"].dtype))
    if torch.det(q) < 0:
        q[:, 0] *= -1

    rotated = dict(batch)
    rotated["X"] = batch["X"] @ q.T

    names = [("h_block", {}), ("z_block", {}),
             ("z_graph", {"pool": "mean_std_global"}),
             ("z_graph", {"pool": "mean_component_normalized"})]
    with torch.no_grad():
        for name, kw in names:
            a = R.get(model, batch, name, **kw)
            b = R.get(model, rotated, name, **kw)
            assert torch.allclose(a, b, atol=1e-4), f"{name} moved under rotation"


def test_component_normalize_makes_parts_unit_norm(model_and_batch):
    from atomica import representations as R
    model, batch = model_and_batch
    dims = list(model.invariant_component_dims().values())
    with torch.no_grad():
        v = R.get(model, batch, "z_graph", pool="mean_component_normalized")
    off = 0
    for width in dims:
        part = v[..., off:off + width]
        assert torch.allclose(part.norm(dim=-1), torch.ones(v.shape[0]), atol=1e-5)
        off += width


def test_h_family_unchanged_by_the_additive_edits(model_and_batch):
    """The default forward path must return exactly what it returned before the z family was
    added, otherwise every released checkpoint's published numbers move."""
    model, batch = model_and_batch
    with torch.no_grad():
        plain = model.infer(batch)
        withz = model.infer(batch, return_invariant_repr=True)
    assert torch.equal(plain.unit_repr, withz.unit_repr)
    assert torch.equal(plain.block_repr, withz.block_repr)
    assert torch.equal(plain.graph_repr, withz.graph_repr)
    assert plain.block_invariant_repr is None
    assert plain.atom_node_attr is None


def test_legacy_aliases_still_read(model_and_batch):
    model, batch = model_and_batch
    with torch.no_grad():
        rv = model.infer(batch)
    assert torch.equal(rv.atom_scalar_repr, rv.unit_repr)
    assert torch.equal(rv.block_scalar_repr, rv.block_repr)
    assert torch.equal(rv.graph_scalar_repr, rv.graph_repr)


def test_batched_attention_is_refused_by_default(model_and_batch):
    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    model, _ = model_and_batch
    ds = PDBDataset(DATA)
    if len(ds) < 2:
        pytest.skip("needs at least two example structures")
    batch2 = ds.collate_fn([ds[0], ds[1]])
    with pytest.raises(ValueError, match="batch composition"):
        R.get(model, batch2, "h_graph")
    with torch.no_grad():
        ok = R.get(model, batch2, "h_graph", allow_batched_attention=True)
    assert ok.shape[0] == 2


def test_describe_runs(model_and_batch):
    from atomica import representations as R
    model, _ = model_and_batch
    text = R.describe(model)
    assert "h_interface" in text and "z_graph" in text
    assert "global_message_passing" in text
    assert R.available()[0] == "h_atom"


def test_get_many_mixes_levels_that_need_different_arguments(model_and_batch):
    """One call must be able to ask for a plain block vector and an interface vector together."""
    from atomica import representations as R
    model, batch = model_and_batch
    with torch.no_grad():
        got = R.get_many(model, batch, ["h_block", "z_block", "z_interface"],
                         pool="mean_component_normalized", segment=1)
    ns, z_block, _ = _widths(model)
    assert got["h_block"].shape[-1] == ns
    assert got["z_block"].shape[-1] == z_block
    assert got["z_interface"].shape == (1, z_block)


def test_embed_items_slices_each_structure_back_out(model_and_batch):
    """Atom and block rows must be handed back to the structure they came from."""
    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    model, _ = model_and_batch
    ds = PDBDataset(DATA)
    if len(ds) < 2:
        pytest.skip("needs at least two example structures")
    items = ds.data[:2]
    rows = R.embed_items(model, items, ["h_atom", "h_block", "z_graph"], pool="mean_std_global")
    ns, z_block, _ = _widths(model)
    for row, item in zip(rows, items):
        n_atoms, n_blocks = len(item["data"]["A"]), len(item["data"]["B"])
        assert row["id"] == item["id"]
        assert row["h_atom"].shape == (n_atoms, ns)
        assert row["h_block"].shape == (n_blocks, ns)
        assert row["z_graph"].shape == (3 * z_block,)
        assert len(row["block_id"]) == n_blocks


def test_drop_global_block_removes_exactly_the_global_nodes(model_and_batch):
    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    model, _ = model_and_batch
    ds = PDBDataset(DATA)
    item = ds.data[0]
    n_global = sum(1 for b in item["data"]["B"] if b == model.global_block_id)
    assert n_global >= 1
    kept = R.embed_items(model, [item], ["z_block"], drop_global_block=True)[0]
    assert kept["z_block"].shape[0] == len(item["data"]["B"]) - n_global
    assert len(kept["block_id"]) == kept["z_block"].shape[0]
    assert model.global_block_id not in kept["block_id"]


def test_batch_composition_changes_the_vectors(model_and_batch):
    """Documents a property of the released checkpoints rather than asserting a fix.

    Two attention softmaxes run over zero padding sized by the batch, so a structure is identical
    to its batch-of-one value only while no other structure in the batch has a block with more
    atoms than its own largest block. The test exists so that a future change to the padding does
    not pass silently.
    """
    import numpy as np

    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    model, _ = model_and_batch
    ds = PDBDataset(DATA)
    if len(ds) < 2:
        pytest.skip("needs at least two example structures")

    def largest_block(item):
        return max(item["data"]["block_lengths"])

    query = ds.data[0]
    alone = R.embed_items(model, [query], ["z_block"])[0]["z_block"]
    smaller = [it for it in ds.data[1:] if largest_block(it) <= largest_block(query)]
    bigger = [it for it in ds.data[1:] if largest_block(it) > largest_block(query)]

    for partner in smaller[:1]:
        with_partner = R.embed_items(model, [query, partner], ["z_block"])[0]["z_block"]
        assert np.array_equal(alone, with_partner)
    for partner in bigger[:1]:
        with_partner = R.embed_items(model, [query, partner], ["z_block"])[0]["z_block"]
        assert not np.array_equal(alone, with_partner)


def test_guidance_covers_every_name(model_and_batch):
    from atomica import representations as R
    text = R.guidance()
    for name in R.available():
        assert name in text
    for rule in R.POOLING:
        assert rule in text


def test_group_batches_keeps_each_batch_to_one_block_width():
    """Pure logic, no model. Every batch must hold one largest-block value, cover every item once,
    respect the batch size, and respect an atom budget."""
    from atomica import representations as R

    def item(i, largest, atoms):
        return {"id": f"s{i}", "data": {"block_lengths": [1, largest, 2], "A": [0] * atoms}}

    items = [item(0, 14, 100), item(1, 23, 100), item(2, 14, 100),
             item(3, 10, 100), item(4, 23, 100), item(5, 14, 100)]
    widths = lambda idx: {max(items[i]["data"]["block_lengths"]) for i in idx}

    batches = R.group_batches(items, 4)
    assert all(len(widths(idx)) == 1 for idx, _ in batches)
    assert sorted(i for idx, _ in batches for i in idx) == list(range(6))
    assert all(len(idx) <= 4 for idx, _ in batches)

    assert [idx for idx, _ in R.group_batches(items, 1)] == [[i] for i in range(6)]
    assert [idx for idx, _ in R.group_batches(items, 4, group_by_max_block=False)] == \
        [[0, 1, 2, 3], [4, 5]]

    budgeted = R.group_batches(items, 4, atom_budget=250)
    assert all(sum(len(items[i]["data"]["A"]) for i in idx) <= 250 or len(idx) == 1
               for idx, _ in budgeted)
    assert sorted(i for idx, _ in budgeted for i in idx) == list(range(6))


def test_grouped_batching_reproduces_batch_of_one(model_and_batch):
    """The point of the grouping: batch size stops changing the numbers.

    Also covers the out-of-memory retry, which splits a batch into singletons: within one group
    that is the same split, so the retry cannot change a vector.
    """
    import numpy as np

    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    model, _ = model_and_batch
    ds = PDBDataset(DATA)
    if len(ds) < 3:
        pytest.skip("needs at least three example structures")

    names = ["h_block", "z_graph"]
    kw = dict(pool="mean_std_global", progress=False)
    alone = {r["id"]: r for r in R.embed_dataset(model, ds, names, batch_size=1, **kw)}
    grouped = {r["id"]: r for r in R.embed_dataset(model, ds, names, batch_size=len(ds), **kw)}
    naive = {r["id"]: r for r in R.embed_dataset(model, ds, names, batch_size=len(ds),
                                                 group_by_max_block=False, **kw)}
    assert set(grouped) == set(alone)
    for name in names:
        assert max(float(np.abs(alone[i][name] - grouped[i][name]).max()) for i in alone) == 0.0
    # and the ungrouped path is the one that moves, so the test is not vacuous
    assert max(float(np.abs(alone[i]["z_graph"] - naive[i]["z_graph"]).max()) for i in alone) > 0.0


def test_embed_dataset_returns_rows_in_dataset_order(model_and_batch):
    """Grouping reorders the work, so the rows must be put back before they are written."""
    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    model, _ = model_and_batch
    ds = PDBDataset(DATA)
    rows = R.embed_dataset(model, ds, ["h_atom"], batch_size=len(ds), progress=False)
    assert [r["id"] for r in rows] == [item["id"] for item in ds.data]
