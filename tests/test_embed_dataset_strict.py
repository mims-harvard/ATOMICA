"""``embed_dataset(strict=True)`` refuses to repair a failed batch.

The grouping fix removed the batch-size dependence, and
``tests/test_representations.py::test_grouped_batching_reproduces_batch_of_one`` covers that. What
grouping does not fix is the other half of the default behaviour: a structure that still fails
after the retry is reported and skipped, so the function returns fewer rows than the dataset has
without raising. That is silent, and it is not benign on an ordered dataset.

It happened while building tutorial 4: a contended GPU returned 420 of the 467 MaSIF-ligand test
pockets, and because that parquet is ordered by class, all 47 losses fell inside ADP. The run would
have produced a full-looking seven-class table computed on a third of one class.

``strict=True`` makes the first failure raise instead. Anything whose output will be compared
against a published number should pass it.
"""
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain")
CONFIG = os.path.join(CKPT, "pretrain_model_config.json")
WEIGHTS = os.path.join(CKPT, "pretrain_model_weights.pt")
DATA = os.path.join(REPO, "data", "example", "example_processed_data.parquet")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CONFIG) and os.path.exists(WEIGHTS) and os.path.exists(DATA)),
    reason="needs the pretrain checkpoint and the processed example data",
)


def test_strict_raises_where_the_default_returns_a_short_result(monkeypatch):
    from atomica import representations as R
    from atomica.data.dataset import PDBDataset
    from atomica.models.prediction_model import PredictionModel

    model = PredictionModel.load_from_config_and_weights(CONFIG, WEIGHTS)
    model.eval()
    dataset = PDBDataset(DATA)

    def always_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB")

    monkeypatch.setattr(R, "embed_items", always_oom)
    # A batch of one cannot be retried any smaller, so the permissive path re-raises there instead
    # of skipping. To reach the skip on every batch: turn grouping off, so batches are formed in
    # file order rather than by block width, and drop to an even number of structures so the last
    # batch is not a singleton. The example set has seven.
    dataset.data = dataset.data[:6]
    kw = dict(pool="mean_std_global", batch_size=2, group_by_max_block=False, progress=False)

    with pytest.raises(RuntimeError, match="strict=True"):
        R.embed_dataset(model, dataset, ["z_graph"], strict=True, **kw)

    # The permissive path retries one at a time, skips what still fails, and hands back a short
    # result with no exception. That silence is the whole reason strict exists.
    rows = R.embed_dataset(model, dataset, ["z_graph"], strict=False, **kw)
    assert len(rows) < len(dataset.data)

    # strict must also fire on a grouped run, where a failing batch may be a singleton.
    with pytest.raises(RuntimeError, match="strict=True"):
        R.embed_dataset(model, dataset, ["z_graph"], strict=True, pool="mean_std_global",
                        batch_size=2, group_by_max_block=True, progress=False)
