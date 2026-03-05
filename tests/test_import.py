"""Basic import tests for ATOMICA."""

def test_import_atomica():
    """Test that the atomica package can be imported."""
    import atomica
    assert atomica is not None


def test_import_models():
    """Test that models can be imported."""
    from atomica import models
    assert models is not None


def test_import_data():
    """Test that data utilities can be imported."""
    from atomica import data
    assert data is not None


def test_cli_entry_points():
    """Test that CLI entry points are defined."""
    from atomica.train import cli as train_cli
    from atomica.get_embeddings import cli as embeddings_cli

    assert train_cli is not None
    assert embeddings_cli is not None
