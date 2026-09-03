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
    from atomica.representations import cli as representations_cli

    assert train_cli is not None
    assert representations_cli is not None


def test_representation_cli_requires_a_pooling_rule():
    """The command line must refuse a pooled representation with no rule named, since the two
    rules are not comparable and a silent default makes a saved file unreproducible."""
    import pytest

    from atomica.representations import parse_args

    args = parse_args(["--model_config", "c.json", "--model_weights", "w.pt",
                       "--data_path", "d.parquet", "--output_path", "o.parquet",
                       "--representations", "z_graph"])
    assert args.pool is None
    with pytest.raises(SystemExit, match="--pool is required"):
        from atomica.representations import main
        main(args)
