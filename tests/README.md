# ATOMICA Tests

This directory contains tests for the ATOMICA package.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_import.py
```

To run with verbose output:
```bash
pytest -v tests/
```

## Test Structure

- `test_import.py`: Basic import tests to ensure the package is properly installed
- More tests to be added for specific functionality

## Adding New Tests

When adding new functionality to ATOMICA, please add corresponding tests to ensure reliability and prevent regressions.
