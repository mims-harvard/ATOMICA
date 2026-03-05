#!/bin/bash
# ATOMICA Installation Script (Mamba version)
# This script sets up a complete environment for ATOMICA using mamba (faster conda replacement)

set -e  # Exit on error

ENVPATH=/path/to/atomica-env
ATOMICA_DIR=/path/to/atomica-repo

echo "======================================"
echo "ATOMICA Installation Script (Mamba)"
echo "======================================"
echo "Environment: $ENVPATH"
echo "ATOMICA directory: $ATOMICA_DIR"
echo ""

# Step 1: Create mamba environment with Python 3.11
echo "Step 1: Creating mamba environment..."
if [ -d "$ENVPATH" ]; then
    echo "Environment already exists, removing..."
    rm -rf "$ENVPATH"
fi
mamba create -p "$ENVPATH" python=3.11 -y
echo "✓ Mamba environment created"
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate "$ENVPATH"

# Step 2: Install PyTorch with CUDA 11.8 from conda
echo "Step 2: Installing PyTorch 2.1.1 with CUDA 11.8..."
echo "Using mamba for faster installation with pre-built binaries..."
mamba install -y pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
echo "✓ PyTorch installed"
echo ""

# Step 3: Install PyTorch Geometric dependencies
echo "Step 3: Installing torch-scatter and torch-cluster..."
echo "Installing from PyTorch Geometric channel..."
# Use pip for PyG dependencies as they're not always available in conda
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
    --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html
echo "✓ PyTorch Geometric dependencies installed"
echo ""

# Step 4: Install remaining packages from environment.yml
echo "Step 4: Installing remaining packages from environment.yml..."
cd "$ATOMICA_DIR"
mamba env update -p "$ENVPATH" -f environment.yml --prune
echo "✓ Packages from environment.yml installed"
echo ""

# Step 5: Install ATOMICA and remaining dependencies
# This will use pip to install ATOMICA and any dependencies not covered by conda
echo "Step 5: Installing ATOMICA package and remaining dependencies..."
echo "This reads dependencies from pyproject.toml..."
cd "$ATOMICA_DIR"
pip install -e ".[dev]"
echo "✓ ATOMICA and all dependencies installed"
echo ""

# Step 6: Verify installation
echo "======================================"
echo "Verifying Installation"
echo "======================================"
echo ""

echo "Testing console scripts..."
if atomica-embeddings --help > /dev/null 2>&1; then
    echo "✓ atomica-embeddings command works"
else
    echo "✗ atomica-embeddings command failed"
    exit 1
fi

if atomica-train --help > /dev/null 2>&1; then
    echo "✓ atomica-train command works"
else
    echo "✗ atomica-train command failed"
    exit 1
fi
echo ""

echo "Testing Python imports..."
python -c "import atomica; print('✓ atomica module imports successfully')" || exit 1
python -c "from atomica.train import cli; print('✓ atomica.train.cli imports successfully')" || exit 1
python -c "from atomica.get_embeddings import cli; print('✓ atomica.get_embeddings.cli imports successfully')" || exit 1
echo ""

# Step 7: Run tests
echo "Step 7: Running tests..."
cd "$ATOMICA_DIR"
pytest tests/ -v
echo ""

# Step 8: Summary
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "ATOMICA has been successfully installed in mamba/conda environment:"
echo "  Environment: $ENVPATH"
echo "  Package: $ATOMICA_DIR"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENVPATH"
echo ""
echo "Available commands:"
echo "  atomica-train --help"
echo "  atomica-embeddings --help"
echo ""
echo "To run tests:"
echo "  pytest $ATOMICA_DIR/tests/"
echo ""
