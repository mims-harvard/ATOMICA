#!/bin/bash
# ATOMICA Installation Script
# This script sets up a complete environment for ATOMICA

set -e  # Exit on error

# Increase timeout for large downloads (default is 30s)
export UV_HTTP_TIMEOUT=300

ENVPATH=/path/to/atomica-env
ATOMICA_DIR=/path/to/atomica-repo

echo "======================================"
echo "ATOMICA Installation Script"
echo "======================================"
echo "Environment: $ENVPATH"
echo "ATOMICA directory: $ATOMICA_DIR"
echo ""

# Step 1: Create virtual environment with uv (much faster than conda)
echo "Step 1: Creating virtual environment..."
if [ -d "$ENVPATH" ]; then
    echo "Environment already exists, removing..."
    rm -rf "$ENVPATH"
fi
uv venv "$ENVPATH" --python 3.11
echo "✓ Virtual environment created"
echo ""

# Activate environment
source "$ENVPATH/bin/activate"

# Step 2: Install pip and basic tools
echo "Step 2: Installing pip and setuptools..."
uv pip install --python "$ENVPATH/bin/python" pip setuptools wheel
echo "✓ pip installed"
echo ""

# Step 3: Install numpy first (some packages depend on it)
echo "Step 3: Installing numpy..."
uv pip install --python "$ENVPATH/bin/python" numpy==1.26.4
echo "✓ numpy installed"
echo ""

# Step 4: Install PyTorch with CUDA 11.8
echo "Step 4: Installing PyTorch 2.1.1 with CUDA 11.8..."
uv pip install --python "$ENVPATH/bin/python" \
    torch==2.1.1 \
    --extra-index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch installed"
echo ""

# Step 5: Install PyTorch Geometric dependencies
echo "Step 5: Installing torch-scatter and torch-cluster..."
echo "These need to be installed from PyTorch Geometric wheels..."
uv pip install --python "$ENVPATH/bin/python" \
    torch-scatter==2.1.2 \
    torch-cluster==1.6.3 \
    --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html
echo "✓ PyTorch Geometric dependencies installed"
echo ""

# Step 6: Install ATOMICA in editable mode
# This will automatically install all remaining dependencies from pyproject.toml
echo "Step 6: Installing ATOMICA package and remaining dependencies..."
echo "This reads dependencies from pyproject.toml..."
cd "$ATOMICA_DIR"
uv pip install --python "$ENVPATH/bin/python" -e ".[dev]"
echo "✓ ATOMICA and all dependencies installed"
echo ""

# Step 7: Verify installation
echo "======================================"
echo "Verifying Installation"
echo "======================================"
echo ""

echo "Testing console scripts..."
if "$ENVPATH/bin/atomica-embeddings" --help > /dev/null 2>&1; then
    echo "✓ atomica-embeddings command works"
else
    echo "✗ atomica-embeddings command failed"
    exit 1
fi

if "$ENVPATH/bin/atomica-train" --help > /dev/null 2>&1; then
    echo "✓ atomica-train command works"
else
    echo "✗ atomica-train command failed"
    exit 1
fi
echo ""

echo "Testing Python imports..."
"$ENVPATH/bin/python" -c "import atomica; print('✓ atomica module imports successfully')" || exit 1
"$ENVPATH/bin/python" -c "from atomica.train import cli; print('✓ atomica.train.cli imports successfully')" || exit 1
"$ENVPATH/bin/python" -c "from atomica.get_embeddings import cli; print('✓ atomica.get_embeddings.cli imports successfully')" || exit 1
echo ""

# Step 8: Run tests
echo "Step 8: Running tests..."
cd "$ATOMICA_DIR"
"$ENVPATH/bin/pytest" tests/ -v
echo ""

# Step 9: Summary
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "ATOMICA has been successfully installed at:"
echo "  Environment: $ENVPATH"
echo "  Package: $ATOMICA_DIR"
echo ""
echo "To activate the environment, run:"
echo "  source $ENVPATH/bin/activate"
echo ""
echo "Available commands:"
echo "  atomica-train --help"
echo "  atomica-embeddings --help"
echo ""
echo "To run tests:"
echo "  pytest $ATOMICA_DIR/tests/"
echo ""
