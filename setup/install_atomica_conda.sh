#!/bin/bash
# ATOMICA installation with conda/mamba.
#
# Usage:
#   ./install_atomica_conda.sh                   # CUDA 12.8 (default)
#   CUDA=cu130 ./install_atomica_conda.sh        # CUDA 13.0
#   CUDA=cu118 ./install_atomica_conda.sh        # CUDA 11.8
#   CUDA=cpu   ./install_atomica_conda.sh        # CPU only
#
# conda creates the Python environment; PyTorch comes from the pip wheel index,
# which covers more CUDA versions than the pytorch conda channel and bundles
# its own CUDA runtime. `nvidia-smi` prints your driver version, and
# setup/README.md maps drivers to CUDA versions.

set -euo pipefail

CUDA="${CUDA:-cu128}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ENVNAME="${ENVNAME:-atomica-env}"
ATOMICA_DIR="${ATOMICA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONDA="${CONDA:-mamba}"
command -v "$CONDA" >/dev/null 2>&1 || CONDA=conda

echo "======================================"
echo "ATOMICA Installation (conda/mamba)"
echo "======================================"
echo "Solver:      $CONDA"
echo "CUDA build:  $CUDA"
echo "Python:      $PYTHON_VERSION"
echo "Environment: $ENVNAME"
echo "Repository:  $ATOMICA_DIR"
echo ""

echo "Step 1: Creating environment..."
"$CONDA" create -y -n "$ENVNAME" -c conda-forge "python=$PYTHON_VERSION" pip
eval "$(conda shell.bash hook)"
conda activate "$ENVNAME"
echo ""

echo "Step 2: Installing PyTorch ($CUDA)..."
pip install torch --index-url "https://download.pytorch.org/whl/$CUDA"
echo ""

echo "Step 3: Installing ATOMICA and its dependencies..."
pip install -e "$ATOMICA_DIR[dev]"
echo ""

echo "======================================"
echo "Verifying"
echo "======================================"
python -c "
import torch
print('torch          ', torch.__version__)
print('built for CUDA ', torch.version.cuda)
print('CUDA available ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            ', torch.cuda.get_device_name(0))
"
python -c "import atomica; print('atomica imports OK')"
python -c "
from atomica.utils.scatter import TORCH_SCATTER_AVAILABLE
print('torch_scatter in use:', TORCH_SCATTER_AVAILABLE, '(optional; pure-PyTorch path used when False)')
"
atomica-embeddings --help > /dev/null && echo "atomica-embeddings OK"
atomica-train --help > /dev/null && echo "atomica-train OK"
echo ""

echo "Step 4: Running tests..."
(cd "$ATOMICA_DIR" && pytest tests/ -q)
echo ""

echo "======================================"
echo "Installation complete"
echo "======================================"
echo "Activate with:  conda activate $ENVNAME"
echo ""
echo "Torsion-denoising pretraining additionally needs torch-cluster:"
echo "  TORCH=\$(python -c \"import torch; print(torch.__version__.split('+')[0])\")"
echo "  pip install torch-cluster -f https://data.pyg.org/whl/torch-\${TORCH}+$CUDA.html"
