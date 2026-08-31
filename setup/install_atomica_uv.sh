#!/bin/bash
# ATOMICA installation with uv.
#
# Usage:
#   ./install_atomica_uv.sh                      # CUDA 12.8 (default)
#   CUDA=cu130 ./install_atomica_uv.sh           # CUDA 13.0
#   CUDA=cu118 ./install_atomica_uv.sh           # CUDA 11.8
#   CUDA=cpu   ./install_atomica_uv.sh           # CPU only
#
# Pick the CUDA version your NVIDIA driver supports; `nvidia-smi` prints the
# driver version, and setup/README.md maps drivers to CUDA versions. The
# PyTorch wheel bundles its own CUDA runtime, so no CUDA toolkit needs to be
# installed on the host.

set -euo pipefail

CUDA="${CUDA:-cu128}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ENVPATH="${ENVPATH:-$(pwd)/atomica-env}"
ATOMICA_DIR="${ATOMICA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

echo "======================================"
echo "ATOMICA Installation (uv)"
echo "======================================"
echo "CUDA build:  $CUDA"
echo "Python:      $PYTHON_VERSION"
echo "Environment: $ENVPATH"
echo "Repository:  $ATOMICA_DIR"
echo ""

echo "Step 1: Creating virtual environment..."
uv venv "$ENVPATH" --python "$PYTHON_VERSION"
PY="$ENVPATH/bin/python"
uv pip install --python "$PY" pip setuptools wheel
echo ""

echo "Step 2: Installing PyTorch ($CUDA)..."
uv pip install --python "$PY" torch --index-url "https://download.pytorch.org/whl/$CUDA"
echo ""

echo "Step 3: Installing ATOMICA and its dependencies..."
uv pip install --python "$PY" -e "$ATOMICA_DIR[dev]"
echo ""

echo "======================================"
echo "Verifying"
echo "======================================"
"$PY" -c "
import torch
print('torch          ', torch.__version__)
print('built for CUDA ', torch.version.cuda)
print('CUDA available ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            ', torch.cuda.get_device_name(0))
"
"$PY" -c "import atomica; print('atomica imports OK')"
"$PY" -c "
from atomica.utils.scatter import TORCH_SCATTER_AVAILABLE
print('torch_scatter in use:', TORCH_SCATTER_AVAILABLE, '(optional; pure-PyTorch path used when False)')
"
"$ENVPATH/bin/atomica-embeddings" --help > /dev/null && echo "atomica-embeddings OK"
"$ENVPATH/bin/atomica-train" --help > /dev/null && echo "atomica-train OK"
echo ""

echo "Step 4: Running tests..."
(cd "$ATOMICA_DIR" && "$ENVPATH/bin/pytest" tests/ -q)
echo ""

echo "======================================"
echo "Installation complete"
echo "======================================"
echo "Activate with:  source $ENVPATH/bin/activate"
echo ""
echo "Torsion-denoising pretraining additionally needs torch-cluster:"
echo "  TORCH=\$($PY -c \"import torch; print(torch.__version__.split('+')[0])\")"
echo "  pip install torch-cluster -f https://data.pyg.org/whl/torch-\${TORCH}+$CUDA.html"
