# ATOMICA container image. Build from the repository root:
#     docker build -f setup/Dockerfile -t atomica:cu128 .
#     docker run --gpus all -v "$PWD":/work -w /work atomica:cu128 atomica-embeddings --help
#
# By default this installs the pinned stack in setup/requirements-lock.txt on a
# digest-pinned base image, so rebuilds reproduce the same versions. The lock
# pins a CUDA 12.8 build of PyTorch; for another CUDA version, resolve fresh:
#     docker build -f setup/Dockerfile --build-arg USE_LOCK=0 \
#         --build-arg TORCH_VERSION=2.9.1 \
#         --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
#         -t atomica:cu130 .
# and regenerate the lock from the result if you want it reproducible in turn:
#     docker run --rm atomica:cu130 pip freeze --exclude-editable > setup/requirements-lock.txt
#
# The PyTorch wheel bundles its own CUDA runtime, so the base is a plain Python
# image and only the host driver is needed (`--gpus all`, or `--nv` for Apptainer).

# python:3.12.13-slim-bookworm, pinned by digest so the base cannot drift.
FROM python@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579

ARG USE_LOCK=1
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 for OpenMP; libxrender1, libxext6 and libexpat1 are all needed by
# RDKit's rdMolDraw2D, which atomica.data.tokenizer imports at module load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libxrender1 \
        libxext6 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/atomica
COPY pyproject.toml README.md ./
COPY setup/requirements-lock.txt ./setup/
COPY src ./src

# With USE_LOCK=1 every version comes from the lock file, including torch.
# With USE_LOCK=0 torch comes from TORCH_INDEX_URL and the rest is resolved
# from pyproject.toml.
RUN pip install --upgrade pip setuptools wheel && \
    if [ "$USE_LOCK" = "1" ]; then \
        pip install -r setup/requirements-lock.txt --extra-index-url "${TORCH_INDEX_URL}" && \
        pip install --no-deps . ; \
    else \
        pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" && \
        pip install ".[dev]" ; \
    fi

WORKDIR /work

# Fail the build rather than ship a broken image.
RUN python -c "import torch, atomica; \
from atomica.utils.scatter import TORCH_SCATTER_AVAILABLE; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('torch_scatter in use:', TORCH_SCATTER_AVAILABLE)"

CMD ["python", "-c", "import torch; print('ATOMICA image. torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"]
