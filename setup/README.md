# Installing ATOMICA

ATOMICA runs on any CUDA build of PyTorch from 11.8 through 13.0, and on CPU.
Install the PyTorch wheel matching your NVIDIA driver, then install ATOMICA on
top of it.

## Which CUDA do I need?

Your NVIDIA **driver** sets the ceiling. The CUDA version inside the PyTorch
wheel does not have to match a CUDA toolkit on your system — the wheel bundles
its own CUDA runtime.

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
```

| CUDA wheel | Minimum Linux driver | `--index-url` |
| --- | --- | --- |
| CUDA 13.0 | 580.65 | `https://download.pytorch.org/whl/cu130` |
| CUDA 12.x | 525.60 | `https://download.pytorch.org/whl/cu128` |
| CUDA 11.8 | 520.61 | `https://download.pytorch.org/whl/cu118` |

Take the newest row your driver satisfies. No GPU? Use
`https://download.pytorch.org/whl/cpu` and pass `--device cpu` to
`atomica-representations`.

## Install with uv or pip

```bash
# 1. Create an environment (Python >= 3.10)
uv venv atomica-env --python 3.12
source atomica-env/bin/activate

# 2. Install PyTorch for YOUR CUDA version. This is the only line that changes.
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# 3. Install ATOMICA
git clone https://github.com/mims-harvard/ATOMICA.git
cd ATOMICA
uv pip install -e ".[dev]"
```

Every command works with plain `pip` if you drop the `uv` prefix. That is the
whole installation — `torch-scatter` and `torch-cluster` are not required (see
[optional compiled extensions](#optional-compiled-extensions)).

`setup/install_atomica_uv.sh` runs these steps as a script:

```bash
CUDA=cu130 ./setup/install_atomica_uv.sh
```

## Install with conda / mamba

```bash
mamba env create -f environment.yml
mamba activate atomica-env
```

`environment.yml` creates the environment and installs PyTorch with pip, since
the pip index covers CUDA 11.8-13.0 while the `pytorch` conda channel does not.
It defaults to CUDA 12.8; edit the `--extra-index-url` line to change that.
`setup/install_atomica_conda.sh` does the same as a script and takes
`CUDA=cu130` the same way.

## Container (Docker / Apptainer)

A container pins the whole stack, which is the most durable option for
long-term reproducibility. The base image is fixed by digest and all packages
come from [`requirements-lock.txt`](requirements-lock.txt), so rebuilds
reproduce the same versions.

```bash
docker build -f setup/Dockerfile -t atomica:cu128 .
docker run --gpus all -v "$PWD":/work -w /work atomica:cu128 \
    atomica-representations --help
```

`podman build` works in place of `docker build`. The lock pins a CUDA 12.8
build of PyTorch; to target another CUDA version, rebuild with
`--build-arg USE_LOCK=0` and the `TORCH_VERSION` / `TORCH_INDEX_URL` args
documented at the top of the Dockerfile.

On an HPC cluster with Apptainer/Singularity, export the image to an archive
first — `docker-daemon://` needs a Docker daemon, which such clusters usually
lack:

```bash
podman save --format docker-archive -o atomica_cu128.tar atomica:cu128
singularity build atomica.sif docker-archive://atomica_cu128.tar
singularity exec --nv -B /path/to/checkpoints:/ckpt atomica.sif \
    atomica-representations --help
```

`--nv` exposes the host's NVIDIA driver; without it `torch.cuda.is_available()`
is `False`. `apptainer` substitutes for `singularity` in both commands. Building
the `.sif` runs `mksquashfs` and needs several GB of RAM, so run it in a batch
job rather than on a login node. GPU access from rootless `podman run` depends
on your site having CDI devices registered; use Singularity if it is not set up.

## Verifying the install

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pytest tests/
```

Then run the end-to-end example in
[`tutorials/1_get_embeddings`](../tutorials/1_get_embeddings).

## Tested configurations

Each row loads the pretrained checkpoint, embeds the seven structures in
`data/example/example_inputs.csv`, and compares all 158,848 returned values
against `data/example/example_embeddings.parquet` (NVIDIA H100 80GB).

| CUDA | PyTorch | Python | `torch-scatter` | Max abs. diff vs. reference |
| --- | --- | --- | --- | --- |
| 11.8 | 2.1.1 | 3.11 | installed | 3.1e-6 |
| 12.6 | 2.7.1 | 3.12 | not installed | 2.3e-6 |
| 12.8 | 2.8.0 | 3.12 | not installed | 2.7e-6 |
| 13.0 | 2.9.1 | 3.12 | not installed | 2.0e-6 |
| 13.0 | 2.13.0 | 3.12 | not installed | 2.0e-6 |

Cosine similarity is 1.000000 throughout; the differences are float32
reduction-order noise, and the reference was generated under CUDA 11.8.
Re-running `process_pdbs` reproduced all 4,404 atoms exactly. CPU runs and the
container match to the same tolerance.

## Optional compiled extensions

`torch-scatter` and `torch-cluster` are PyTorch Geometric CUDA extensions.
Neither is required:

- `torch-scatter` — used automatically when installed; otherwise ATOMICA uses
  the equivalent pure-PyTorch code in `atomica/utils/scatter.py`.
- `torch-cluster` — imported lazily, needed only for torsion-denoising
  **pretraining**. Embeddings, finetuning and inference do not use it.

If you do want them, install from the PyG wheel index matching your torch and
CUDA build (PyPI has only a source distribution, which triggers a long compile):

```bash
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
pip install torch-scatter torch-cluster -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
```

## Troubleshooting

**`torch.cuda.is_available()` is `False`.** The wheel's CUDA version is newer
than your driver supports. Check `nvidia-smi` and reinstall torch from an older
`--index-url` per the table above.

**`GLIBC_2.32 not found` when importing `torch_scatter`.** PyG's wheels for
torch >= 2.6 need glibc 2.32, newer than RHEL/Rocky 8 provides. Just remove the
extension — ATOMICA does not need it:
`pip uninstall torch-scatter torch-cluster`.

**CUDA out of memory.** Lower `--batch_size` or set `--atom_budget`.
`atomica-representations` already retries a failing batch one structure at a
time.

**`--model_ckpt` reports a pickled model object.** ATOMICA loads models from a
config JSON plus a weights state dict, which is portable across PyTorch, CUDA
and e3nn versions:

```bash
atomica-representations --model_config <model>_config.json --model_weights <model>_weights.pt ...
```

Training writes `config.json` and a `.pt` next to every `.ckpt`, and
[huggingface.co/ada-f/ATOMICA](https://huggingface.co/ada-f/ATOMICA) publishes
models in this form.
