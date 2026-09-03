"""Embed the structures in data/ with the pretrained ATOMICA model.

Writes the three files the two analysis scripts read into embeddings/. Takes about ten
minutes on one A100 and runs on CPU with --device cpu.

    python compute_embeddings.py --ckpt_dir checkpoints/ATOMICA_checkpoints/pretrain
"""

import argparse
import subprocess
import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent
DATA_DIR = TUTORIAL_DIR / "data"
EMBEDDING_DIR = TUTORIAL_DIR / "embeddings"
DEFAULT_CKPT = (TUTORIAL_DIR.parents[1]
                / "checkpoints/ATOMICA_checkpoints/pretrain")

# (input, output, representation, segment). h_block is one vector per residue or ligand
# fragment; h_interface is the learned attention pooling over one molecule's blocks.
# Segment 1 of an inhibitor graph is the ligand, segment 0 of a patch graph is the patch.
JOBS = [
    ("inhibitors_processed.parquet", "inhibitors_h_block.parquet", "h_block", None),
    ("peptide_partners_processed.parquet", "peptide_partners_h_block.parquet",
     "h_block", None),
    ("surface_patches_processed.parquet", "surface_patches_h_interface.parquet",
     "h_interface", 0),
]

BATCH_SIZE = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT),
                        help="directory holding the pretrained config and weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    ckpt = Path(args.ckpt_dir)
    config, weights = ckpt / "pretrain_model_config.json", ckpt / "pretrain_model_weights.pt"
    for path in (config, weights):
        if not path.exists():
            raise SystemExit(f"missing {path}; see the README for the download command")

    EMBEDDING_DIR.mkdir(exist_ok=True)
    for source, output, representation, segment in JOBS:
        cmd = [sys.executable, "-m", "atomica.representations",
               "--model_config", str(config), "--model_weights", str(weights),
               "--data_path", str(DATA_DIR / source),
               "--output_path", str(EMBEDDING_DIR / output),
               "--representations", representation,
               "--batch_size", str(BATCH_SIZE), "--device", args.device,
               # Reproduces the embeddings the published results were computed from.
               "--no_group_by_max_block"]
        if segment is not None:
            cmd += ["--segment", str(segment)]
        if representation == "h_interface":
            cmd += ["--allow_batched_attention"]
        print(f"\n=== {output} ===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
