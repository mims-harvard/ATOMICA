"""Worked example: pull several ATOMICA representations for the same structures and compare them.

Runs on CPU in about a minute on the seven complexes in `data/example`. From the repository root:

    python tutorials/1_get_embeddings/example_representations.py

It prints, in order:

  1. the representations this checkpoint can produce, and how wide each one is;
  2. the shapes you get back for one complex, so you can see which are per-atom, per-block and
     per-complex;
  3. the other complexes ranked against the first one twice, once by z_graph and once by
     h_graph, to show that the choice of representation is not cosmetic.
"""

import os

import numpy as np
import torch

from atomica import representations as R

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain",
                      "pretrain_model_config.json")
WEIGHTS = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain",
                       "pretrain_model_weights.pt")
DATA = os.path.join(REPO, "data", "example", "example_processed_data.parquet")


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    for path in (CONFIG, WEIGHTS, DATA):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}. The README in this directory has the two "
                             f"commands that produce these: `hf download` for the checkpoint and "
                             f"`python -m atomica.data.process_pdbs` for the structures.")

    # `load_model` reads model_type out of the config and returns the matching dataset class, so
    # the same two lines work for the pretrain checkpoint and for a fine-tuned one.
    model, dataset_class = R.load_model(CONFIG, WEIGHTS)
    model.eval()
    dataset = dataset_class(DATA)

    print("\n== what this checkpoint can produce ==\n")
    print(R.describe(model))

    names = ["h_atom", "h_block", "h_graph", "z_block", "z_graph"]
    rows = R.embed_dataset(
        model, dataset, names,
        pool="mean_component_normalized",   # z_graph is being compared with a cosine, not trained on
        batch_size=1,
        progress=False,
    )

    print("\n== shapes for one complex ==\n")
    first = rows[0]
    print(f"{first['id']}: {len(first['atom_id'])} atoms, {len(first['block_id'])} blocks")
    for name in names:
        print(f"  {name:<8} {np.shape(first[name])}")

    print("\n== the choice of representation changes the answer ==\n")
    by_id = {row["id"]: row for row in rows}
    ids = list(by_id)
    query = ids[0]
    print(f"cosine similarity to {query}, ranked by z_graph:\n")
    print(f"  {'complex':<25} {'z_graph':>9} {'h_graph':>9}")
    ranked = sorted(ids[1:], key=lambda other: -cosine(by_id[query]["z_graph"],
                                                       by_id[other]["z_graph"]))
    for other in ranked:
        z = cosine(by_id[query]["z_graph"], by_id[other]["z_graph"])
        h = cosine(by_id[query]["h_graph"], by_id[other]["h_graph"])
        print(f"  {other:<25} {z:>9.3f} {h:>9.3f}")
    print("\nz_graph here uses mean_component_normalized, the rule for comparing frozen vectors."
          "\nh_graph is the 32-number graph-level readout. They are different quantities, so never"
          "\nmix them in one comparison.")

    print("\nRun `python -m atomica.representations --guidance` for which one to use when.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
