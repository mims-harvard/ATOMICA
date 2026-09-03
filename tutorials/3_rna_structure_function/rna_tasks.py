"""Task definitions and paths for the four RNAglib benchmarks."""

from __future__ import annotations

import os

import pandas as pd

from atomica.data.dataset import LabelledPDBDataset, MultiClassLabelledPDBDataset
from atomica.probe import ProbeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CHECKPOINTS = os.path.join(HERE, "checkpoints")
EMBEDDINGS = os.path.join(HERE, "embeddings")
PREDICTIONS = os.path.join(HERE, "predictions")

SPLITS = ("train", "val", "test")

# residue_level decides the representation: z_block per residue, or z_graph per molecule.
TASKS = {
    "RNA_Protein": dict(prefix="rna_protein", residue_level=True, task_type="binary",
                        primary="auprc", loss="bce", dataset=LabelledPDBDataset,
                        class_names=None,
                        blurb="protein-binding residues, binary, one point per residue"),
    "RNA_Site": dict(prefix="rna_site", residue_level=True, task_type="binary",
                     primary="auprc", loss="bce", dataset=LabelledPDBDataset,
                     class_names=None,
                     blurb="small-molecule-binding residues, binary, one point per residue"),
    "RNA_Ligand": dict(prefix="rna_ligand", residue_level=False, task_type="multiclass",
                       primary="f1_macro", loss="weighted_ce",
                       dataset=MultiClassLabelledPDBDataset, class_names=("PAR", "LLL", "8UZ"),
                       blurb="pocket ligand class, 3-way, one point per pocket"),
    "RNAGo": dict(prefix="rna_go", residue_level=False, task_type="multilabel",
                  primary="f1_macro", loss="focal_bce", dataset=MultiClassLabelledPDBDataset,
                  class_names=("GO:0000353", "GO:0010468", "GO:0005682",
                               "GO:0005688", "GO:0005686"),
                  blurb="five Gene Ontology terms, multilabel, one point per molecule"),
}

# The paper selects between two losses on the graph-level tasks, on validation.
ALTERNATE_LOSS = {"RNA_Ligand": "ce", "RNAGo": "bce"}

BACKBONES = {
    "standard": os.path.join(CHECKPOINTS, "pretrain"),
    "no_protein_rna": os.path.join(CHECKPOINTS, "pretrain_no_protein_rna"),
    "no_nucleic_acid_ligand": os.path.join(CHECKPOINTS, "pretrain_no_nucleic_acid_ligand"),
}

# Three tasks are scored with an encoder that never saw the interaction type the labels are about.
# Any other backbone gives a number the paper does not report.
PUBLISHED_BACKBONE = {
    "RNA_Protein": "no_protein_rna",
    "RNA_Site": "no_nucleic_acid_ligand",
    "RNA_Ligand": "no_nucleic_acid_ligand",
    "RNAGo": "standard",
}

# Structures per forward pass. embed_dataset batches only structures that share a largest block, so
# raising this changes speed and not the vectors.
EXTRACT_BATCH_SIZE = 1

# Fixed before any result was looked at. The loss is selected on validation on the graph tasks only.
PROBE = ProbeConfig(hidden_dim=512, final_hidden_dim=32, dropout=0.3, lr=1e-3,
                    weight_decay=1e-4, epochs=200, patience=20, batch_size=256,
                    seeds=[0, 1, 2, 3, 4], use_batchnorm=True)

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

REPORTED_METRICS = {
    True: ["auprc", "auroc"],
    False: ["f1_macro", "f1_micro", "auprc_macro", "auroc_macro"],
}


# Residue labels keyed by PDB residue index. The per-split task files carry the same labels in
# block order, and the two orders differ for a minority of structures.
RESIDUE_LABELS = os.path.join(HERE, "residue_labels.parquet")


def residue_label_map(task: str, split: str) -> dict:
    """point id -> label for a residue-level task."""
    frame = pd.read_parquet(RESIDUE_LABELS)
    frame = frame[(frame["task"] == TASKS[task]["prefix"]) & (frame["split"] == split)]
    return dict(zip(frame["id"].astype(str), frame["label"].astype(float)))


def data_file(task: str, split: str) -> str:
    return os.path.join(DATA, f"{TASKS[task]['prefix']}_{split}.parquet")


def load_dataset(task: str, split: str):
    path = data_file(task, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}; see the README section on data")
    return TASKS[task]["dataset"](path)


def embedding_dir(task: str, backbone: str) -> str:
    return os.path.join(EMBEDDINGS, backbone, task)


def losses_for(task: str):
    committed = TASKS[task]["loss"]
    other = ALTERNATE_LOSS.get(task)
    return [committed] + ([other] if other and other != committed else [])
