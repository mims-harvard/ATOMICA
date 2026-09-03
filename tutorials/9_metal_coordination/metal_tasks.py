"""Paths, label vocabulary and tasks for the metal-coordination probes."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from atomica.probe import LinearProbeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LABELS = os.path.join(HERE, "data", "metal_coordination_labels.parquet")
POCKETS = os.path.join(HERE, "data", "metal_pockets.parquet")
EMBEDDINGS = os.path.join(HERE, "embeddings")
PREDICTIONS = os.path.join(HERE, "predictions")

SPLITS = ("train", "valid", "test")

#: The released pretrained ATOMICA checkpoint. See checkpoints/README.md.
CHECKPOINT = os.path.join(REPO, "checkpoints", "ATOMICA_checkpoints", "pretrain")

#: Extraction batch size. Safe to change: batches are grouped so that every structure in one shares
#: a largest block, so batch size does not affect the vectors.
EXTRACT_BATCH_SIZE = 16


# ------------------------------------------------------------------------------ label vocabulary
#: FindGeo labels a site whose donors trace a polyhedron with one vertex unoccupied as "with a
#: vacancy". That qualifier is kept as part of the class, since folding it away would pool three
#: donors in a tetrahedron-minus-a-vertex with four donors in a full tetrahedron. Rarer polyhedra
#: fold into `other` rather than being dropped, so the class prior stays the real one.
VACANCY_SUFFIX = " [vacancy]"
GEOMETRY_CLASSES: List[str] = [
    "tetrahedron",
    "octahedron",
    "pentagonal bipyramid",
    "square pyramid" + VACANCY_SUFFIX,
    "square pyramid",
    "trigonal bipyramid" + VACANCY_SUFFIX,
    "pentagonal bipyramid" + VACANCY_SUFFIX,
    "trigonal plane" + VACANCY_SUFFIX,
    "tetrahedron" + VACANCY_SUFFIX,
    "square plane" + VACANCY_SUFFIX,
    "trigonal bipyramid",
    "square plane",
    "trigonal plane",
    "other",
]

#: Sites FindGeo left unassigned. Dropped for geometry, kept for coordination number, which is why
#: the two targets have different N.
UNASSIGNED_GEOMETRY = "unassigned"

CN_CLASSES = [1, 2, 3, 4, 5, 6, 7]
CN_TAIL = "8+"

#: Only metal sites matched uniquely to a MetalPDB record are used. See data/README.md.
MATCHED = "matched"


def collapse_geometry(base, vacancy=None) -> str:
    """FindGeo polyhedron plus its vacancy qualifier -> one of GEOMETRY_CLASSES."""
    if base is None or (isinstance(base, float) and np.isnan(base)):
        return UNASSIGNED_GEOMETRY
    has_vacancy = vacancy is not None and not (isinstance(vacancy, float) and np.isnan(vacancy))
    label = str(base) + (VACANCY_SUFFIX if has_vacancy else "")
    return label if label in GEOMETRY_CLASSES else "other"


def collapse_cn(cn) -> Optional[str]:
    """Coordination number -> class string, or None where MetalPDB gives no number."""
    if cn is None or (isinstance(cn, float) and np.isnan(cn)):
        return None
    cn = int(cn)
    return str(cn) if cn in CN_CLASSES else (CN_TAIL if cn >= 8 else "other")


def derive_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add geometry_class, cn_full_class and cn_protein_class from the raw MetalPDB fields."""
    df = df.copy()
    df["geometry_class"] = [collapse_geometry(b, v) for b, v
                            in zip(df["geometry_base"], df["geometry_vacancy"])]
    df["cn_full_class"] = df["cn_full"].map(collapse_cn)
    df["cn_protein_class"] = df["cn_protein"].map(collapse_cn)
    return df


# ------------------------------------------------------------------------------------ the split
def probe_split() -> Dict[str, pd.DataFrame]:
    """The three probe splits, grouped by PDB entry so no entry appears in two of them.

    Test keeps all its labelled sites; validation and then train keep only entries not already
    used. Gives 20,159 / 2,421 / 3,654 sites.
    """
    table = pd.read_parquet(LABELS)
    table = derive_label_columns(table[table["status"] == MATCHED].copy())

    raw = {s: table[table["split"] == s] for s in SPLITS}
    test_entries = set(raw["test"]["pdb_code"])
    valid = raw["valid"][~raw["valid"]["pdb_code"].isin(test_entries)].copy()
    train = raw["train"][~raw["train"]["pdb_code"].isin(
        test_entries | set(valid["pdb_code"]))].copy()
    out = {"train": train, "valid": valid, "test": raw["test"].copy()}

    assert not (set(train["pdb_code"]) & test_entries), "probe train and test share a PDB entry"
    assert not (set(valid["pdb_code"]) & test_entries), "probe valid and test share a PDB entry"
    assert not (set(train["pdb_code"]) & set(valid["pdb_code"])), "train and valid share an entry"
    for name, df in out.items():
        df["probe_split"] = name
    return out


# ----------------------------------------------------------------------------------- the tasks
#: Two coordination numbers are reported because ATOMICA's pockets hold amino acids only, no waters
#: and no cofactors, while MetalPDB counts the complete deposited first sphere. Half the test sites
#: have at least one donor the model cannot see, so `cn_full` needs information the input withholds
#: and `cn_protein` does not.
TASKS: Dict[str, Dict] = {
    "cn_full": dict(label="cn_full_class", drop_unassigned=False, fixed_cn=None,
                    blurb="coordination number over all deposited donors"),
    "cn_protein": dict(label="cn_protein_class", drop_unassigned=False, fixed_cn=None,
                       blurb="coordination number over visible protein donors"),
    "geometry": dict(label="geometry_class", drop_unassigned=True, fixed_cn=None,
                     blurb="FindGeo coordination geometry"),
    # Every site here has the same deposited coordination number, so nothing can be scored by
    # counting donors and only the angular arrangement separates the classes.
    "geometry_cn6": dict(label="geometry_class", drop_unassigned=True, fixed_cn=6,
                         blurb="FindGeo geometry at deposited coordination number six"),
}

#: Inside a fixed-coordination-number task a class needs this many sites in train and in test;
#: balanced accuracy weights classes equally, so a two-member class would move it by 1/k.
MIN_CLASS_FIXED_CN = 10
MIN_TEST_SITES = 50


def task_rows(task: str, df: pd.DataFrame) -> pd.DataFrame:
    """The rows one task is fitted and scored on."""
    spec = TASKS[task]
    out = df[df[spec["label"]].notna()]
    if spec["drop_unassigned"]:
        out = out[out[spec["label"]] != UNASSIGNED_GEOMETRY]
    if spec["fixed_cn"] is not None:
        out = out[out["cn_full"] == spec["fixed_cn"]]
    return out.copy()


def classes_to_keep(task: str, train: pd.DataFrame, test: pd.DataFrame) -> set:
    """Classes the probe may predict. A class absent from train can never be predicted."""
    label = TASKS[task]["label"]
    floor = MIN_CLASS_FIXED_CN if TASKS[task]["fixed_cn"] is not None else 1
    counts = train[label].value_counts()
    keep = set(counts[counts >= floor].index)
    if TASKS[task]["fixed_cn"] is not None:
        test_counts = test[label].value_counts()
        keep &= set(test_counts[test_counts >= MIN_CLASS_FIXED_CN].index)
    return keep


# ---------------------------------------------------------------------------------- the protocol
#: L2 multinomial logistic regression, no hidden layer. C is chosen on validation balanced
#: accuracy; the test split is read once, at the end, and selects nothing.
LINEAR = LinearProbeConfig(C_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0), max_iter=2000, tol=1e-3,
                           standardize=True, seed=0, primary="balanced_acc")

#: Balanced accuracy leads because the classes are strongly imbalanced.
METRICS = ("balanced_acc", "accuracy", "f1_macro")
PRIMARY_METRIC = "balanced_acc"

#: Intervals resample PDB entries, not sites: several metal sites come from one structure and are
#: not independent observations.
BOOTSTRAP_CLUSTER = "pdb_code"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


def chance_level(n_classes: int) -> float:
    """Balanced accuracy of a constant prediction."""
    return 1.0 / n_classes
