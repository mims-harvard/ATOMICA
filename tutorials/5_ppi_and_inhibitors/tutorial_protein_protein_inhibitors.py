"""Retrieve surface patches on a PPI partner using a small-molecule inhibitor as the query.

For a protein-protein complex A-B, 1,000 patches sampled on the surface of partner B are
ranked by ATOMICA embedding distance to an inhibitor that binds A. Fold Change@10 asks
whether the ten closest patches lie near the native A-B binding site. The query and the
candidates come from different structures and different modalities, and A and B are never
embedded together; protein A's coordinates are used only after ranking, to label how far
each patch is from the interface.

    python tutorial_protein_protein_inhibitors.py
"""

import os

import numpy as np
import pandas as pd
import scipy.spatial.distance
import scipy.stats

from common import (K, by_superfamily, fold_change_at_k, load_metadata,
                    plot_by_superfamily, plot_top_bottom, precision_at_k, report)

TUTORIAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TUTORIAL_DIR, "data")
EMBEDDING_DIR = os.path.join(TUTORIAL_DIR, "embeddings")
FIG_DIR = os.path.join(TUTORIAL_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# A patch is close to the native interface at 12 A from its centre to the nearest target
# CA atom, the 25th percentile of the sampled distances.
GEOM_THRESH = 12.0

# ATOMICA prepends one global block node per segment, with block type 3 and one atom.
GLOBAL_BLOCK_ID = 3

# Complexes without a binary A-B interface: an AABB assembly and a self-binding trimer.
NON_BINARY = ("INTEGRASE/LEDGF", "TNFA/TNFA")

# Featured example, chosen by identity rather than by rank: an HRAS/SOS1 inhibitor.
EXAMPLE = ("HRAS/SOS1", "6ZL3")


def read_embedding(structure_file, embedding_file, column):
    """Read an embedding column and check it is row-aligned with its structure file."""
    ids = pd.read_parquet(os.path.join(DATA_DIR, structure_file), columns=["id"])["id"]
    emb = pd.read_parquet(os.path.join(EMBEDDING_DIR, embedding_file))
    if not (ids.values == emb["id"].values).all():
        raise ValueError(f"{structure_file} and {embedding_file} are not row-aligned")
    return emb[column]


def load_queries(metadata):
    """One query vector per protein-inhibitor structure.

    The query is the unweighted mean of the inhibitor's block embeddings over segment 1,
    the ligand, excluding the global block node. Those blocks are still contextualized by
    the target pocket, because message passing runs over the whole pocket-plus-ligand
    graph before any block is read out, but no pocket block enters the query.
    """
    structure = pd.read_parquet(os.path.join(DATA_DIR, "inhibitors_processed.parquet"),
                                columns=["id", "B", "segment_ids"])
    h_block = read_embedding("inhibitors_processed.parquet",
                             "inhibitors_h_block.parquet", "h_block")

    queries = pd.DataFrame({"id": structure["id"].values})
    entry = queries["id"].str.split("_").str[0]
    queries["pdb_code"] = entry.map(metadata["pdb_code"])
    queries["family"] = entry.map(metadata["family"])
    queries["vector"] = [
        np.stack(v)[(np.asarray(b) != GLOBAL_BLOCK_ID) & (np.asarray(s) == 1)].mean(axis=0)
        for v, b, s in zip(h_block, structure["B"], structure["segment_ids"])]
    print(f"Loaded {len(queries)} inhibitor queries")
    return queries


def load_candidates(metadata):
    """One vector per sampled patch, plus its distance to the target chain.

    A patch graph is a single segment, so its interface vector pools every block it has.
    """
    structure = pd.read_parquet(
        os.path.join(DATA_DIR, "surface_patches_processed.parquet"),
        columns=["id", "distance_to_target"])
    pdb_to_family = dict(zip(metadata["ppi_pdb"], metadata["family"]))

    candidates = pd.DataFrame({"id": structure["id"].values})
    candidates["family"] = candidates["id"].str.split("_").str[0].map(pdb_to_family)
    candidates["distance"] = structure["distance_to_target"].values.astype(float)
    candidates["vector"] = list(np.stack(read_embedding(
        "surface_patches_processed.parquet", "surface_patches_h_interface.parquet",
        "h_interface")))
    print(f"Loaded {len(candidates)} surface patches")
    return candidates


def restrict_to_binary(queries, candidates):
    shared = set(queries["family"]) & set(candidates["family"])
    kept = queries[queries["family"].isin(shared - set(NON_BINARY))].copy()
    print(f"\nKept {kept['family'].nunique()} protein-protein complexes "
          f"({len(kept)} matched inhibitor structures)")
    print(kept["family"].value_counts().sort_index().to_string())
    return kept


def run_retrieval(queries, candidates):
    """Rank each complex's patches from each of its inhibitor queries."""
    per_family = dict(tuple(candidates.groupby("family")))
    scored = queries.copy()

    def rank(row):
        group = per_family[row["family"]]
        emb = scipy.spatial.distance.cdist(
            np.asarray(row["vector"]).reshape(1, -1),
            np.stack(group["vector"].values), metric="cosine").ravel()
        return emb, group["distance"].values, group["id"].tolist()

    scored[["emb_dist", "distance", "patch_ids"]] = scored.apply(
        lambda r: pd.Series(rank(r)), axis=1)
    scored["fold_change"] = scored.apply(
        lambda r: fold_change_at_k(r["emb_dist"], r["distance"], GEOM_THRESH), axis=1)
    scored["precision"] = scored.apply(
        lambda r: precision_at_k(r["emb_dist"], r["distance"], GEOM_THRESH), axis=1)
    return scored


def spearman_by_complex(scored, sfmap):
    """Per-complex correlation between embedding distance and distance to the interface.

    Every inhibitor of one complex is ranked against the same patches, so the pooled pairs
    are not independent observations. Reported for sign and magnitude only, with no test.
    """
    rows = []
    for family, group in scored.groupby("family"):
        emb = np.concatenate(group["emb_dist"].values)
        geom = np.concatenate(group["distance"].values)
        rows.append({"complex": family,
                     "spearman_rho": scipy.stats.spearmanr(emb, geom).statistic,
                     "n_structures": len(group)})
    table = pd.DataFrame(rows).sort_values("complex").reset_index(drop=True)
    print("\nSpearman(embedding distance, distance to interface) by complex:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    per_sf = by_superfamily(table.set_index("complex")["spearman_rho"], sfmap)
    print(f"  positive in {int((table['spearman_rho'] > 0).sum())}/{len(table)} complexes "
          f"and {int((per_sf > 0).sum())}/{len(per_sf)} superfamilies")
    return table


def plot_example(scored):
    family, pdb = EXAMPLE
    subset = scored[(scored["family"] == family) & (scored["pdb_code"] == pdb)]
    if subset.empty:
        available = sorted(scored.loc[scored["family"] == family, "pdb_code"].unique())
        raise ValueError(f"no inhibitor {pdb} in {family}; available: {available}")
    entry = subset.iloc[0]
    print(f"\nExample: {family} inhibitor {pdb}, {len(entry['emb_dist'])} candidate "
          f"patches, Fold Change@{K} = {entry['fold_change']:.2f}")

    rng = np.random.default_rng(0)
    order = np.argsort(entry["emb_dist"])
    reference = rng.integers(0, len(order), size=100)
    index = np.concatenate([order[:K], order[-K:], reference])
    print(f"  top {K} patches: {[entry['patch_ids'][i] for i in order[:K]]}")
    plot_top_bottom(
        entry["distance"][index], "Distance to native interface (A)", (0, 50),
        f"{family} {pdb}",
        os.path.join(FIG_DIR, f"{family.replace('/', '.')}_{pdb}_patches.svg"),
        reference)


def main():
    metadata, sfmap = load_metadata(DATA_DIR)
    queries = load_queries(metadata)
    candidates = load_candidates(metadata)
    queries = restrict_to_binary(queries, candidates)

    base = candidates.groupby("family")["distance"].apply(
        lambda d: float(np.mean(d <= GEOM_THRESH)))
    print(f"\nPatches within {GEOM_THRESH:.0f} A of the target chain: "
          f"{base.min():.3f} to {base.max():.3f} across complexes")

    scored = run_retrieval(queries, candidates)
    fold_change = scored.groupby("family")["fold_change"].mean()
    precision = scored.groupby("family")["precision"].mean()
    report(fold_change, sfmap, f"Fold Change@{K}", reference=1.0)
    report(precision, sfmap, f"Precision@{K}")
    spearman_by_complex(scored, sfmap)

    plot_by_superfamily(fold_change, sfmap, f"Fold Change@{K}",
                        os.path.join(FIG_DIR, "protein_fold_change.svg"), reference=1.0)
    plot_by_superfamily(precision, sfmap, f"Precision@{K}",
                        os.path.join(FIG_DIR, "protein_precision.svg"))
    plot_example(scored)


if __name__ == "__main__":
    main()
