"""Compare inhibitor blocks with peptide blocks across a protein-peptide interface.

For each protein-peptide complex and each small-molecule inhibitor of the same target, the
inhibitor's ligand blocks are ranked against the peptide's blocks by ATOMICA embedding
distance. Fold Change@10 asks whether the ten closest pairs in embedding space are also
close in space, once the two crystal structures are superposed on their shared target
chain.

    python tutorial_protein_peptide_inhibitors.py
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

# A block pair is spatially close at 4 A between superposed block centres.
GEOM_THRESH = 4.0

# A match is kept when the superposed inhibitor lands within 2 A of the peptide and leaves
# more than 10 block pairs after single-atom blocks are dropped.
MIN_DIST = 2.0
MIN_PAIRS = 10

# Featured example, chosen by identity rather than by rank: the MENIN/MLL inhibitor MIV-7.
EXAMPLE = ("MENIN/MLL", "4OG7", "2SE")


def block_mask(processed, segment=None):
    """Blocks with more than one atom, optionally restricted to one segment.

    Single-atom blocks have no orientation, and the global block node ATOMICA prepends to
    each segment is also one atom, so this drops both.
    """
    keep = np.asarray(processed["block_lengths"]) > 1
    if segment is not None:
        keep = keep & (np.asarray(processed["segment_ids"]) == segment)
    return keep


def embedding_distances(geometry):
    """Cosine distance between inhibitor and peptide block embeddings, per matched pair."""
    # Indexed by row, not by id: inhibitors_processed.parquet is exploded over ligand
    # residue numbers, so its ids are not unique.
    inhibitors = pd.read_parquet(os.path.join(DATA_DIR, "inhibitors_processed.parquet"),
                                 columns=["block_lengths", "segment_ids"])
    inhibitors["h_block"] = pd.read_parquet(
        os.path.join(EMBEDDING_DIR, "inhibitors_h_block.parquet"))["h_block"].values

    peptides = pd.read_parquet(os.path.join(DATA_DIR, "peptide_partners_processed.parquet"),
                               columns=["id", "block_lengths"])
    peptides["h_block"] = pd.read_parquet(
        os.path.join(EMBEDDING_DIR, "peptide_partners_h_block.parquet"))["h_block"].values
    peptides = peptides.set_index("id")

    out = []
    for row in geometry.itertuples():
        inhibitor = inhibitors.iloc[row.inhibitor_row]
        peptide = peptides.loc[row.peptide_id]
        ligand = np.stack(inhibitor["h_block"])[block_mask(inhibitor, segment=1)]
        partner = np.stack(peptide["h_block"])[block_mask(peptide)]
        distances = scipy.spatial.distance.cdist(ligand, partner, metric="cosine").ravel()
        if distances.shape != np.asarray(row.block_coords_dist).shape:
            raise ValueError(f"row {row.inhibitor_row}: embedding and geometry disagree")
        out.append(distances)
    return out


def load_matches(metadata):
    """Matched pairs that meet the inclusion cut-offs, with both distance matrices."""
    geometry = pd.read_parquet(os.path.join(DATA_DIR, "peptide_inhibitor_geometry.parquet"))
    ids = pd.read_parquet(os.path.join(DATA_DIR, "inhibitors_processed.parquet"),
                          columns=["id"])["id"]
    entry = ids.iloc[geometry["inhibitor_row"]].reset_index(drop=True)
    geometry["lig_code"] = entry.str.split("_").str[-1]
    entry = entry.str.split("_").str[0]
    geometry["family"] = entry.map(metadata["family"]).values
    geometry["inhibitor_pdb_id"] = entry.map(metadata["pdb_code"]).values
    geometry["min_dist"] = [float(np.min(d)) for d in geometry["block_coords_dist"]]
    geometry["n_pairs"] = [int(np.size(d)) for d in geometry["block_coords_dist"]]
    print(f"Scored {len(geometry)} inhibitor-complex matches over "
          f"{geometry['family'].nunique()} protein-peptide complexes")

    kept = geometry[(geometry["min_dist"] < MIN_DIST)
                    & (geometry["n_pairs"] > MIN_PAIRS)].reset_index(drop=True)
    print(f"Kept {len(kept)} matched protein-inhibitor structures over "
          f"{kept['family'].nunique()} complexes")
    print(f"  block pairs per match: median {int(kept['n_pairs'].median())}, "
          f"IQR {int(kept['n_pairs'].quantile(0.25))}-{int(kept['n_pairs'].quantile(0.75))}")
    print(f"  target-chain superposition RMSD: median {kept['align_rmsd'].median():.2f} A")

    kept["block_emb_dist"] = embedding_distances(kept)
    kept["fold_change"] = [fold_change_at_k(e, g, GEOM_THRESH) for e, g
                           in zip(kept["block_emb_dist"], kept["block_coords_dist"])]
    kept["precision"] = [precision_at_k(e, g, GEOM_THRESH) for e, g
                         in zip(kept["block_emb_dist"], kept["block_coords_dist"])]
    return kept


def spearman_by_complex(matches, sfmap):
    """Per-complex correlation between embedding distance and spatial distance.

    Block pairs within one complex share blocks with each other, so they are not
    independent observations. Reported for sign and magnitude only, with no test.
    """
    rows = []
    for family, group in matches.groupby("family"):
        emb = np.concatenate(list(group["block_emb_dist"]))
        geom = np.concatenate(list(group["block_coords_dist"]))
        rows.append({"complex": family,
                     "spearman_rho": scipy.stats.spearmanr(emb, geom).statistic,
                     "n_structures": len(group), "n_block_pairs": emb.size})
    table = pd.DataFrame(rows).sort_values("complex").reset_index(drop=True)
    print("\nSpearman(embedding distance, spatial distance) by complex:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    per_sf = by_superfamily(table.set_index("complex")["spearman_rho"], sfmap)
    print(f"  positive in {int((table['spearman_rho'] > 0).sum())}/{len(table)} complexes "
          f"and {int((per_sf > 0).sum())}/{len(per_sf)} superfamilies")
    return table


def plot_example(matches):
    family, pdb, ligand = EXAMPLE
    subset = matches[(matches["family"] == family)
                     & (matches["inhibitor_pdb_id"] == pdb)
                     & (matches["lig_code"] == ligand)]
    if subset.empty:
        raise ValueError(f"no match for {family} {pdb} {ligand} among those kept")
    entry = subset.iloc[0]
    print(f"\nExample: {family} inhibitor {pdb} ligand {ligand} against complex "
          f"{entry['peptide_id']}, {entry['n_pairs']} block pairs, "
          f"Fold Change@{K} = {entry['fold_change']:.2f}")

    order = np.argsort(entry["block_emb_dist"])
    reference = np.arange(len(order))
    index = np.concatenate([order[:K], order[-K:], reference])
    plot_top_bottom(
        np.asarray(entry["block_coords_dist"])[index], "Distance between blocks (A)",
        (0, 25),
        f"{family} {pdb} {ligand} vs {entry['peptide_id']}\n"
        f"(n = {entry['n_pairs']} pairs)",
        os.path.join(FIG_DIR, f"{family.replace('/', '.')}_{pdb}_{ligand}_blocks.svg"),
        reference, seed=42)


def main():
    metadata, sfmap = load_metadata(DATA_DIR)
    matches = load_matches(metadata)

    fold_change = matches.groupby("family")["fold_change"].mean()
    precision = matches.groupby("family")["precision"].mean()
    report(fold_change, sfmap, f"Fold Change@{K}", reference=1.0)
    report(precision, sfmap, f"Precision@{K}")
    spearman_by_complex(matches, sfmap)

    plot_by_superfamily(fold_change, sfmap, f"Fold Change@{K}",
                        os.path.join(FIG_DIR, "peptide_fold_change.svg"), reference=1.0)
    plot_by_superfamily(precision, sfmap, f"Precision@{K}",
                        os.path.join(FIG_DIR, "peptide_precision.svg"))
    plot_example(matches)


if __name__ == "__main__":
    main()
