"""
Tutorial: Cross-modality interface comparison for protein-protein PPI inhibitors

Compares ATOMICA embeddings of orthosteric small-molecule inhibitors to
ATOMICA embeddings of surface patches sampled on the partner protein (B)
in the native A-B complex they inhibit. The goal is to test whether
embedding similarity between the inhibitor (bound to A) and a patch on
B is highest for patches that sit close to the native A-B binding site
on B.

Pipeline:
  1. Load 2P2IDB metadata and ATOMICA inhibitor embeddings. The
     inhibitor graph embedding is the mean of its non-pocket block
     embeddings (segment_ids == 1).
  2. Load 1,000 sampled interface-patch embeddings per protein B and
     the precomputed distance from each patch to the nearest Cα on
     protein A
     (``protein_partner_surface_patches_distances.csv``).
  3. For each inhibitor, rank all patches in its family by cosine
     distance to the inhibitor embedding, and compute Precision@10 and
     Enrichment@10 using a 12 Å spatial threshold (the ~25th percentile
     of sampled patch-to-A distances).
  4. Run per-family Spearman correlations with FDR correction and a
     binomial test.
  5. Save figures to ``figures/``: enrichment stripplot, precision
     stripplot, and the HRAS/SOS1 inhibitor (PDB 6ZL3, ligand EZZ)
     violin+strip plot of patch-to-interface distances for top-10,
     bottom-10, and reference patches.

Usage:
  python tutorial_protein_protein_inhibitors.py
"""

import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from statsmodels.stats.multitest import multipletests

TUTORIAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TUTORIAL_DIR, "data")
FIG_DIR = os.path.join(TUTORIAL_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

K = 10
GEOM_THRESH = 12.0

# Featured swarm-plot example: the HRAS/SOS1 inhibitor (PDB 6ZL3,
# ligand EZZ). Selected by system identity, not by rank.
SWARM_FAMILY = "HRAS/SOS1"
SWARM_PDB_CODE = "6ZL3"  # PDBProtLig for the HRAS-SOS1 small-molecule inhibitor


def load_metadata():
    df1 = pd.read_csv(os.path.join(DATA_DIR, "2p2idb.csv"), sep=";")
    df2 = pd.read_csv(os.path.join(DATA_DIR, "ppi_inhibitor_mapping.csv"))
    df = df2.merge(df1[["PDBProtProt", "PDBProtLig", "InChI"]],
                   on=["PDBProtProt", "PDBProtLig"], how="left")
    df = df[df["PDBProtProt"] != "na"]
    return df


def load_inhibitor_embeddings(df_2p2idb):
    inh_emb = pd.read_parquet(os.path.join(DATA_DIR, "inhibitors_embeddings.parquet"))
    inh_in = pd.read_parquet(os.path.join(DATA_DIR, "inhibitors_processed.parquet"))
    inh_emb = pd.concat([inh_in, inh_emb.drop(columns=["id"])], axis=1)
    print("Inhibitor embeddings loaded:", inh_emb.shape[0])

    id_to_pdblig = df_2p2idb.set_index("2P2IDB_ID")["PDBProtLig"].to_dict()
    id_to_family = df_2p2idb.set_index("2P2IDB_ID")["Family"].to_dict()

    def inhibitor_graph_embedding(row):
        mask = np.array(row["segment_ids"]).astype(bool)
        return np.mean(np.stack(row["block_embedding"])[mask, :], axis=0)

    inh_emb["pdb_code"] = inh_emb["id"].str.split("_").str[0].map(id_to_pdblig)
    inh_emb["family"] = inh_emb["id"].str.split("_").str[0].map(id_to_family)
    inh_emb["inhibitor_embedding"] = inh_emb.apply(inhibitor_graph_embedding, axis=1)
    return inh_emb


def load_ppi_patches(df_2p2idb):
    ppis = df_2p2idb[["PDBProtProt", "Chain_Target", "Chain_Partner", "Family"]].drop_duplicates()
    ppi_to_family = ppis.set_index("PDBProtProt")["Family"].to_dict()

    ppi_points = pd.read_parquet(os.path.join(DATA_DIR, "protein_partner_surface_patches.parquet"))
    ppi_points["center"] = ppi_points["X"].apply(lambda x: x[0])
    dist_df = pd.read_csv(os.path.join(DATA_DIR, "protein_partner_surface_patches_distances.csv"))
    dist_df["family"] = dist_df["id"].str.split("_").str[0].map(ppi_to_family)

    ppi_emb = pd.read_parquet(os.path.join(DATA_DIR, "protein_partner_surface_patches_embeddings.parquet"))
    ppi_emb["family"] = ppi_emb["id"].str.split("_").str[0].map(ppi_to_family)
    ppi_emb["pdb_code"] = ppi_emb["id"].str.split("_").str[0]
    return ppi_points, dist_df, ppi_emb


def precision_at_k(E, G, k=K, geom_thresh=GEOM_THRESH):
    k = min(k, max(10, int(E.size * 0.1)))
    E = np.asarray(E).reshape(-1)
    G = np.asarray(G).reshape(-1)
    if E.size <= k:
        k = 1
    topk = np.argsort(E)[:k]
    return float(np.mean(G[topk] <= geom_thresh))


def enrichment_at_k(E, G, k=K, geom_thresh=GEOM_THRESH):
    E = np.asarray(E).reshape(-1)
    G = np.asarray(G).reshape(-1)
    base_rate = float(np.mean(G <= geom_thresh))
    if base_rate == 0.0:
        return float("nan")
    return precision_at_k(E, G, k=k, geom_thresh=geom_thresh) / base_rate


def run_retrieval(inh_emb, ppi_emb, dist_df):
    families = list(
        set(inh_emb["family"].unique()).intersection(set(ppi_emb["family"].unique()))
    )
    print(f"Shared families between inhibitors and protein-protein PPIs: {len(families)}")

    # A-B binary interactions only
    inh_emb = inh_emb[inh_emb["family"].isin(families)].copy()
    inh_emb = inh_emb[inh_emb["family"] != "INTEGRASE/LEDGF"]  # AABB complex
    inh_emb = inh_emb[inh_emb["family"] != "TNFA/TNFA"]        # trimer + self-interaction

    def run_one(row):
        df_ppi = ppi_emb[ppi_emb["family"] == row["family"]]
        df_dist = dist_df[dist_df["family"] == row["family"]]
        assert (df_dist["id"].values == df_ppi["id"].values).all(), \
            f"Mismatched PPI/distance IDs for family {row['family']}"
        dist_to_target = df_dist["distance"].values.astype(float)
        ppi_embs = np.stack(df_ppi["graph_embedding"])
        ppi_ids = df_ppi["id"].tolist()
        q = row["inhibitor_embedding"].reshape(1, -1)
        emb_dist = scipy.spatial.distance.cdist(q, ppi_embs, metric="cosine").flatten()
        return emb_dist, dist_to_target, ppi_ids

    result = inh_emb.apply(lambda r: pd.Series(run_one(r)), axis=1)
    inh_emb[["emb_dist", "dist_to_target", "ppi_patch_ids"]] = result

    print("Total inhibitor embeddings retained:", inh_emb.shape[0])
    print("Per-family counts:\n", inh_emb["family"].value_counts())

    q25 = inh_emb["dist_to_target"].apply(lambda x: np.percentile(x, 25))
    print("25th percentile of patch-to-target distances (describe):")
    print(q25.describe())

    inh_emb["enrichment_at_k"] = inh_emb.apply(
        lambda r: enrichment_at_k(r["emb_dist"], r["dist_to_target"]), axis=1
    )
    inh_emb["precision_at_k"] = inh_emb.apply(
        lambda r: precision_at_k(r["emb_dist"], r["dist_to_target"]), axis=1
    )
    print("Inhibitors with enrichment > 1:",
          int(inh_emb["enrichment_at_k"].gt(1.0).sum()),
          "out of", inh_emb.shape[0])

    return inh_emb


def report_stats(inh_emb):
    agg = inh_emb.groupby("family").agg({
        "emb_dist": lambda x: np.concatenate(x.values),
        "dist_to_target": lambda x: np.concatenate(x.values),
    }).reset_index()
    agg[["spearmanr", "p_value"]] = agg.apply(
        lambda r: pd.Series(scipy.stats.spearmanr(r["emb_dist"], r["dist_to_target"])),
        axis=1
    )
    n_sig_pos = int(((agg["p_value"] <= 0.05) & (agg["spearmanr"] > 0)).sum())
    binom = scipy.stats.binomtest(n_sig_pos, len(agg), p=0.05, alternative="greater")
    print(f"Families with positive Spearman & p<=0.05: {n_sig_pos}/{len(agg)}, "
          f"binomial p={binom.pvalue:.3e}")
    print(f"Median Spearman r: {agg['spearmanr'].median():.3f}")

    reject, qvals, _, _ = multipletests(agg["p_value"].values, method="fdr_bh")
    agg["q_value"] = qvals
    agg["significant_fdr_0.05"] = reject
    print("Families FDR-BH significant with positive Spearman:",
          int((agg["significant_fdr_0.05"] & (agg["spearmanr"] > 0)).sum()),
          "out of", len(agg))
    print(agg[["family", "spearmanr", "p_value", "q_value", "significant_fdr_0.05"]])


def plot_stripplot(df, value_col, ylabel, out_name, yticks=None):
    plt.figure(figsize=(6, 2.5))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gray_to_aqua", ["#B0B0B0", "#00C4C7"]
    )
    order = sorted(df["family"].unique())
    ax = sns.stripplot(
        x="family", y=value_col, data=df, size=4, alpha=0.8, jitter=0.25,
        palette=cmap, hue=value_col, order=order, legend=False,
    )
    new_labels = [t.get_text().replace("/", "\n") for t in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels)
    plt.xticks(rotation=90)
    if value_col == "enrichment_at_k":
        plt.axhline(1.0, linestyle="--", color="black", linewidth=0.5)
    plt.ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
    plt.xlabel("")
    for i in range(df["family"].nunique() - 1):
        plt.axvline(i + 0.5, color="lightgray", linewidth=0.5, zorder=0)
    sns.despine()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def pick_inhibitor_by_name(inh_emb, family, pdb_code):
    """Select a single inhibitor row by family and PDBProtLig code."""
    sub = inh_emb[(inh_emb["family"] == family) & (inh_emb["pdb_code"] == pdb_code)]
    if len(sub) == 0:
        raise ValueError(
            f"No inhibitor for family={family!r} pdb_code={pdb_code!r}. "
            f"Available in family: "
            f"{inh_emb[inh_emb['family']==family]['pdb_code'].tolist()}"
        )
    return sub.iloc[0]


def plot_swarm_by_name(inh_emb, ppi_points, family, pdb_code, seed=0):
    rng = np.random.default_rng(seed)
    entry = pick_inhibitor_by_name(inh_emb, family, pdb_code)
    print(f"Swarm-plot example: family {entry['family']} id {entry['id']} "
          f"pdb_code {entry['pdb_code']}, n_patches={len(entry['emb_dist'])}")

    closest = np.argsort(entry["emb_dist"])[:K]
    furthest = np.argsort(entry["emb_dist"])[-K:]
    reference = rng.integers(0, len(entry["emb_dist"]), size=100)
    idx = np.concatenate([closest, furthest, reference])
    types = np.array([f"Top {K}"] * len(closest) +
                     [f"Bottom {K}"] * len(furthest) +
                     ["Reference"] * len(reference))
    plot_data = pd.DataFrame({
        "Distance to Target Interface": entry["dist_to_target"][idx],
        "type": types,
    })

    topk_ids = [entry["ppi_patch_ids"][i] for i in closest]
    bottomk_ids = [entry["ppi_patch_ids"][i] for i in furthest]
    print(f"Top {K} patch IDs: {topk_ids}")
    print("Top-3 centers:",
          ppi_points.set_index("id").loc[topk_ids[:3], "center"].values)
    print(f"Bottom {K} patch IDs: {bottomk_ids}")
    print("Bottom-3 centers:",
          ppi_points.set_index("id").loc[bottomk_ids[:3], "center"].values)

    plt.figure(figsize=(3, 3))
    ax = sns.violinplot(x="type", y="Distance to Target Interface", data=plot_data,
                        color="lightgray", alpha=0.4, inner=None)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gray_to_purple", ["#D86ECC", "#6805F2"]
    )
    xticks = ax.get_xticks()
    categories = [t.get_text() for t in ax.get_xticklabels()]
    pos_dict = dict(zip(categories, xticks))
    x_vals = plot_data["type"].map(pos_dict).values
    jitter = rng.uniform(-0.2, 0.2, size=len(x_vals))
    ax.scatter(x_vals + jitter, plot_data["Distance to Target Interface"],
               c=plot_data["Distance to Target Interface"], cmap=cmap,
               s=10, edgecolor="black", linewidth=0.2)
    sns.despine()
    plt.ylabel("Distance to Target Interface (Å)")
    plt.ylim(0, 50)
    plt.xlabel("")
    plt.title(f"{entry['family']} {entry['id']} {entry['pdb_code']}", fontsize=10)
    plt.tight_layout()
    safe_family = entry["family"].replace("/", ".")
    out_path = os.path.join(
        FIG_DIR,
        f"{safe_family}_{entry['pdb_code']}_patch_dist_swarm.svg"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    df_2p2idb = load_metadata()
    inh_emb = load_inhibitor_embeddings(df_2p2idb)
    ppi_points, dist_df, ppi_emb = load_ppi_patches(df_2p2idb)
    inh_emb = run_retrieval(inh_emb, ppi_emb, dist_df)
    report_stats(inh_emb)

    plot_stripplot(inh_emb, "enrichment_at_k", f"Enrichment@{K}",
                   f"protein_enrichment_at_k_{K}.svg", yticks=range(7))
    plot_stripplot(inh_emb, "precision_at_k", f"Precision@{K}",
                   f"protein_precision_at_k_{K}.svg")
    plot_swarm_by_name(inh_emb, ppi_points, SWARM_FAMILY, SWARM_PDB_CODE)


if __name__ == "__main__":
    main()
