"""
Tutorial: Cross-modality interface comparison for protein-peptide PPI inhibitors

Compares ATOMICA embeddings of orthosteric small-molecule inhibitors to
ATOMICA embeddings of the peptide partner in the native protein-peptide
complex they inhibit. The goal is to test whether embedding similarity
between inhibitor blocks and peptide blocks localizes to the spatially
corresponding regions of the interface after 3D alignment.

Pipeline:
  1. Load 2P2IDB metadata and ATOMICA embeddings for matched
     protein-peptide PPI structures and their protein-inhibitor
     structures.
  2. Load cached pairwise block comparisons
     (``data/peptide_inhibitor_block_results.parquet``). Each row holds,
     for one inhibitor-PPI match, the cosine distance matrix between
     ATOMICA block embeddings and the Kabsch-aligned block-center
     distance matrix.
  3. Compute Precision@10 and Enrichment@10 using a 4.0 Å spatial
     threshold.
  4. Run Spearman correlations per family with FDR correction and a
     binomial test.
  5. Save figures to ``figures/``: enrichment stripplot, precision
     stripplot, and a MENIN/MLL (ligand 2SE, PDB 4OG7 vs. PPI 4GQ6)
     violin+strip plot of block spatial distances for top-10,
     bottom-10, and reference embedding pairs.

Usage:
  python tutorial_protein_peptide_inhibitors.py
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
GEOM_THRESH = 4.0

# Featured swarm-plot example: the MENIN/MLL MIV-7 inhibitor.
# Selected by system identity, not by rank.
SWARM_FAMILY = "MENIN/MLL"
SWARM_LIG_CODE = "2SE"       # PDB ligand code for MIV-7
SWARM_INHIBITOR_PDB = "4OG7"  # protein-inhibitor PDB


def load_metadata():
    df1 = pd.read_csv(os.path.join(DATA_DIR, "2p2idb.csv"), sep=";")
    df2 = pd.read_csv(os.path.join(DATA_DIR, "ppi_inhibitor_mapping.csv"))
    df = df2.merge(df1[["PDBProtProt", "PDBProtLig", "InChI"]],
                   on=["PDBProtProt", "PDBProtLig"], how="left")
    df = df[df["PDBProtProt"] != "na"]
    return df


def load_embeddings(df_2p2idb):
    ppi_emb = pd.read_parquet(os.path.join(DATA_DIR, "peptide_partners_embeddings.parquet"))
    ppi_in = pd.read_parquet(os.path.join(DATA_DIR, "peptide_partners_processed.parquet"))
    ppi_emb = ppi_emb.merge(ppi_in, on="id", how="left")
    ppi_to_family = df_2p2idb[["PDBProtProt", "Chain_Target", "Chain_Partner", "Family"]].drop_duplicates()
    ppi_emb["pdb_id"] = ppi_emb["id"].str.split("_").str[0]
    ppi_emb = ppi_emb.merge(ppi_to_family, left_on="pdb_id", right_on="PDBProtProt", how="left")

    inh_emb = pd.read_parquet(os.path.join(DATA_DIR, "inhibitors_embeddings.parquet"))
    inh_in = pd.read_parquet(os.path.join(DATA_DIR, "inhibitors_processed.parquet"))
    inh_emb = pd.concat([inh_in, inh_emb.drop(columns=["id"])], axis=1)
    inh_emb["2P2IDB_ID"] = inh_emb["id"].str.split("_").str[0]
    meta = pd.read_csv(os.path.join(DATA_DIR, "inhibitors_metadata.csv"))
    meta.rename(columns={
        "pdb_id": "2P2IDB_ID", "chain1": "Chain_Target",
        "chain2": "ChainID_Ligand", "pdb_code": "PDBProtLig"
    }, inplace=True)
    inh_emb = inh_emb.merge(meta, on="2P2IDB_ID", how="left")
    return ppi_emb, inh_emb


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


def load_block_results():
    path = os.path.join(DATA_DIR, "peptide_inhibitor_block_results.parquet")
    br = pd.read_parquet(path)
    n_singleton = br["min_dist"].isna().sum()
    print(f"Inhibitor-PPI pairs with only singleton blocks: {n_singleton}")
    br = br.dropna(subset=["min_dist"]).copy()
    br = br[br["min_dist"] < 2].reset_index(drop=True)
    br = br[br["block_emb_dist"].apply(lambda x: len(x) > K)].copy()
    br["precision_at_k"] = br.apply(
        lambda r: precision_at_k(r["block_emb_dist"], r["block_coords_dist"]), axis=1
    )
    br["enrichment_at_k"] = br.apply(
        lambda r: enrichment_at_k(r["block_emb_dist"], r["block_coords_dist"]), axis=1
    )
    return br


def report_stats(br, families):
    print("Families with >10 blocks:",
          br["family"].nunique(), "out of", len(families))
    print("Inhibitor structures in these families:", len(br))
    print("Inhibitor-PPI pairs with enrichment > 1:",
          int((br["enrichment_at_k"] > 1).sum()), "out of", len(br))
    print("Inhibitor-PPI pairs with enrichment > 1.5:",
          int((br["enrichment_at_k"] > 1.5).sum()))

    best = br.groupby("family").agg({
        "precision_at_k": "max", "enrichment_at_k": "max"
    }).reset_index()
    print("Families with best enrichment > 1:",
          int((best["enrichment_at_k"] > 1).sum()), "out of", len(best))
    print("Families with best enrichment > 1.5:",
          int((best["enrichment_at_k"] > 1.5).sum()), "out of", len(best))

    agg = br.groupby("family").agg({
        "block_emb_dist": lambda x: np.concatenate(list(x)),
        "block_coords_dist": lambda x: np.concatenate(list(x)),
    }).reset_index()
    agg[["spearmanr", "p_value"]] = agg.apply(
        lambda r: pd.Series(scipy.stats.spearmanr(r["block_emb_dist"], r["block_coords_dist"])),
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
    return agg


def plot_stripplot(br, value_col, ylabel, out_name, yticks=None):
    plt.figure(figsize=(12, 12 / 5))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gray_to_aqua", ["#B0B0B0", "#00C4C7"]
    )
    order = sorted(br["family"].unique())
    ax = sns.stripplot(
        x="family", y=value_col, data=br, size=4, alpha=0.8, jitter=0.25,
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
    for i in range(br["family"].nunique() - 1):
        plt.axvline(i + 0.5, color="lightgray", linewidth=0.5, zorder=0)
    sns.despine()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def pick_entry_by_name(br, family, lig_code=None, inhibitor_pdb=None):
    """Select a single block-results row by system identity."""
    sub = br[br["family"] == family]
    if lig_code is not None:
        sub = sub[sub["lig_code"] == lig_code]
    if inhibitor_pdb is not None:
        sub = sub[sub["inhibitor_pdb_id"] == inhibitor_pdb]
    if len(sub) == 0:
        raise ValueError(
            f"No block-results entry for family={family!r}, "
            f"lig_code={lig_code!r}, inhibitor_pdb={inhibitor_pdb!r}"
        )
    return sub.iloc[0]


def plot_swarm_by_name(br, family, lig_code, inhibitor_pdb):
    entry = pick_entry_by_name(br, family, lig_code, inhibitor_pdb)
    print(f"Swarm-plot example: {entry['family']} inhibitor {entry['inhibitor_pdb_id']} "
          f"chain {entry['inhibitor_chain_target']} ligand {entry['lig_code']} "
          f"vs. PPI {entry['ppi_pdb_id']} "
          f"chains {entry['ppi_chain_target']}/{entry['ppi_chain_partner']}, "
          f"n_pairs={len(entry['block_coords_dist'])}")

    closest = np.argsort(entry["block_emb_dist"])[:K]
    furthest = np.argsort(entry["block_emb_dist"])[-K:]
    reference = np.arange(len(entry["block_emb_dist"]))
    idx = np.concatenate([closest, furthest, reference])
    types = np.array([f"Top {K}"] * len(closest) +
                     [f"Bottom {K}"] * len(furthest) +
                     ["Reference"] * len(reference))
    plot_data = pd.DataFrame({
        "Distance between Blocks": entry["block_coords_dist"][idx],
        "type": types,
    })

    plt.figure(figsize=(3, 3))
    ax = sns.violinplot(x="type", y="Distance between Blocks", data=plot_data,
                        inner=None, color="lightgray", alpha=0.4)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gray_to_purple", ["#D86ECC", "#6805F2"]
    )
    xticks = ax.get_xticks()
    categories = [t.get_text() for t in ax.get_xticklabels()]
    pos_dict = dict(zip(categories, xticks))
    x_vals = plot_data["type"].map(pos_dict).values
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.2, 0.2, size=len(x_vals))
    ax.scatter(x_vals + jitter, plot_data["Distance between Blocks"],
               c=plot_data["Distance between Blocks"], cmap=cmap,
               s=10, edgecolor="black", linewidth=0.2)
    sns.despine()
    plt.ylabel("Distance between Blocks (Å)")
    plt.ylim(0, 25)
    plt.xlabel("")
    plt.title(f"{entry['family']} {entry['inhibitor_pdb_id']} "
              f"{entry['inhibitor_chain_target']} {entry['lig_code']} v.s. "
              f"{entry['ppi_pdb_id']} {entry['ppi_chain_target']} "
              f"{entry['ppi_chain_partner']}\n(n={len(entry['block_coords_dist'])})",
              fontsize=10)
    plt.tight_layout()
    safe_family = entry["family"].replace("/", ".")
    out_path = os.path.join(
        FIG_DIR,
        f"{safe_family}_{entry['inhibitor_pdb_id']}_{entry['lig_code']}_block_dist_swarm.svg"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    df_2p2idb = load_metadata()
    ppi_emb, inh_emb = load_embeddings(df_2p2idb)

    families = list(
        set(inh_emb["Family"].unique()).intersection(set(ppi_emb["Family"].unique()))
    )
    print(f"Shared families between inhibitors and protein-peptide PPIs: {len(families)}")

    br = load_block_results()
    report_stats(br, families)

    plot_stripplot(br, "enrichment_at_k", f"Enrichment@{K}",
                   f"peptide_enrichment_at_k_{K}.svg", yticks=range(9))
    plot_stripplot(br, "precision_at_k", f"Precision@{K}",
                   f"peptide_precision_at_k_{K}.svg")
    plot_swarm_by_name(br, SWARM_FAMILY, SWARM_LIG_CODE, SWARM_INHIBITOR_PDB)


if __name__ == "__main__":
    main()
