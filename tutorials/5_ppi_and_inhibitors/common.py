"""Retrieval metrics and superfamily aggregation, shared by both analysis scripts."""

import os

import numpy as np
import pandas as pd

K = 10

# 2P2Idb labels the bromodomain superfamilies with bare Roman numerals.
ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}
DISPLAY = {"MENIN": "Menin", "ZIPA": "ZipA", "INTERLEUKIN": "Interleukin"}


def precision_at_k(emb_dist, geom_dist, geom_thresh, k=K):
    """Fraction of the k embedding-nearest candidates that are within geom_thresh."""
    emb_dist = np.asarray(emb_dist).reshape(-1)
    geom_dist = np.asarray(geom_dist).reshape(-1)
    k = min(k, max(K, int(emb_dist.size * 0.1)))
    if emb_dist.size <= k:
        k = 1
    return float(np.mean(geom_dist[np.argsort(emb_dist)[:k]] <= geom_thresh))


def fold_change_at_k(emb_dist, geom_dist, geom_thresh, k=K):
    """Precision@k over the fraction of all candidates that are within geom_thresh.

    A uniform random ranking gives 1.
    """
    geom_dist = np.asarray(geom_dist).reshape(-1)
    base_rate = float(np.mean(geom_dist <= geom_thresh))
    if base_rate == 0.0:
        return float("nan")
    return precision_at_k(emb_dist, geom_dist, geom_thresh, k) / base_rate


def superfamily_label(name):
    if name in ROMAN:
        return f"Bromodomain {name}"
    return DISPLAY.get(name, name)


def by_superfamily(values, sfmap):
    """Per-complex values, indexed by family, to one unweighted value per superfamily."""
    values = values.dropna().astype(float)
    groups = pd.Index([sfmap[f] for f in values.index], name="superfamily")
    return values.groupby(groups).mean()


def report(values, sfmap, what, reference=None):
    """Print the per-superfamily table and the summary lines."""
    per_sf = by_superfamily(values, sfmap).sort_values(ascending=False)
    table = pd.DataFrame({
        "superfamily": [superfamily_label(s) for s in per_sf.index],
        what: per_sf.values,
        "complexes": [sum(1 for f in values.index if sfmap[f] == s) for s in per_sf.index],
    })
    print(f"\n{what} by superfamily ({len(values)} complexes, {len(per_sf)} superfamilies):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if reference is not None:
        print(f"\n  superfamilies with {what} > {reference:g}: "
              f"{int((per_sf > reference).sum())}/{len(per_sf)}")
    print(f"  mean {what} over superfamilies: {per_sf.mean():.3f}")
    print(f"  mean {what} over complexes:    {values.mean():.3f}")
    return per_sf


def load_metadata(data_dir):
    """One row per protein-inhibitor entry, plus the family to superfamily map."""
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    sfmap = dict(zip(meta["family"], meta["superfamily"]))
    return meta.drop_duplicates("entry_id").set_index("entry_id"), sfmap


def plot_by_superfamily(values, sfmap, ylabel, out_path, reference=None):
    """One point per complex, grouped by superfamily, with the superfamily mean drawn."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = pd.DataFrame({
        "value": values.values,
        "superfamily": [superfamily_label(sfmap[f]) for f in values.index],
    }).sort_values("superfamily")
    order = sorted(frame["superfamily"].unique())

    plt.figure(figsize=(max(3.0, 0.62 * len(order) + 1.4), 2.8))
    ax = sns.stripplot(x="superfamily", y="value", data=frame, order=order, size=5,
                       jitter=0.18, color="#00C4C7", edgecolor="black", linewidth=0.3)
    for i, name in enumerate(order):
        mean = frame.loc[frame["superfamily"] == name, "value"].mean()
        ax.plot([i - 0.3, i + 0.3], [mean, mean], color="black", lw=1.1, zorder=5)
        if i:
            plt.axvline(i - 0.5, color="lightgray", linewidth=0.5, zorder=0)
    if reference is not None:
        plt.axhline(reference, linestyle="--", color="black", linewidth=0.5)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel("")
    sns.despine()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_top_bottom(distances, ylabel, ylim, title, out_path, reference_idx, seed=0):
    """Violin and strip plot of the top-K, bottom-K and reference candidate distances."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "distance": distances,
        "type": [f"Top {K}"] * K + [f"Bottom {K}"] * K + ["Reference"] * len(reference_idx),
    })
    plt.figure(figsize=(3, 3))
    ax = sns.violinplot(x="type", y="distance", data=data, color="lightgray",
                        alpha=0.4, inner=None)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("p", ["#D86ECC", "#6805F2"])
    pos = dict(zip([t.get_text() for t in ax.get_xticklabels()], ax.get_xticks()))
    x = data["type"].map(pos).values + rng.uniform(-0.2, 0.2, size=len(data))
    ax.scatter(x, data["distance"], c=data["distance"], cmap=cmap, s=10,
               edgecolor="black", linewidth=0.2)
    sns.despine()
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.xlabel("")
    plt.title(title, fontsize=9)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
