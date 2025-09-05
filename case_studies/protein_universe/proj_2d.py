import numpy as np
import os
import pickle
from openTSNE import TSNE
import pandas as pd
from collections import defaultdict

graph_embeddings = []
graph_ids = []


embedding_dir = "/n/holyscratch01/mzitnik_lab/afang/InteractNN_embeddings/version_149_epoch115_step1192946/"
for emb_file in os.listdir(embedding_dir):
    if not emb_file.endswith(".pkl") or not emb_file.startswith('is_dark_90_plddt_PeSTo_80_'):
        continue
    binder_type = emb_file.split("_")[-1].split(".")[0]
    with open(os.path.join(embedding_dir, emb_file), "rb") as f:
        embeddings = pickle.load(f)
    graph_embeddings.extend([x['graph_embedding'] for x in embeddings])
    graph_ids.extend([f"{x['id']}_{binder_type}" for x in embeddings])
graph_embeddings = np.array(graph_embeddings)

combined_df = pd.DataFrame({"id": graph_ids})
perplexities = [5, 10, 20, 30, 50, 100]
for perplexity in perplexities:
    tsne = TSNE(
        perplexity=perplexity,
        metric="cosine",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    proj_2d_graph = tsne.fit(graph_embeddings)

    df = pd.DataFrame({
        f"x_{perplexity}": proj_2d_graph[:, 0],
        f"y_{perplexity}": proj_2d_graph[:, 1],
        "id": graph_ids
    })
    combined_df = pd.merge(combined_df, df, on="id")

combined_df.to_csv(f"{embedding_dir}/is_dark_90_plddt_PeSTo_80_graph_embedding_2d.csv", index=False)