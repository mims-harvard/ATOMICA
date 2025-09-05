from openTSNE import TSNE
from umap import UMAP
import numpy as np
import pandas as pd
import os 
from datetime import datetime
import pickle
import os


def main(graph_embedding_files, out_dir, projection_type="umap", perplexity=30, n_neighbors=20, min_dist=0.1):
    source = []
    embeddings = []
    item_ids = []
    for key, file in graph_embedding_files:
        print(f"Loading {file}")
        with open(file, "rb") as f:
            data = pickle.load(f)
            embeddings1 = np.array([x["graph_embedding"] for x in data])
            item_ids1 = [x["id"] for x in data]
        item_ids = item_ids + item_ids1
        embeddings.append(embeddings1)
        source = source + [key] * embeddings1.shape[0]

    embeddings = np.concatenate(embeddings, axis=0)
    print("Embeddings shape: ", embeddings.shape)

    if projection_type == "umap":
        umap_2d = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            init="random",
            random_state=0,
        )
        proj_2d_graph = umap_2d.fit_transform(embeddings)
        metrics_str = f'{projection_type}_neighbors_{n_neighbors}_min_dist_{min_dist}'
    elif projection_type == "tsne":
        tsne = TSNE(
            perplexity=perplexity,
            metric="cosine",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        proj_2d_graph = tsne.fit(embeddings)
        metrics_str = f'{projection_type}_perplexity_{perplexity}_{tsne.metric}'
    else:
        raise ValueError(f"Invalid projection type: {projection_type}")

    df = pd.DataFrame(columns=["x", "y", "source"])
    df["x"] = proj_2d_graph[:, 0]
    df["y"] = proj_2d_graph[:, 1]
    df["source"] = source
    df["item_id"] = item_ids
    os.makedirs(out_dir, exist_ok=True)
    formatted_date = datetime.today().strftime('%Y%m%d')
    out_fname = f'graph_embedding_2d_all_modalities_{metrics_str}_{formatted_date}.csv'
    df.to_csv(os.path.join(out_dir, out_fname), index=False)
    print(f"Saved to {os.path.join(out_dir, out_fname)}")

if __name__ == "__main__":
    embedding_dir = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/latent_space/embeddings/version54_epoch73/"
    out_dir = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/latent_space/embeddings/version54_epoch73/proj_2d/"

    graph_embedding_files = []
    for fname in os.listdir(embedding_dir):
        if not fname.endswith(".pkl"):
            continue
        datatype = fname.replace(".pkl", "")
        graph_embedding_files.append((datatype, f"{embedding_dir}/{fname}"))

    for num_neighbours in [100]: # 10, 30, 50
        for min_dist in [0.001, 0.01, 0.1, 0.5]:
            main(graph_embedding_files, out_dir, projection_type="umap", n_neighbors=num_neighbours, min_dist=min_dist)