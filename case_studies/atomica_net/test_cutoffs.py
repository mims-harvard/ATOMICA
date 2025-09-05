import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional

pesto_cutoff = 70
plddt_cutoff = 70

def create_similarity_graph(similarity_matrix: np.ndarray, 
                          k: int = 50, 
                          min_similarity: Optional[float] = None) -> nx.Graph:
    """
    Create an undirected graph where each node is connected to its k most similar nodes,
    and all edges have a similarity greater than or equal to min_similarity.
    
    Args:
        similarity_matrix: A square numpy array containing similarity scores
        k: Number of most similar neighbors to connect (default: 50)
        min_similarity: Minimum similarity threshold for edges (default: None)
    
    Returns:
        NetworkX undirected graph
    """
    # Ensure the similarity matrix is square
    n = similarity_matrix.shape[0]
    assert similarity_matrix.shape == (n, n), "Similarity matrix must be square"
    
    # Create empty graph
    G = nx.Graph()
    
    # Add all nodes to the graph
    G.add_nodes_from(range(n))
    
    # For each node, find its k most similar neighbors
    # Use argpartition for efficiency (partial sort)
    for i in range(n):
        # Get similarities for current node
        similarities = similarity_matrix[i]
        
        # We don't want to include the node itself
        similarities[i] = -np.inf
        
        # Get indices of k largest similarities
        # argpartition is more efficient than argsort as it only partially sorts
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        
        # Filter by minimum similarity if specified
        if min_similarity is not None:
            top_k_indices = top_k_indices[similarities[top_k_indices] >= min_similarity]
        
        # Add edges
        for j in top_k_indices:
            G.add_edge(i, j, weight=similarities[j])
    
    return G

modality_cutoffs = {
    "protein": [0.96, 0.965, 0.97, 0.975, 0.98],
    "lipid": [0.97, 0.971, 0.972, 0.973, 0.974, 0.975],
    "nucleic_acid": [0.969, 0.97, 0.971],
    "ligand": [0.965, 0.969, 0.97, 0.971, 0.975],
    "ion": [0.965, 0.964, 0.963, 0.962, 0.961, 0.960],
}

import sys
modality = sys.argv[1]
cutoffs = modality_cutoffs[modality]

similarity_matrices = []
uniprot_to_node_idxs = []

embeddings_data_dirs = [
    "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/embeddings/embeddings_fixed_irreps_v1/",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/embeddings/embeddings_fixed_irreps_v2/",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/embeddings/embeddings_fixed_irreps_v3/",
]
for cutoff in cutoffs:
    for embeddings_data_dir in embeddings_data_dirs:
        with open(os.path.join(embeddings_data_dir, f"pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_{modality}.pkl"), "rb") as f:
            embeddings_dataset = pickle.load(f)

        embeddings = np.array([x['graph_embedding'] for x in embeddings_dataset])
        uniprot_to_node_idx = {x['id']: i for i, x in enumerate(embeddings_dataset)}
        uniprot_to_node_idxs.append(uniprot_to_node_idx)

        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrices.append(similarity_matrix)

    assert list(uniprot_to_node_idxs[0].keys()) == list(uniprot_to_node_idxs[1].keys()) == list(uniprot_to_node_idxs[2].keys())
    similarity_matrix = np.mean(similarity_matrices, axis=0)
    std = np.std(similarity_matrices, axis=0)

    indices = np.triu_indices_from(similarity_matrix, k=1)
    filtered_indices = indices[0][similarity_matrix[indices] > cutoff], indices[1][similarity_matrix[indices] > cutoff]

    print(f"-----------------{modality}-----------------")
    print("Cutoff: ", cutoff)
    print("Standard deviation: ", np.mean(std))
    print("Number of embeddings: ", embeddings.shape)
    print("num edges: ", filtered_indices[0].shape[0])

    G = create_similarity_graph(similarity_matrix, k=50, min_similarity=cutoff)
    # G.add_nodes_from(range(len(embeddings)))
    # edges = zip(filtered_indices[0], filtered_indices[1], similarity_matrix[filtered_indices])
    # G.add_weighted_edges_from(edges)
    size_of_largest_connected_component = max([len(x) for x in nx.connected_components(G)])
    whole_graph_density = nx.density(G)
    whole_graph_clustering = nx.average_clustering(G)
    print("size of largest connected component: ", size_of_largest_connected_component)
    print("proportion of largest connected component: ", size_of_largest_connected_component / len(embeddings))
    print("Average degree: ", np.mean([x[1] for x in G.degree()]))