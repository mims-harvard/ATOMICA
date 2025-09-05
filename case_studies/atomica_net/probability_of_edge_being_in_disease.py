import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import os
from disease_categorisation import disease_categories
import argparse

parser = argparse.ArgumentParser(description="Calculate disease association statistics for protein pairs based on cosine similarity.")
parser.add_argument('--disease_category', type=str, default='all')
args = parser.parse_args()

pesto_cutoff = 70
plddt_cutoff = 70

uniprot_human_df = pd.read_csv("/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_model_organism_9606_2024_12_16.tsv", sep="\t", usecols=['Entry', 'Entry Name', 'Protein names', 'Gene Names'])
uniprot_to_gene = dict(zip(uniprot_human_df['Entry'], uniprot_human_df['Gene Names']))
uniprot_to_protein_name = dict(zip(uniprot_human_df['Entry'], uniprot_human_df['Protein names']))

with open("chosen_diseases.txt", "r") as f:
    chosen_disease_names = [x.strip() for x in f.readlines()]

disease_associations = pd.read_csv('/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/disease_network/open_target_2024_09/association_by_overall_indirect_disease_counts_cutoff_0.5.csv')
disease_associations = disease_associations[disease_associations['diseaseName'].isin(chosen_disease_names)]
disease_id_to_name = dict(zip(disease_associations['diseaseId'], disease_associations['diseaseName']))
disease_name_to_id = dict(zip(disease_associations['diseaseName'], disease_associations['diseaseId']))
print("Number of unique diseases: ", len(disease_associations['diseaseId'].unique()))

modality_counts = pd.read_csv(f"/n/holylabs/LABS/mzitnik_lab/Users/afang/ATOMICA/case_studies/atomica_net/modality_chosen_disease_counts_pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_fix_irreps_ensemble.csv")

with open("/n/holylabs/LABS/mzitnik_lab/Users/afang/ATOMICA/case_studies/atomica_net/modality_graphs_20250308.pkl", "rb") as f:
    similar_modality_graphs = pickle.load(f)

embeddings_data_dirs = [
    "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/embeddings/embeddings_fixed_irreps_v1/",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/embeddings/embeddings_fixed_irreps_v2/",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/embeddings/embeddings_fixed_irreps_v3/",
]

protein_pairs = []
agg_results = []

modalities = ['protein', 'lipid', 'nucleic_acid', 'ion', 'ligand']
for modality in modalities:
    similarity_matrices = []
    uniprot_to_node_idxs = []
    for embeddings_data_dir in embeddings_data_dirs:
        with open(os.path.join(embeddings_data_dir, f"pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_{modality}.pkl"), "rb") as f:
            embeddings_dataset = pickle.load(f)

        embeddings = np.array([x['graph_embedding'] for x in embeddings_dataset])
        uniprot_to_node_idx = {x['id']: i for i, x in enumerate(embeddings_dataset)}
        node_idx_to_uniprot = {i: x['id'] for i, x in enumerate(embeddings_dataset)}
        uniprot_to_node_idxs.append(uniprot_to_node_idx)

        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrices.append(similarity_matrix)
    
    assert list(uniprot_to_node_idxs[0].keys()) == list(uniprot_to_node_idxs[1].keys()) == list(uniprot_to_node_idxs[2].keys())
    similarity_matrix = np.mean(similarity_matrices, axis=0)

    rows, cols = np.triu_indices_from(similarity_matrix, k=1)

    valid_disease_ids = modality_counts[(modality_counts['modality'] == modality) & (modality_counts['num_proteins']>=25)]['diseaseId'].tolist()
    disease_associations = disease_associations[disease_associations['diseaseId'].isin(valid_disease_ids)]
    if args.disease_category != 'all':
        if args.disease_category in disease_categories:
            selected_disease_ids = disease_categories[args.disease_category]
            disease_associations = disease_associations[disease_associations['diseaseId'].isin(selected_disease_ids)]
        else:
            raise ValueError(f"Invalid disease category: {args.disease_category}. Available categories: {list(disease_categories.keys())}")
    protein_disease_associations = disease_associations.groupby('uniprot_targetId')['diseaseId'].agg(set).to_dict()

    G = similar_modality_graphs[modality]['graph']
    edges = set(G.edges())

    for node1, node2 in tqdm(zip(rows, cols), total=len(rows), desc=modality):
        uniprot1 = node_idx_to_uniprot[node1]
        uniprot2 = node_idx_to_uniprot[node2]
        diseases1 = protein_disease_associations.get(uniprot1, set())
        diseases2 = protein_disease_associations.get(uniprot2, set())
        is_overlap = int(len(diseases1.intersection(diseases2)) > 0)
        is_edge = int((node1, node2) in edges or (node2, node1) in edges)
        protein_pairs.append({
            'cosine_similarity': similarity_matrix[node1, node2],
            'disease_association': is_overlap,
            'edge': is_edge,
        })
    protein_pairs_df = pd.DataFrame(protein_pairs)
    if args.disease_category == 'all':
        protein_pairs_df.to_csv(f"results/protein_pairs_{modality}.csv", index=False)

    percentiles = np.arange(0, 101, 5)
    quantile_cutoffs = np.percentile(protein_pairs_df['cosine_similarity'], percentiles)
    protein_pairs_df['quantile_bin'] = pd.cut(protein_pairs_df['cosine_similarity'], bins=quantile_cutoffs, right=False)
    protein_pairs_df['quantile_bin'] = protein_pairs_df['quantile_bin'].astype(str)
    statistics = protein_pairs_df.groupby('quantile_bin')['disease_association'].agg(['mean', 'std', 'count']).reset_index()
    statistics['modality'] = modality
    agg_results.append(statistics)
agg_results = pd.concat(agg_results)
agg_results.to_csv(f"results/protein_pairs_cosine_binned_statistics_{pesto_cutoff}_{plddt_cutoff}_{args.disease_category}.csv", index=False)