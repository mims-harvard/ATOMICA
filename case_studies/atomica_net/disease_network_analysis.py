import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from disease_algorithms import DIAMOnD, random_walk_with_restart, neighborhood_approach
import sys

split_idx = int(sys.argv[1])
pesto_cutoff = 70
plddt_cutoff = 70

chosen_diseases = pd.read_csv('/n/holylfs06/LABS/mzitnik_lab/Lab/afang/disease_network/open_target_2024_09/chosen_diseases.csv')
disease_associations = pd.read_csv('/n/holylfs06/LABS/mzitnik_lab/Lab/afang/disease_network/open_target_2024_09/association_by_overall_indirect_disease_counts_cutoff_0.5.csv')
disease_associations = disease_associations[disease_associations['diseaseId'].isin(chosen_diseases['diseaseId'])]

with open("modality_graphs_20250308.pkl", "rb") as f:
    modality_graphs = pickle.load(f)

# set up cross validation splits
with open(f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/disease_network/open_target_2024_09/cv_splits_fix_irreps_ensemble/cv_split_{split_idx}.pkl", "rb") as f:
    cv_splits = pickle.load(f)
print("Number of splits: ", len(cv_splits))


min_modality_protein_count = 25
modality_disease_counts = pd.read_csv(f"/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/case_studies/network_analysis/modality_chosen_disease_counts_pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_fix_irreps_ensemble.csv")
modality_disease_counts = pd.pivot_table(modality_disease_counts, 
                            values="num_proteins", 
                            index='diseaseId', 
                            columns='modality', 
                            fill_value=0)


# run disease association algorithms
results = []
max_number_of_added_nodes = 500
for (disease_id, fold), data_split in tqdm(cv_splits.items(), total=len(cv_splits)):
    for modality in modality_graphs:
        if modality_disease_counts.loc[disease_id][modality] < min_modality_protein_count:
            print(f"Skipping {disease_id} {fold} {modality} due to low protein count")
            continue
        G = modality_graphs[modality]['graph']
        uniprot_to_node_idx = modality_graphs[modality]['uniprot_to_node_idx']
        node_idx_to_uniprot = {v: k for k, v in uniprot_to_node_idx.items()}
        largest_component = modality_graphs[modality]['largest_component']
        A = modality_graphs[modality]['adjacency_matrix']

        seed_proteins = data_split['train_proteins']
        seed_nodes = [uniprot_to_node_idx[x] for x in seed_proteins if x in uniprot_to_node_idx and uniprot_to_node_idx[x] in largest_component]
        gt_proteins = data_split['test_proteins']
        gt_nodes = [uniprot_to_node_idx[x] for x in gt_proteins if x in uniprot_to_node_idx and uniprot_to_node_idx[x] in largest_component]
        gt_proteins_in_graph = [node_idx_to_uniprot[x] for x in gt_nodes]

        if len(gt_nodes) == 0 or len(seed_nodes) == 0:
            results.append((
                disease_id, fold, modality, 
                pd.NA, pd.NA, pd.NA,
                pd.NA, pd.NA, pd.NA,
                pd.NA, pd.NA, pd.NA,
                pd.NA
            ))
            continue
        
        added_nodes = DIAMOnD(
            G,
            seed_nodes,
            max_number_of_added_nodes=max_number_of_added_nodes,
            alpha=10,
            outfile=None
        )
        added_node_indexes = [x[0] for x in added_nodes]
        num_found_diamond = len(set(added_node_indexes) & set(gt_nodes))
        recall_diamond = num_found_diamond/len(gt_nodes)
        found_diamond_proteins = [node_idx_to_uniprot[x] for x in added_node_indexes]

        p = random_walk_with_restart(G, A, seed_nodes)
        p_node_idx = np.array(list(sorted(largest_component)))
        rw_nodes = list(zip(p_node_idx, p))
        rw_nodes = [(node_idx, p) for node_idx, p in rw_nodes if not node_idx in seed_nodes]
        rw_nodes = sorted(rw_nodes, key=lambda x: x[1], reverse=True)[:min(max_number_of_added_nodes, len(rw_nodes))]
        added_node_indexes = [x[0] for x in rw_nodes]
        num_found_rw = len(set(added_node_indexes) & set(gt_nodes))
        recall_rw = num_found_rw/len(gt_nodes)
        found_rw_proteins = [node_idx_to_uniprot[x] for x in added_node_indexes]
        
        predicted_proteins = neighborhood_approach(G, seed_nodes)
        predicted_proteins = sorted(predicted_proteins.items(), key=lambda x: x[1], reverse=True)
        added_node_indexes = [x[0] for x in predicted_proteins[:min(max_number_of_added_nodes, len(predicted_proteins))]]        
        num_found_neighborhood = len(set(added_node_indexes) & set(gt_nodes))
        recall_neighborhood = num_found_neighborhood/len(gt_nodes)
        found_neighborhood_proteins = [node_idx_to_uniprot[x] for x in added_node_indexes]
        
        results.append((
            disease_id, fold, modality, 
            num_found_diamond, recall_diamond, found_diamond_proteins, 
            num_found_rw, recall_rw, found_rw_proteins, 
            num_found_neighborhood, recall_neighborhood, found_neighborhood_proteins,
            gt_proteins_in_graph,
        ))

results_df = pd.DataFrame(results, columns=[
    'diseaseId', 'fold', 'modality',
    'num_found_diamond', 'recall_diamond', 'found_diamond',
    'num_found_rw', 'recall_rw', 'found_rw',
    'num_found_neighborhood', 'recall_neighborhood', 'found_neighborhood',
    'gt_proteins',
])

disease_id_to_name = disease_associations[['diseaseId', 'diseaseName']].drop_duplicates().set_index('diseaseId')['diseaseName']
results_df['diseaseName'] = results_df['diseaseId'].map(disease_id_to_name)

with open(f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/human_proteome/disease_network_analysis/results_at_pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_max_{max_number_of_added_nodes}_split_{split_idx}_fix_irreps_ensemble.pkl", "wb") as f:
    pickle.dump(results_df, f)