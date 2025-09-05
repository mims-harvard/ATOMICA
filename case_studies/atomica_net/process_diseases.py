import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

pesto_cutoff = 70
plddt_cutoff = 70

with open("modality_graphs_20250306.pkl", "rb") as f:
    modality_graphs = pickle.load(f)

disease_associations = pd.read_csv('/n/holylfs06/LABS/mzitnik_lab/Lab/afang/disease_network/open_target_2024_09/association_by_overall_indirect_disease_counts_cutoff_0.5.csv')
disease_id_to_name = dict(zip(disease_associations['diseaseId'], disease_associations['diseaseName']))
chosen_diseases = pd.read_csv('/n/holylfs06/LABS/mzitnik_lab/Lab/afang/disease_network/open_target_2024_09/chosen_diseases.csv')

modality_disease_counts = []
for disease in tqdm(chosen_diseases['diseaseId'].unique(), total=len(chosen_diseases['diseaseId'].unique())):
    proteins = disease_associations[disease_associations['diseaseId'] == disease]['uniprot_targetId'].values.tolist()
    for modality in modality_graphs:
        G = modality_graphs[modality]['graph'].copy()
        uniprot_to_node_idx = modality_graphs[modality]['uniprot_to_node_idx']
        largest_component = modality_graphs[modality]['largest_component']

        uniprot_to_node_idx_keys = set(uniprot_to_node_idx.keys())
        disease_proteins = [uniprot_to_node_idx[x] for x in proteins if x in uniprot_to_node_idx_keys]
        disease_proteins = set(disease_proteins) & set(largest_component)

        modality_disease_counts.append({"diseaseId": disease, "modality": modality, "num_proteins": len(disease_proteins)})
modality_disease_counts = pd.DataFrame(modality_disease_counts)
modality_disease_counts['diseaseName'] = modality_disease_counts['diseaseId'].map(disease_id_to_name)
modality_disease_counts.to_csv(f"modality_chosen_disease_counts_pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_fix_irreps_ensemble.csv", index=False)
