import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm
import sys

pesto_cutoff = 70
plddt_cutoff = 70

cpu_id = int(sys.argv[1])


with open("modality_graphs_20250308.pkl", "rb") as f:
    modality_graphs = pickle.load(f)

uniprot_human_df = pd.read_csv("/n/holylfs06/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_model_organism_9606_2024_12_16.tsv", sep="\t", usecols=['Entry', 'Entry Name', 'Protein names', 'Gene Names'])
uniprot_to_gene = dict(zip(uniprot_human_df['Entry'], uniprot_human_df['Gene Names']))
uniprot_to_protein_name = dict(zip(uniprot_human_df['Entry'], uniprot_human_df['Protein names']))


chosen_diseases = pd.read_csv('/n/holylfs06/LABS/mzitnik_lab/Lab/afang/disease_network/open_target_2024_09/chosen_diseases.csv')
disease_associations = pd.read_csv('/n/holylfs06/LABS/mzitnik_lab/Lab/afang/disease_network/open_target_2024_09/association_by_overall_indirect_disease_counts_cutoff_0.5.csv')
disease_associations = disease_associations[disease_associations['diseaseId'].isin(chosen_diseases['diseaseId'])]
disease_id_to_name = dict(zip(disease_associations['diseaseId'], disease_associations['diseaseName']))
disease_name_to_id = dict(zip(disease_associations['diseaseName'], disease_associations['diseaseId']))
print("Number of unique diseases: ", len(disease_associations['diseaseId'].unique()))


modality_disease_counts = pd.read_csv(f"/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/case_studies/network_analysis/modality_chosen_disease_counts_pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_fix_irreps_ensemble.csv")
modality_disease_counts = pd.pivot_table(modality_disease_counts, 
                            values="num_proteins", 
                            index=['diseaseId', 'diseaseName'], 
                            columns='modality', 
                            fill_value=0)
modalities = ['ion', 'ligand', 'lipid', 'nucleic_acid', 'protein']
modality_disease_counts['is_valid'] = modality_disease_counts.apply(lambda row: sum([1 for x in modalities if row[x] >= 25]) >= 2, axis=1)

def get_disease_connectivity(modality, disease_id, seed=-1):
    # seed != None runs permutation test
    G = modality_graphs[modality]['graph'].copy()

    if seed != -1:
        random_network = nx.double_edge_swap(
            G.copy(), nswap=len(G.edges()), max_tries=len(G.edges())*10
        )
        G = random_network

    uniprot_to_node_idx = modality_graphs[modality]['uniprot_to_node_idx']
    node_idx_to_uniprot = {v: k for k, v in uniprot_to_node_idx.items()}
    largest_component = modality_graphs[modality]['largest_component']

    filtered_relations_df = disease_associations[disease_associations['diseaseId'] == disease_id].copy()
    number_of_disease_proteins = len(filtered_relations_df['uniprot_targetId'].unique())
    filtered_relations_df['node_idx'] = filtered_relations_df['uniprot_targetId'].map(uniprot_to_node_idx).fillna(-1).astype(int)
    filtered_relations_df = filtered_relations_df[(filtered_relations_df['node_idx'] != -1) & (filtered_relations_df['node_idx'].isin(largest_component))].copy()
    selected_disease_nodes = filtered_relations_df['node_idx'].tolist()

    if len(selected_disease_nodes) == 0:
        return {"modality": modality, "diseaseId": disease_id, "size_of_largest_connected_component": 0, 
                "number_of_nodes": 0, "size_of_largest_pathway_component": 0, "num_components": 0, 
                "density": 0, "clustering": 0, "proteins_in_largest_component": []}

    subgraph = G.subgraph(selected_disease_nodes)

    components = sorted(nx.connected_components(subgraph), key=len, reverse=True)

    largest_component = components[0]
    size_of_largest_component = len(largest_component)

    subgraph_num_components = nx.number_connected_components(subgraph)
    subgraph_density = nx.density(subgraph)
    subgraph_clustering = nx.average_clustering(subgraph)
    number_of_nodes = len(selected_disease_nodes)
    proteins = [node_idx_to_uniprot[x] for x in largest_component]
    
    return {"modality": modality, "diseaseId": disease_id, "size_of_largest_connected_component": size_of_largest_component, 
            "number_of_nodes": number_of_nodes, "relative_size_of_largest_pathway_component": size_of_largest_component/number_of_nodes, 
            "relative_size_of_number_of_nodes": number_of_nodes/number_of_disease_proteins,
            "num_components": subgraph_num_components, "density": subgraph_density, "clustering": subgraph_clustering,
            "proteins_in_largest_component": proteins}


disease_connectivity = []
disease_id = disease_associations['diseaseId'].unique()[cpu_id]
for seed in tqdm([-1] + list(range(1000)), total=1001, desc=f"Disease {disease_id} permutations"):
    for modality in modality_graphs:
        if modality_disease_counts.loc[(disease_id, disease_id_to_name[disease_id])][modality] < 25:
            print(f"Skipping {disease_id} {modality} due to low protein count")
            continue
        res = get_disease_connectivity(modality, disease_id, seed=seed)
        res['seed'] = seed
        disease_connectivity.append(res)
if len(disease_connectivity) == 0:
    print(f"Skipping disease {disease_id} due to low protein count")
    sys.exit(0)
disease_connectivity_df = pd.DataFrame(disease_connectivity)
disease_connectivity_df['diseaseName'] = disease_connectivity_df['diseaseId'].map(disease_id_to_name)

disease_connectivity_df.to_csv(f"/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/protein_universe/function/disgenet/edge_permutation_test/disease_connectivity_pesto_{pesto_cutoff}_plddt_{plddt_cutoff}_fix_irreps_ensemble_{cpu_id}.csv", index=False)