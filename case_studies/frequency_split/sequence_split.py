import pickle
import pandas as pd
import networkx as nx

def split_PP_with_exclude():
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024-torsion/PP_nonredund_fixed_ids.pkl", 'rb') as f:
        data = pickle.load(f)
    data_ids = [x['id'] for x in data]

    clusters = pd.read_csv("/n/holyscratch01/mzitnik_lab/afang/raw_QBioLiP_06_2024/PPI/clusters_30_v2_cluster.tsv", sep='\t', names=['rep_id', 'seq_id'])
    clusters.set_index('seq_id', inplace=True)

    G = nx.Graph()

    seq_ids = set(clusters.index.values)
    for item in data_ids:
        pdb_id, assembly_id, ch1, ch2 = item.split('_')
        if f'{pdb_id}_{assembly_id}_{ch1}' not in seq_ids:
            print(f"missing {item}")
        else:
            rep_id = clusters.loc[f'{pdb_id}_{assembly_id}_{ch1}']['rep_id']
            G.add_edge(item, rep_id)
        if f'{pdb_id}_{assembly_id}_{ch2}' not in seq_ids:
            print(f"missing {item}")
        else:
            rep_id = clusters.loc[f'{pdb_id}_{assembly_id}_{ch2}']['rep_id']
            G.add_edge(item, rep_id)

    for row in clusters.itertuples():
        G.add_edge(row.Index, row.rep_id)
    

    exclude_pdbs = []
    with open("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/PPA/processed/new_split_30/test.pkl", "rb") as f:
        ppa_data = pickle.load(f)
    ppa_pdbs = [item['id'] for item in ppa_data]
    print(f"PPA test: {len(ppa_pdbs)}")
    exclude_pdbs.extend(ppa_pdbs)

    keep_components = []
    exclude_components = []
    for i, component in enumerate(nx.connected_components(G)):
        component_ids = set(x.split('_')[0].lower() for x in component)
        if component_ids.intersection(exclude_pdbs):
            exclude_components.append(component)
        else:
            keep_components.append(component)
    print("PP: Num components to exclude", len(exclude_components))
    print("PP: Num components to keep", len(keep_components))

    import random
    random.seed(42)
    random.shuffle(keep_components)

    train_components = keep_components[:int(0.9*len(keep_components))]
    valid_components = keep_components[int(0.9*len(keep_components)):]

    set_data_ids = set(data_ids)
    train_data_ids = sum([list(x.intersection(set_data_ids)) for x in train_components], [])
    valid_data_ids = sum([list(x.intersection(set_data_ids)) for x in valid_components], [])
    test_data_ids = sum([list(x.intersection(set_data_ids)) for x in exclude_components], [])

    print("Train: components", len(train_components), "data_ids", len(train_data_ids))
    print("Valid: components", len(valid_components), "data_ids", len(valid_data_ids))
    print("Test/Exclude: components", len(exclude_components), "data_ids", len(test_data_ids))

    train_data_ids = set(train_data_ids)
    valid_data_ids = set(valid_data_ids)
    test_data_ids = set(test_data_ids)

    train_data = [x for x in data if x['id'] in train_data_ids]
    valid_data = [x for x in data if x['id'] in valid_data_ids]
    test_data = [x for x in data if x['id'] in test_data_ids]
    print(f"PP Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PP_train_exclude_downstream_v2.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PP_valid_exclude_downstream_v2.pkl", 'wb') as f:
        pickle.dump(valid_data, f)
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PP_test_exclude_downstream_v2.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"Finished PP exclude downstream: Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")


def split_PP():
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024-torsion/PP_nonredund_fixed_ids.pkl", 'rb') as f:
        data = pickle.load(f)
    data_ids = [x['id'] for x in data]

    clusters = pd.read_csv("/n/holyscratch01/mzitnik_lab/afang/raw_QBioLiP_06_2024/PPI/clusters_30_v2_cluster.tsv", sep='\t', names=['rep_id', 'seq_id'])
    clusters.set_index('seq_id', inplace=True)

    G = nx.Graph()

    seq_ids = set(clusters.index.values)
    for item in data_ids:
        pdb_id, assembly_id, ch1, ch2 = item.split('_')
        if f'{pdb_id}_{assembly_id}_{ch1}' not in seq_ids:
            print(f"missing {item}")
        else:
            rep_id = clusters.loc[f'{pdb_id}_{assembly_id}_{ch1}']['rep_id']
            G.add_edge(item, rep_id)
        if f'{pdb_id}_{assembly_id}_{ch2}' not in seq_ids:
            print(f"missing {item}")
        else:
            rep_id = clusters.loc[f'{pdb_id}_{assembly_id}_{ch2}']['rep_id']
            G.add_edge(item, rep_id)

    for row in clusters.itertuples():
        G.add_edge(row.Index, row.rep_id)
    

    keep_components = []
    for i, component in enumerate(nx.connected_components(G)):
        if component.intersection(seq_ids):
            keep_components.append(component)
    print("PP: Num components to exclude", len(list(nx.connected_components(G)))-len(keep_components))
    print("PP: Num components to keep", len(keep_components))

    import random
    random.seed(42)
    random.shuffle(keep_components)

    train_components = keep_components[:int(0.8*len(keep_components))]
    valid_components = keep_components[int(0.8*len(keep_components)):int(0.9*len(keep_components))]
    test_components = keep_components[int(0.9*len(keep_components)):]

    set_data_ids = set(data_ids)
    train_data_ids = sum([list(x.intersection(set_data_ids)) for x in train_components], [])
    valid_data_ids = sum([list(x.intersection(set_data_ids)) for x in valid_components], [])
    test_data_ids = sum([list(x.intersection(set_data_ids)) for x in test_components], [])

    print("Train: components", len(train_components), "data_ids", len(train_data_ids))
    print("Valid: components", len(valid_components), "data_ids", len(valid_data_ids))
    print("Test/Exclude: components", len(test_components), "data_ids", len(test_data_ids))

    train_data_ids = set(train_data_ids)
    valid_data_ids = set(valid_data_ids)
    test_data_ids = set(test_data_ids)

    train_data = [x for x in data if x['id'] in train_data_ids]
    valid_data = [x for x in data if x['id'] in valid_data_ids]
    test_data = [x for x in data if x['id'] in test_data_ids]
    print(f"PP Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PP_train_v2.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PP_valid_v2.pkl", 'wb') as f:
        pickle.dump(valid_data, f)
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PP_test_v2.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"Finished PP: Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")


def split_PL_with_exclude():
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024-torsion/PL_nonredund.pkl", 'rb') as f:
        data = pickle.load(f)
    data_ids = [x['id'] for x in data]

    clusters = pd.read_csv("/n/holyscratch01/mzitnik_lab/afang/raw_QBioLiP_06_2024/PL/clusters_30_v2_cluster.tsv", sep='\t', names=['rep_id', 'seq_id'])
    clusters.set_index('seq_id', inplace=True)

    G = nx.Graph()

    seq_ids = set(clusters.index.values)
    for item in data_ids:
        item_prefix = item.split('.pdb')[0]

        matching_seq_ids = [x for x in seq_ids if x.startswith(item_prefix)]

        if len(matching_seq_ids) == 0:
            print(f"missing {item}")
        else:
            for seq_id in matching_seq_ids:
                rep_id = clusters.loc[seq_id]['rep_id']
                G.add_edge(item, rep_id)

    for row in clusters.itertuples():
        G.add_edge(row.Index, row.rep_id)

    connected_components = nx.connected_components(G)
    print(len(list(connected_components)), len(data_ids), len(clusters['rep_id'].unique()))

    exclude_pdbs = []

    pdbbind_dir = "/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/datasets/PDBBind/processed_PS_300/identity30/"
    for split in ["test"]: #"train", "valid", 
        with open(f"{pdbbind_dir}/{split}.pkl", "rb") as f:
            exclude_data = pickle.load(f)
        pdbs = [item['id'] for item in exclude_data]
        print(f"PL30: {len(pdbs)}")
        exclude_pdbs.extend(pdbs)

    keep_components = []
    exclude_components = []
    for i, component in enumerate(nx.connected_components(G)):
        component_ids = set(x.split('_')[0].lower() for x in component)
        if component_ids.intersection(exclude_pdbs):
            exclude_components.append(component)
        else:
            keep_components.append(component)
    print("Num components to exclude", len(exclude_components))
    print("Num components to keep", len(keep_components))

    import random
    random.seed(42)
    random.shuffle(keep_components)

    train_components = keep_components[:int(0.9*len(keep_components))]
    valid_components = keep_components[int(0.9*len(keep_components)):]

    set_data_ids = set(data_ids)
    train_data_ids = sum([list(x.intersection(set_data_ids)) for x in train_components], [])
    valid_data_ids = sum([list(x.intersection(set_data_ids)) for x in valid_components], [])
    test_data_ids = sum([list(x.intersection(set_data_ids)) for x in exclude_components], [])

    print("Train: components", len(train_components), "data_ids", len(train_data_ids))
    print("Valid: components", len(valid_components), "data_ids", len(valid_data_ids))
    print("Test/Exclude: components", len(exclude_components), "data_ids", len(test_data_ids))

    train_data_ids = set(train_data_ids)
    valid_data_ids = set(valid_data_ids)
    test_data_ids = set(test_data_ids)

    train_data = [x for x in data if x['id'] in train_data_ids]
    valid_data = [x for x in data if x['id'] in valid_data_ids]
    test_data = [x for x in data if x['id'] in test_data_ids]
    print(f"PL Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PL_train_exclude_downstream_v2.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PL_valid_exclude_downstream_v2.pkl", 'wb') as f:
        pickle.dump(valid_data, f)
    with open("/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/PL_test_exclude_downstream_v2.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"Finished PL exclude downstream: Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    


def split_modality(modality_data, modality_raw):
    data_path = f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024-torsion/{modality_data}_nonredund.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data_ids = [x['id'] for x in data]

    clusters = pd.read_csv(f"/n/holyscratch01/mzitnik_lab/afang/raw_QBioLiP_06_2024/{modality_raw}/clusters_30_v2_cluster.tsv", sep='\t', names=['rep_id', 'seq_id'])
    clusters.set_index('seq_id', inplace=True)

    G = nx.Graph()

    seq_ids = set(clusters.index.values)
    for item in data_ids:
        item_prefix = item.split('.pdb')[0]

        matching_seq_ids = [x for x in seq_ids if x.startswith(item_prefix)]

        if len(matching_seq_ids) == 0:
            print(f"missing {item}")
        else:
            for seq_id in matching_seq_ids:
                rep_id = clusters.loc[seq_id]['rep_id']
                G.add_edge(item, rep_id)

    for row in clusters.itertuples():
        G.add_edge(row.Index, row.rep_id)

    keep_components = list(nx.connected_components(G))
    print("Num components to keep", len(keep_components))

    import random
    random.seed(135)
    random.shuffle(keep_components)

    train_components = keep_components[:int(0.8*len(keep_components))]
    valid_components = keep_components[int(0.8*len(keep_components)):int(0.9*len(keep_components))]
    test_components = keep_components[int(0.9*len(keep_components)):]

    set_data_ids = set(data_ids)
    train_data_ids = sum([list(x.intersection(set_data_ids)) for x in train_components], [])
    valid_data_ids = sum([list(x.intersection(set_data_ids)) for x in valid_components], [])
    test_data_ids = sum([list(x.intersection(set_data_ids)) for x in test_components], [])

    print("Train: components", len(train_components), "data_ids", len(train_data_ids))
    print("Valid: components", len(valid_components), "data_ids", len(valid_data_ids))
    print("Test: components", len(test_components), "data_ids", len(test_data_ids))

    train_data_ids = set(train_data_ids)
    valid_data_ids = set(valid_data_ids)
    test_data_ids = set(test_data_ids)

    train_data = [x for x in data if x['id'] in train_data_ids]
    valid_data = [x for x in data if x['id'] in valid_data_ids]
    test_data = [x for x in data if x['id'] in test_data_ids]
    print(f"{modality_data} Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    with open(f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/{modality_data}_train_v2.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open(f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/{modality_data}_valid_v2.pkl", 'wb') as f:
        pickle.dump(valid_data, f)
    with open(f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/{modality_data}_test_v2.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"Finished {modality_data}: Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    split_PP()
    split_PP_with_exclude()
    split_PL_with_exclude()
    
    modality_dict = {
        "RNAL": "RNAL",
        "PIII": "Ppeptide",
        "Pion": "Pion",
        "PDNA": "PDNA",
        "PRNA": "PRNA",
        "PL": "PL",
    }

    for modality_raw, modality_data in modality_dict.items():
        split_modality(modality_data, modality_raw)