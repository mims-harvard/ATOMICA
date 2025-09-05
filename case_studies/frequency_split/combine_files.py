import glob
import pickle

data_dir = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/sequence_30_split/"
modalities = ['PP', 'PDNA', 'PRNA', 'Pion', 'PL', 'Ppeptide', 'RNAL']

for split in ['train', 'test', 'valid']:
    split_data = []
    for modality in modalities:
        with open(data_dir + f"{modality}_{split}_v2.pkl", "rb") as f:
            data = pickle.load(f)
            for item in data:
                item['modality'] = modalities.index(modality) + 1
            split_data.extend(data)
    with open(data_dir + f"QBioLiP_{split}_v2.pkl", "wb") as f:
        pickle.dump(split_data, f)

for split in ['train', 'test', 'valid']:
    split_data = []
    for modality in modalities:
        if modality in ['PP', 'PL']:
            fname = f"{modality}_{split}_exclude_downstream_v2.pkl"
        else:
            fname = f"{modality}_{split}_v2.pkl"
        with open(data_dir + fname, "rb") as f:
            data = pickle.load(f)
            for item in data:
                item['modality'] = modalities.index(modality) + 1
            split_data.extend(data)
    with open(data_dir + f"QBioLiP_{split}_exclude_downstream_v2.pkl", "wb") as f:
        pickle.dump(split_data, f)

for split in ['train', 'test', 'valid']:
    with open(f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/CSD_PS_300_{split}.pkl", "rb") as f:
        data = pickle.load(f)
        for item in data:
            item['modality'] = 0
    with open(f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/CSD_PS_300_{split}_with_modality.pkl", "wb") as f:
        pickle.dump(data, f)