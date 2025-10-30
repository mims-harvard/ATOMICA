import glob
import pickle

data_dir = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/pretrain/QBioLiP/sequence_30_split/"
output_dir = "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/pretrain_data/"
modalities = ['PP', 'PDNA', 'PRNA', 'Pion', 'PL', 'Ppeptide', 'RNAL']

for split in ['train', 'test', 'valid']:
    all_data = {}
    for modality in modalities:
        with open(data_dir + f"{modality}_{split}.pkl", "rb") as f:
            data = pickle.load(f)
            all_data[modality] = data

    for leave_out_modality in modalities:
        split_data = []
        for modality in modalities:
            if modality == leave_out_modality:
                continue
            split_data.extend(all_data[modality])
        with open(output_dir + f"QBioLiP_exclude_{leave_out_modality}_{split}.pkl", "wb") as f:
            pickle.dump(split_data, f)

# for split in ['train', 'test', 'valid']:
#     with open(f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/CSD_PS_300_{split}.pkl", "rb") as f:
#         data = pickle.load(f)
#         for item in data:
#             item['modality'] = 0
#     with open(f"/n/holystore01/LABS/mzitnik_lab/Lab/afang/frequency_splits_torsion_09_2024/CSD_PS_300_{split}_with_modality.pkl", "wb") as f:
#         pickle.dump(data, f)