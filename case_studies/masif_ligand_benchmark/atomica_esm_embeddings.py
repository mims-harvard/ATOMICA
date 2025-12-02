from atomica.models import MultiClassClassifierModel
from atomica.data.dataset import MultiClassLabelledPDBDataset
from torch.utils.data import DataLoader
from atomica.trainers import Trainer
import torch
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/atomica_ligand/"

import sys
sys.path.append("/n/holylabs/LABS/mzitnik_lab/Users/afang/ATOMICA/baselines/esm2")
from train_mlp import MLPClassifier, ESMEmbeddingDataset

def get_atomica_embeddings(dist_th, model_config, model_weights, split):
    batch_size = 16
    results = []
    dataset = MultiClassLabelledPDBDataset(f"{DATA_DIR}/masif_ligand_pdbs_{dist_th}A_pocket_only_{split}.parquet")
    model = MultiClassClassifierModel.load_from_config_and_weights(model_config, model_weights)
    model.eval()
    model.to("cuda")
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size, desc=f"Processing {dist_th}A"):
        with torch.no_grad():
            batch_items = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            batch = MultiClassLabelledPDBDataset.collate_fn(batch_items)
            batch = Trainer.to_device(batch, "cuda")
            _, output = model.infer(batch, extra_info=True)
            # Use the actual batch size from the output (number of samples in this batch)
            actual_batch_size = output.graph_repr.shape[0]
            for j in range(actual_batch_size):
                results.append({
                    'id': dataset.indexes[i+j],
                    'atomica_embedding': output.graph_repr[j].detach().cpu().numpy(),
                })
    results = pd.DataFrame(results)
    return results


def get_esm_embeddings(dist_th, seed, split, model_type):
    ESM_DIR = f"{DATA_DIR}/{model_type}_embeddings"
    data_path = f"{ESM_DIR}/masif_ligand_pdbs_{dist_th}A_pocket_only_{split}_{model_type}.parquet"
    data_df = pd.read_parquet(data_path)
    dataset = ESMEmbeddingDataset(data_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    protein_embedding_dim = dataset.protein_embeddings.shape[1]
    pocket_embedding_dim = dataset.pocket_embeddings.shape[1]

    model_dir = f"/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/{model_type}/esm{model_type.replace('esm', '')}_mlp_dist{dist_th}_finaldim32/{seed}"
    checkpoint = torch.load(f"{model_dir}/best_model.pt")
    model = MLPClassifier(
        protein_embedding_dim=protein_embedding_dim,
        pocket_embedding_dim=pocket_embedding_dim,
        hidden_dim=512,
        num_classes=dataset.num_classes,
        dropout=0.0,
        pocket_only=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    model.to("cuda")
    all_embeddings = []
    for batch in loader:
        all_embeddings.append(model.get_embedding(batch['protein_embedding'].to("cuda"), batch['pocket_embedding'].to("cuda")).detach().cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    results = pd.DataFrame({
        'id': data_df['id'],
        'esm_embedding': all_embeddings.tolist(),
        'label': data_df['label'],
    })
    return results


model_checkpoints={
    5: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_24/checkpoint/epoch92_step2976.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_25/checkpoint/epoch104_step3360.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_33/checkpoint/epoch127_step4096.pt",
    ],
    6: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_21/checkpoint/epoch147_step6808.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_27/checkpoint/epoch106_step4922.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_34/checkpoint/epoch98_step4554.pt",
    ],
    7: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_22/checkpoint/epoch55_step3528.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_28/checkpoint/epoch197_step12474.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_35/checkpoint/epoch183_step11592.pt",
    ],
    8: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_15/checkpoint/epoch59_step5100.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_29/checkpoint/epoch146_step12495.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_36/checkpoint/epoch68_step5865.pt",
    ],
    9: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_16/checkpoint/epoch47_step5280.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_26/checkpoint/epoch197_step21780.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_37/checkpoint/epoch199_step22200.pt",
    ],
    10: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_19/checkpoint/epoch48_step6664.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_31/checkpoint/epoch187_step25568.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_39/checkpoint/epoch193_step26384.pt",
    ],
    11: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_20/checkpoint/epoch34_step5740.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_30/checkpoint/epoch187_step30644.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_38/checkpoint/epoch81_step13366.pt"
    ],
    12: [
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_14/checkpoint/epoch64_step12545.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_32/checkpoint/epoch140_step27072.pt",
        "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_40/checkpoint/epoch151_step29032.pt",
    ],
}


if __name__ == "__main__":
    for model_type in ['esm2', 'esm3']:
        for dist_th in range(5, 13):
            for seed in range(0, 3):
                for split in ['train', 'val', 'test']:
                    model_weights = model_checkpoints[dist_th][seed]
                    model_config = os.path.join(os.path.dirname(model_weights), "config.json")
                    atomica_embeddings = get_atomica_embeddings(dist_th, model_config, model_weights, split)
                    esm_embeddings = get_esm_embeddings(dist_th, seed, split, model_type)
                    assert np.all(atomica_embeddings.id == esm_embeddings.id)
                    results = pd.concat([atomica_embeddings, esm_embeddings[['esm_embedding', 'label']]], axis=1)
                    os.makedirs(f"{DATA_DIR}/atomica_{model_type}_embeddings", exist_ok=True)
                    results.to_parquet(f"{DATA_DIR}/atomica_{model_type}_embeddings/masif_ligand_pdbs_{dist_th}A_pocket_only_{split}_atomica_{model_type}_seed{seed}.parquet")