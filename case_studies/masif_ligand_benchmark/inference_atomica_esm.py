import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import re
from collections import Counter

from atomica.models import MultiClassClassifierModel
from atomica.data.dataset import MultiClassLabelledPDBDataset, PocketEmbeddingDatasetWrapper
from torch.utils.data import DataLoader
from atomica.trainers import Trainer
import torch


def get_config_and_weights_path(ckpt_path):
    weights_path = ckpt_path.replace('.ckpt', '.pt')
    config_path = os.path.dirname(ckpt_path) + "/config.json"
    return config_path, weights_path

def get_model(model_config: str, model_checkpoint: str) -> str:
    model = MultiClassClassifierModel.load_from_config_and_weights(model_config, model_checkpoint)
    return model

def run_model(ckpt_path, split, dist_th, embeddings_file):
    config_path, weights_path = get_config_and_weights_path(ckpt_path)

    # Load base dataset
    base_dataset = MultiClassLabelledPDBDataset(f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/atomica_ligand/masif_ligand_pdbs_{dist_th}A_pocket_only_{split}.parquet")

    # Wrap with pocket embeddings
    dataset = PocketEmbeddingDatasetWrapper(
        base_dataset=base_dataset,
        embeddings_file=embeddings_file
    )

    model = get_model(config_path, weights_path)
    atomica_preds = []
    model.eval()
    model.to("cuda")
    batch_size = 16
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size, desc=f"Processing {dist_th}A"):
        with torch.no_grad():
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            # Use the wrapped dataset's collate_fn to include pocket embeddings
            batch = dataset.collate_fn(batch)
            batch = Trainer.to_device(batch, "cuda")
            atomica_preds.append(model.infer(batch).cpu().numpy())
    atomica_preds = np.concatenate(atomica_preds)
    atomica_labels = np.array([x['label'] for x in base_dataset.data])
    pred_indxes = np.argmax(atomica_preds, axis=1)
    df = pd.DataFrame({
        'dist_th': dist_th,
        'id': [x['id'] for x in base_dataset.data],
        'label': atomica_labels,
        'pred': pred_indxes,
        'pred_probability': [atomica_preds[i] for i in range(len(atomica_preds))],
        'ckpt_path': ckpt_path,
    })
    df.to_parquet(f"{os.path.dirname(ckpt_path)}/{split}_preds.parquet", index=False)
    return df


if __name__ == "__main__":
    # # Model checkpoints for dist_th=8
    # model_ckpts = ['/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_263/checkpoint/epoch69_step5950.ckpt',
        #  '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_262/checkpoint/epoch32_step2805.ckpt',
        #  '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_264/checkpoint/epoch49_step4250.ckpt',
        #  '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_283/checkpoint/epoch43_step3740.ckpt',
        #  '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_277/checkpoint/epoch23_step2040.ckpt']

    DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/atomica_ligand/"
    test_embeddings_file = f"{DATA_DIR}/esm2_embeddings/masif_ligand_pdbs_8A_pocket_only_test_esm2.npy"
    for seed in range(0,5):
        ckpt_path = f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/MASIF/models/atomica_esm2/late_fusion/seed{seed}/model.ckpt"
        run_model(ckpt_path, "test", dist_th=8, embeddings_file=test_embeddings_file)
