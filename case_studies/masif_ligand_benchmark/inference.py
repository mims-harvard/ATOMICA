import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import re
from collections import Counter

from atomica.models import MultiClassClassifierModel
from atomica.data.dataset import MultiClassLabelledPDBDataset
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

def run_model(ckpt_path, split, dist_th):
    config_path, weights_path = get_config_and_weights_path(ckpt_path)
    dataset = MultiClassLabelledPDBDataset(f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/atomica_ligand/masif_ligand_pdbs_{dist_th}A_pocket_only_{split}.parquet")
    model = get_model(config_path, weights_path)
    atomica_preds = []
    model.eval()
    model.to("cuda")
    batch_size = 16
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size, desc=f"Processing {dist_th}A"):
        with torch.no_grad():
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            batch = MultiClassLabelledPDBDataset.collate_fn(batch)
            batch = Trainer.to_device(batch, "cuda")
            atomica_preds.append(model.infer(batch).cpu().numpy())
    atomica_preds = np.concatenate(atomica_preds)
    atomica_labels = np.array([x['label'] for x in dataset.data])
    pred_indxes = np.argmax(atomica_preds, axis=1)
    df = pd.DataFrame({
        'dist_th': dist_th,
        'id': [x['id'] for x in dataset.data],
        'label': atomica_labels,
        'pred': pred_indxes,
        'pred_probability': [atomica_preds[i] for i in range(len(atomica_preds))],
        'ckpt_path': ckpt_path,
    })
    df.to_parquet(f"{os.path.dirname(ckpt_path)}/{split}_preds.parquet", index=False)
    return df


if __name__ == "__main__":
    model_ckpts = {
        8: ['/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_142/checkpoint/epoch186_step15895.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_206/checkpoint/epoch138_step11815.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_174/checkpoint/epoch140_step11985.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_196/checkpoint/epoch253_step21590.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_150/checkpoint/epoch286_step24395.ckpt',
            ],
        7: ['/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_204/checkpoint/epoch299_step18900.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_143/checkpoint/epoch236_step14931.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_181/checkpoint/epoch275_step17388.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_162/checkpoint/epoch284_step17955.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_190/checkpoint/epoch196_step12411.ckpt',],

        6: ['/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_205/checkpoint/epoch222_step10258.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_180/checkpoint/epoch176_step8142.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_198/checkpoint/epoch273_step12604.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_149/checkpoint/epoch185_step8556.ckpt',
            '/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/masif_benchmark/models/version_188/checkpoint/epoch223_step10304.ckpt',
            ],
    }

    for dist_th, ckpt_paths in model_ckpts.items():
        for ckpt_path in ckpt_paths:
            run_model(ckpt_path, "train", dist_th)