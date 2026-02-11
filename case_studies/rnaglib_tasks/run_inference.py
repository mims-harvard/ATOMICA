from atomica.models import MultiClassClassifierModel, MultiLabelClassifierModel, ResidueClassifierModel
from atomica.data.dataset import MultiClassLabelledPDBDataset, LabelledPDBDataset, PocketEmbeddingDatasetWrapper
from atomica.trainers import Trainer

from multiclass_metrics import compute_multiclass_metrics
from multilabel_metrics import compute_multilabel_metrics

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from glob import glob

MODEL_DIR = "/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_benchmark/"
DATA_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"

def get_model(model_checkpoint: str, task_name: str) -> str:
    model_config = os.path.join(os.path.dirname(model_checkpoint), "config.json")
    if task_name == "RNAGo":
        return MultiLabelClassifierModel.load_from_config_and_weights(model_config, model_checkpoint)
    elif task_name == "RNA_Ligand":
        return MultiClassClassifierModel.load_from_config_and_weights(model_config, model_checkpoint)
    elif task_name == "RNA_Protein" or task_name == "RNA_Site":
        return ResidueClassifierModel.load_from_config_and_weights(model_config, model_checkpoint)
    else:
        raise ValueError(f"Task name {task_name} not supported")

def predict_with_thresholds(proba, class_thresholds):
    """Helper function to predict using class-specific thresholds."""
    proba = np.array(proba)
    # Check which classes exceed their thresholds
    above_threshold = np.array([proba[i] >= class_thresholds.get(i, 0.0) for i in range(len(proba))])
    
    if np.sum(above_threshold) == 1:
        # Exactly one class exceeds threshold
        return np.argmax(above_threshold)
    elif np.sum(above_threshold) > 1:
        # Multiple classes exceed thresholds, pick the one with highest probability
        candidates = np.where(above_threshold)[0]
        return candidates[np.argmax(proba[candidates])]
    else:
        # No class exceeds threshold, fallback to argmax
        return np.argmax(proba)

def get_atomica_results(model_checkpoint, task_name, split="val", threshold=0.5, class_thresholds=None, recompute=False, pocket_embeddings_file=None):
    if not recompute and os.path.exists(os.path.join(os.path.dirname(model_checkpoint), f"{split}_atomica_results.parquet")):
        df = pd.read_parquet(os.path.join(os.path.dirname(model_checkpoint), f"{split}_atomica_results.parquet"))
        if class_thresholds is not None and task_name == "RNA_Ligand":
            # Convert class_thresholds to dict if it's a list
            if isinstance(class_thresholds, (list, np.ndarray)):
                class_thresholds = {i: float(t) for i, t in enumerate(class_thresholds)}
            
            df['pred'] = df['pred_probability'].apply(
                lambda x: predict_with_thresholds(x, class_thresholds)
            )
        if 'ckpt' in df.columns and df['ckpt'].iloc[0] == model_checkpoint:
            return df
    
    dataset = MultiClassLabelledPDBDataset(f"{DATA_DIR}/{task_name}/{task_name}_{split}_processed_RNA.parquet")
    model = get_model(model_checkpoint, task_name)

    # Validate pocket embeddings configuration
    is_residue_task = task_name in ["RNA_Protein", "RNA_Site"]
    model_expects_pocket_emb = hasattr(model, 'pocket_embedding_size') and model.pocket_embedding_size is not None
    
    if is_residue_task and pocket_embeddings_file is not None:
        raise ValueError(f"Task {task_name} is a residue-level task and does not support pocket embeddings")
    
    if model_expects_pocket_emb and pocket_embeddings_file is None:
        raise ValueError(
            f"Model was trained with pocket embeddings (embedding_size={model.pocket_embedding_size}) "
            f"but no pocket_embeddings_file was provided. Please provide the embeddings file."
        )
    
    if not model_expects_pocket_emb and pocket_embeddings_file is not None:
        print(f"Warning: Model was trained without pocket embeddings but pocket_embeddings_file was provided. "
              f"Ignoring pocket embeddings file.")
        pocket_embeddings_file = None
    
    # Wrap dataset with pocket embeddings if needed
    if pocket_embeddings_file is not None:
        print(f"Loading pocket embeddings from {pocket_embeddings_file}")
        dataset = PocketEmbeddingDatasetWrapper(dataset, pocket_embeddings_file)

    model.eval()
    model.to("cuda")
    batch_size = 1

    atomica_preds_run = []
    # Use the dataset's collate_fn to ensure pocket embeddings are properly handled
    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else MultiClassLabelledPDBDataset.collate_fn
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size, desc="Running inference"):
        with torch.no_grad():
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            batch = collate_fn(batch)
            batch = Trainer.to_device(batch, "cuda")
            atomica_preds_run.append(model.infer(batch).cpu().numpy())
    atomica_preds = np.concatenate(atomica_preds_run)

    # Get the underlying dataset data (handle wrapped datasets)
    if hasattr(dataset, 'base_dataset'):
        dataset_data = dataset.base_dataset.data
    else:
        dataset_data = dataset.data

    if task_name == "RNA_Protein":
        atomica_labels = np.concatenate([x['label'] for x in dataset_data])
        atomica_ids = sum([[x['id']] * len(x['label']) for x in dataset_data], [])
        atomica_preds = atomica_preds.flatten()
    elif task_name == "RNA_Site":
        atomica_labels = np.concatenate([x['label'] for x in dataset_data])
        atomica_preds = atomica_preds.flatten()
        atomica_ids = []
        for x in dataset_data:
            assert len(x['label']) == len(x['block_to_pdb_indexes'])
            for _, pdb_index in sorted(x['block_to_pdb_indexes'].items()):
                atomica_ids.append(x['id'] + '_' + str(pdb_index))
    else:
        atomica_labels = np.array([x['label'] for x in dataset_data])
        atomica_ids = np.array([x['id'] for x in dataset_data])
        atomica_labels = list(atomica_labels)
        atomica_preds = list(atomica_preds)
    atomica_results = pd.DataFrame({
        'id': atomica_ids,
        'label': atomica_labels,
        'pred_probability': atomica_preds,
    })
    if task_name == "RNA_Ligand":
        if class_thresholds is not None:
            # Convert class_thresholds to dict if it's a list
            if isinstance(class_thresholds, (list, np.ndarray)):
                class_thresholds = {i: float(t) for i, t in enumerate(class_thresholds)}
            
            atomica_results['pred'] = atomica_results['pred_probability'].apply(
                lambda x: predict_with_thresholds(x, class_thresholds)
            )
        else:
            atomica_results['pred'] = atomica_results['pred_probability'].apply(lambda x: np.argmax(x))
    elif task_name == "RNAGo":
        atomica_results['pred'] = atomica_results['pred_probability'].apply(lambda x: (x > threshold).astype(int))
    else:
        atomica_results['pred'] = (atomica_results['pred_probability'] > threshold).astype(int)

    atomica_results['ckpt'] = model_checkpoint
    atomica_results.to_parquet(os.path.join(os.path.dirname(model_checkpoint), f"{split}_atomica_results.parquet"))
    return atomica_results


SAVED_CKPT_DIR = "/n/holylabs/LABS/mzitnik_lab/Users/afang/ATOMICA/checkpoints/benchmarks"
ckpt_dirs = [
    # ("rna_go", "RNAGo"),
    # ("rna_ligand/atomica", "RNA_Ligand"),
    # ("rna_ligand/atomica_rnafm", "RNA_Ligand"),
    # ("rna_protein/atomica", "RNA_Protein"),
    # ("rna_protein/atomica_no_PRNA_in_pretrain", "RNA_Protein"),
    # ("rna_site/atomica", "RNA_Site"),
    ("rna_site/atomica_no_RNAL_in_pretrain", "RNA_Site"),
    ("rna_ligand/atomica_no_RNAL_in_pretrain", "RNA_Ligand"),
]

for ckpt_dir, task_name in ckpt_dirs:
    ckpt_dir = os.path.join(SAVED_CKPT_DIR, ckpt_dir)
    for seed in range(5):
        ckpt = os.path.join(ckpt_dir, f"seed{seed}/model.pt")
        for split in ["val", "test"]:
            if "rnafm" in ckpt_dir:
                pocket_embeddings_file = f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks/RNA_Ligand/RNA_Ligand_{split}_embeddings_rnafm.npy"
            else:
                pocket_embeddings_file = None
            results = get_atomica_results(ckpt, task_name, split=split, pocket_embeddings_file=pocket_embeddings_file)