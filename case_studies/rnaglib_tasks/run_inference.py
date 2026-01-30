
from atomica.models import MultiClassClassifierModel, MultiLabelClassifierModel, ResidueClassifierModel
from atomica.data.dataset import MultiClassLabelledPDBDataset, LabelledPDBDataset
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from glob import glob

sns.set_context("notebook")
custom_colors = ['#E0B8E0', '#C8A0C8', '#B088B0', '#987098', '#00C4C7', '#059094']
xtick_map = {
    'RNAGlib': 'RNAGlib',
    'rinalmo': 'RiNALMo',
    'rnafm': 'RNA-FM',
    'rnaernie': 'RNAErnie',
    'atomica': 'ATOMICA',
    'atomica-ensemble': 'ATOMICA\nEnsemble'
}

DATA_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"
MODEL_DIR="/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_benchmark"
SAVED_MODEL_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks/models/atomica"

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

def get_atomica_results(model_checkpoint, task_name, split="val", threshold=0.5):
    if os.path.exists(os.path.join(os.path.dirname(model_checkpoint), f"{split}_atomica_results.parquet")):
        return pd.read_parquet(os.path.join(os.path.dirname(model_checkpoint), f"{split}_atomica_results.parquet"))
    dataset = MultiClassLabelledPDBDataset(f"{DATA_DIR}/{task_name}/{task_name}_{split}_processed_RNA.parquet")
    model = get_model(model_checkpoint, task_name)

    model.eval()
    model.to("cuda")
    batch_size = 1

    atomica_preds_run = []
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size, desc="Running inference"):
        with torch.no_grad():
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            batch = MultiClassLabelledPDBDataset.collate_fn(batch)
            batch = Trainer.to_device(batch, "cuda")
            atomica_preds_run.append(model.infer(batch).cpu().numpy())
    atomica_preds = np.concatenate(atomica_preds_run)
    
    if task_name == "RNA_Protein":
        atomica_labels = np.concatenate([x['label'] for x in dataset.data])
        atomica_ids = sum([[x['id']] * len(x['label']) for x in dataset.data], [])
        atomica_preds = atomica_preds.flatten()
    elif task_name == "RNA_Site":
        atomica_labels = np.concatenate([x['label'] for x in dataset.data])
        atomica_preds = atomica_preds.flatten()
        atomica_ids = []
        for x in dataset.data:
            assert len(x['label']) == len(x['block_to_pdb_indexes'])
            for _, pdb_index in sorted(x['block_to_pdb_indexes'].items()):
                atomica_ids.append(x['id'] + '_' + str(pdb_index))
    else:
        atomica_labels = np.array([x['label'] for x in dataset.data])
        atomica_ids = np.array([x['id'] for x in dataset.data])
        atomica_labels = list(atomica_labels)
        atomica_preds = list(atomica_preds)
    atomica_results = pd.DataFrame({
        'id': atomica_ids,
        'label': atomica_labels,
        'pred_probability': atomica_preds,
    })
    if task_name == "RNA_Ligand":
        atomica_results['pred'] = atomica_results['pred_probability'].apply(lambda x: np.argmax(x))
    elif task_name == "RNAGo":
        atomica_results['pred'] = atomica_results['pred_probability'].apply(lambda x: (x > threshold).astype(int))
    else:
        atomica_results['pred'] = (atomica_results['pred_probability'] > threshold).astype(int)

    atomica_results.to_parquet(os.path.join(os.path.dirname(model_checkpoint), f"{split}_atomica_results.parquet"))
    return atomica_results

model_checkpoints = [
    f"{SAVED_MODEL_DIR}/RNA_Protein/version_1/epoch14_step3210.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein/version_2/epoch12_step2782.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein/version_3/epoch11_step2568.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein/version_4/epoch14_step3210.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein/version_5/epoch19_step4280.pt",
]

model_checkpoints_no_PRNA = [
    f"{SAVED_MODEL_DIR}/RNA_Protein_no_PRNA_in_pretrain/version_6/epoch15_step3440.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein_no_PRNA_in_pretrain/version_7/epoch9_step2140.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein_no_PRNA_in_pretrain/version_8/epoch11_step2604.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein_no_PRNA_in_pretrain/version_9/epoch13_step2982.pt",
    f"{SAVED_MODEL_DIR}/RNA_Protein_no_PRNA_in_pretrain/version_10/epoch13_step3038.pt",
]

all_metrics = []
ensemble_prob = None
ensemble_prob_val = None
for model_checkpoint in model_checkpoints:
    atomica_results = get_atomica_results(model_checkpoint, "RNA_Protein", split="test")
    atomica_results_val = get_atomica_results(model_checkpoint, "RNA_Protein", split="val")
    if ensemble_prob is None:
        ensemble_prob = atomica_results['pred_probability']
        ensemble_prob_val = atomica_results_val['pred_probability']
    else:
        ensemble_prob += atomica_results['pred_probability']
        ensemble_prob_val += atomica_results_val['pred_probability']
    
    precision, recall, thresholds = precision_recall_curve(atomica_results['label'], atomica_results['pred_probability'])
    auprc = auc(recall, precision)
    metrics_dict = {
        'model': 'atomica',
        'checkpoint_path': model_checkpoint,
        'accuracy': np.mean(atomica_results['label'] == atomica_results['pred']),
        'roc_auc': roc_auc_score(atomica_results['label'], atomica_results['pred_probability']),
        'auprc': auprc,
    }
    all_metrics.append(metrics_dict)

ensemble_prob_no_PRNA = None
for model_checkpoint in model_checkpoints_no_PRNA:
    atomica_results_no_PRNA = get_atomica_results(model_checkpoint, "RNA_Protein", split="test")
    if ensemble_prob_no_PRNA is None:
        ensemble_prob_no_PRNA = atomica_results_no_PRNA['pred_probability']
    else:
        ensemble_prob_no_PRNA += atomica_results_no_PRNA['pred_probability']
    
    precision, recall, thresholds = precision_recall_curve(atomica_results['label'], atomica_results['pred_probability'])
    auprc = auc(recall, precision)
    metrics_dict = {
        'model': 'atomica-no-PRNA',
        'checkpoint_path': model_checkpoint,
        'accuracy': np.mean(atomica_results_no_PRNA['label'] == atomica_results_no_PRNA['pred']),
        'roc_auc': roc_auc_score(atomica_results_no_PRNA['label'], atomica_results_no_PRNA['pred_probability']),
        'auprc': auprc,
    }
    all_metrics.append(metrics_dict)

ensemble_prob = np.stack(ensemble_prob) / len(model_checkpoints)
ensemble_prob_no_PRNA = np.stack(ensemble_prob_no_PRNA) / len(model_checkpoints_no_PRNA)

thresholds = np.linspace(0.0, 1.0, 101)  # e.g. test thresholds from 0.00 to 1.00
f1s = [f1_score(atomica_results_val['label'], (ensemble_prob_val >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
best_f1 = max(f1s)

ensemble_pred = ensemble_prob > best_f1
precision, recall, thresholds = precision_recall_curve(atomica_results['label'], ensemble_prob)
auprc = auc(recall, precision)
metrics_dict = {
    'model': 'atomica-ensemble',
    'checkpoint_path': 'ensemble',
    'accuracy': np.mean(atomica_results['label'] == ensemble_pred),
    'roc_auc': roc_auc_score(atomica_results['label'], ensemble_prob),
    'auprc': auprc,
}
all_metrics.append(metrics_dict)

ensemble_pred_no_PRNA = ensemble_prob_no_PRNA > best_f1
precision, recall, thresholds = precision_recall_curve(atomica_results_no_PRNA['label'], ensemble_prob_no_PRNA)
auprc_no_PRNA = auc(recall, precision)
metrics_dict = {
    'model': 'atomica-ensemble-no-PRNA',
    'checkpoint_path': 'ensemble-no-PRNA',
    'accuracy': np.mean(atomica_results_no_PRNA['label'] == ensemble_pred_no_PRNA),
    'roc_auc': roc_auc_score(atomica_results_no_PRNA['label'], ensemble_prob_no_PRNA),
    'auprc': auprc_no_PRNA,
}
all_metrics.append(metrics_dict)

all_metrics = pd.DataFrame(all_metrics)
atomica_metrics = all_metrics.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'roc_auc': ['mean', 'std'],
    'auprc': ['mean', 'std'],
})

atomica_metrics.loc['atomica-ensemble', ('accuracy', 'std')] = atomica_metrics.loc['atomica', ('accuracy', 'std')] 
atomica_metrics.loc['atomica-ensemble', ('roc_auc', 'std')] = atomica_metrics.loc['atomica', ('roc_auc', 'std')] 
atomica_metrics.loc['atomica-ensemble', ('auprc', 'std')] = atomica_metrics.loc['atomica', ('auprc', 'std')] 



# Ensemble five models with the highest validation AUPRC
model_checkpoints = [
    f"{SAVED_MODEL_DIR}/RNA_Site/version_10/epoch51_step2444.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site/version_28/epoch40_step1927.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site/version_30/epoch67_step3128.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site/version_32/epoch53_step2430.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site/version_33/epoch53_step2592.pt",

    # f"{MODEL_DIR}/RNA_Site/models/version_29/checkpoint/epoch67_step3264.pt",
    # f"{MODEL_DIR}/RNA_Site/models/version_31/checkpoint/epoch73_step3478.pt",
]

model_checkpoints_no_RNAL = [
    f"{SAVED_MODEL_DIR}/RNA_Site_no_RNAL_in_pretrain/version_34/epoch76_step3542.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site_no_RNAL_in_pretrain/version_35/epoch62_step2961.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site_no_RNAL_in_pretrain/version_38/epoch46_step2115.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site_no_RNAL_in_pretrain/version_40/epoch73_step3404.pt",
    f"{SAVED_MODEL_DIR}/RNA_Site_no_RNAL_in_pretrain/version_42/epoch45_step2070.pt",

    # f"{MODEL_DIR}/RNA_Site/models/version_37/checkpoint/epoch47_step2304.pt",
    # f"{MODEL_DIR}/RNA_Site/models/version_39/checkpoint/epoch63_step3072.pt",
    # f"{MODEL_DIR}/RNA_Site/models/version_41/checkpoint/epoch70_step3408.pt",
    # f"{MODEL_DIR}/RNA_Site/models/version_43/checkpoint/epoch43_step2112.pt",
    # f"{MODEL_DIR}/RNA_Site/models/version_44/checkpoint/epoch67_step3128.pt",
    # f"{MODEL_DIR}/RNA_Site/models/version_45/checkpoint/epoch60_step2928.pt",
]

all_metrics = []
ensemble_prob = None
for model_checkpoint in model_checkpoints:
    atomica_results = get_atomica_results(model_checkpoint, "RNA_Site", split="test")
    if ensemble_prob is None:
        ensemble_prob = atomica_results['pred_probability']
    else:
        ensemble_prob += atomica_results['pred_probability']
    
    precision, recall, thresholds = precision_recall_curve(atomica_results['label'], atomica_results['pred_probability'])
    auprc = auc(recall, precision)
    metrics_dict = {
        'model': 'atomica',
        'checkpoint_path': model_checkpoint,
        'accuracy': np.mean(atomica_results['label'] == atomica_results['pred']),
        'roc_auc': roc_auc_score(atomica_results['label'], atomica_results['pred_probability']),
        'auprc': auprc,
    }
    all_metrics.append(metrics_dict)

    atomica_results_val = get_atomica_results(model_checkpoint, "RNA_Site", split="val")
    precision, recall, thresholds = precision_recall_curve(atomica_results_val['label'], atomica_results_val['pred_probability'])
    auprc = auc(recall, precision)
    metrics_dict = {
        'model': 'atomica-val',
        'checkpoint_path': model_checkpoint,
        'accuracy': np.mean(atomica_results_val['label'] == atomica_results_val['pred']),
        'roc_auc': roc_auc_score(atomica_results_val['label'], atomica_results_val['pred_probability']),
        'auprc': auprc,
    }
    all_metrics.append(metrics_dict)

ensemble_prob_no_RNAL = None
for model_checkpoint in model_checkpoints_no_RNAL:
    atomica_results_no_RNAL = get_atomica_results(model_checkpoint, "RNA_Site", split="test")
    if ensemble_prob_no_RNAL is None:
        ensemble_prob_no_RNAL = atomica_results_no_RNAL['pred_probability']
    else:
        ensemble_prob_no_RNAL += atomica_results_no_RNAL['pred_probability']
    
    precision, recall, thresholds = precision_recall_curve(atomica_results_no_RNAL['label'], atomica_results_no_RNAL['pred_probability'])
    auprc = auc(recall, precision)
    metrics_dict = {
        'model': 'atomica-no-RNAL',
        'checkpoint_path': model_checkpoint,
        'accuracy': np.mean(atomica_results_no_RNAL['label'] == atomica_results_no_RNAL['pred']),
        'roc_auc': roc_auc_score(atomica_results_no_RNAL['label'], atomica_results_no_RNAL['pred_probability']),
        'auprc': auprc,
    }
    all_metrics.append(metrics_dict)

    atomica_results_val_no_RNAL = get_atomica_results(model_checkpoint, "RNA_Site", split="val")
    precision, recall, thresholds = precision_recall_curve(atomica_results_val_no_RNAL['label'], atomica_results_val_no_RNAL['pred_probability'])
    auprc = auc(recall, precision)
    metrics_dict = {
        'model': 'atomica-no-RNAL-val',
        'checkpoint_path': model_checkpoint,
        'accuracy': np.mean(atomica_results_val_no_RNAL['label'] == atomica_results_val_no_RNAL['pred']),
        'roc_auc': roc_auc_score(atomica_results_val_no_RNAL['label'], atomica_results_val_no_RNAL['pred_probability']),
        'auprc': auprc,
    }
    all_metrics.append(metrics_dict)


ensemble_prob = np.stack(ensemble_prob) / len(model_checkpoints)
ensemble_prob_no_RNAL = np.stack(ensemble_prob_no_RNAL) / len(model_checkpoints_no_RNAL)

thresholds = np.linspace(0.0, 1.0, 101)  # e.g. test thresholds from 0.00 to 1.00
f1s = [f1_score(atomica_results['label'], (ensemble_prob >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
best_f1 = max(f1s)

ensemble_pred = (ensemble_prob > best_f1).astype(int)
precision, recall, thresholds = precision_recall_curve(atomica_results['label'], ensemble_prob)
auprc = auc(recall, precision)
metrics_dict = {
    'model': 'atomica-ensemble',
    'checkpoint_path': 'ensemble',
    'accuracy': np.mean(atomica_results['label'] == ensemble_pred),
    'roc_auc': roc_auc_score(atomica_results['label'], ensemble_prob),
    'auprc': auprc,
}
all_metrics.append(metrics_dict)

ensemble_pred_no_RNAL = (ensemble_prob_no_RNAL > best_f1).astype(int)
precision, recall, thresholds = precision_recall_curve(atomica_results_no_RNAL['label'], ensemble_prob_no_RNAL)
auprc_no_RNAL = auc(recall, precision)
metrics_dict = {
    'model': 'atomica-ensemble-no-RNAL',
    'checkpoint_path': 'ensemble-no-RNAL',
    'accuracy': np.mean(atomica_results_no_RNAL['label'] == ensemble_pred_no_RNAL),
    'roc_auc': roc_auc_score(atomica_results_no_RNAL['label'], ensemble_prob_no_RNAL),
    'auprc': auprc_no_RNAL,
}
all_metrics.append(metrics_dict)


all_metrics = pd.DataFrame(all_metrics)
atomica_metrics = all_metrics.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'roc_auc': ['mean', 'std'],
    'auprc': ['mean', 'std'],
})

atomica_metrics.loc['atomica-ensemble', ('accuracy', 'std')] = atomica_metrics.loc['atomica', ('accuracy', 'std')] 
atomica_metrics.loc['atomica-ensemble', ('roc_auc', 'std')] = atomica_metrics.loc['atomica', ('roc_auc', 'std')] 
atomica_metrics.loc['atomica-ensemble', ('auprc', 'std')] = atomica_metrics.loc['atomica', ('auprc', 'std')] 
display(atomica_metrics)