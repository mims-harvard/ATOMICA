from atomica.models import (
    MultiClassClassifierModel,
    MultiLabelClassifierModel,
    ClassifierModel,
    ResidueClassifierModel,
)
from atomica.trainers import ClassifierTrainer, MultiClassClassifierTrainer
from atomica.data.dataset import LabelledPDBDataset, MultiClassLabelledPDBDataset

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

from rnafm_models import multiclass_classifier

DATA_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"
MODEL_DIR="/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_benchmark"
RNAFM_MODEL_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks/models_final_hidden_dim_32_no_dropout/"

class RNAGlibTask:
    def __init__(self, 
        task_name: str, 
        residue_level: bool, 
        task_type: str, 
        atomica_model_weights: str, 
        rnafm_model_checkpoint: str, 
        rnafm_name: str,
        split: str,
    ):
        self.task_name = task_name
        self.residue_level = residue_level
        self.task_type = task_type
        self.atomica_model_weights = f'{MODEL_DIR}/{task_name}/models/{atomica_model_weights}'
        self.atomica_model_config = os.path.join(os.path.dirname(self.atomica_model_weights), "config.json")
        if rnafm_model_checkpoint.startswith("version_"):
            self.rnafm_model_checkpoint = f'{RNAFM_MODEL_DIR}/{rnafm_name}/{task_name}/{rnafm_model_checkpoint}/best_model.pt'
        else:
            self.rnafm_model_checkpoint = rnafm_model_checkpoint
        self.rnafm_name = rnafm_name
        self.split = split

    def get_rnafm_embeddings_path(self) -> str:
        return f'{DATA_DIR}/{self.task_name}/{self.task_name}_{self.split}_embeddings_{self.rnafm_name}.npy'
    
    def get_rnafm_labels_path(self) -> str:
        return f'{DATA_DIR}/{self.task_name}/{self.task_name}_{self.split}_labels_{self.rnafm_name}.npy'

    def get_rnafm_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(f"{DATA_DIR}/{self.task_name}/{self.task_name}_{self.split}_input.parquet")

    def get_atomica_dataset_path(self) -> str:
        return f"{DATA_DIR}/{self.task_name}/{self.task_name}_{self.split}_processed.parquet"


# RNA_PROTEIN = RNAGlibTask(
#     task_name="RNA_Protein",
#     residue_level=True,
#     task_type="binary",
#     atomica_model_weights="",
#     rnafm_model_checkpoint="",
#     rnafm_name="rnafm",
# )

RNA_SITE = RNAGlibTask(
    task_name="RNA_Site",
    residue_level=True,
    task_type="binary",
    atomica_model_weights="version_10/checkpoint/epoch51_step2444.pt",
    rnafm_model_checkpoint="version_2",
    rnafm_name="rinalmo",
    split="test",
)

RNA_LIGAND = RNAGlibTask(
    task_name="RNA_Ligand",
    residue_level=False,
    task_type="multiclass",
    atomica_model_weights="version_34/checkpoint/epoch205_step9064.pt",
    rnafm_model_checkpoint="version_1",
    rnafm_name="rnafm",
    split="test",
)

RNA_GO = RNAGlibTask(
    task_name="RNAGo",
    residue_level=False,
    task_type="multilabel",
    atomica_model_weights="version_28/checkpoint/epoch362_step55539.pt",
    rnafm_model_checkpoint="/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks/rinalmo/RNAGo/version_lr_1e-3_dropout_0.4_hidden_dim_256/seed_0/best_model.pt",
    rnafm_name="rinalmo",
    split="test",
)


def get_model_trainer_dataset(task: RNAGlibTask):
    if task.task_type == "multiclass":
        # maps to task='multiclass_classifier'
        return MultiClassClassifierModel, MultiClassClassifierTrainer, MultiClassLabelledPDBDataset
    elif task.task_type == "binary":
        # residue-level vs graph-level binary
        if task.residue_level:
            # maps to task='residue_binary_classifier'
            return ResidueClassifierModel, ClassifierTrainer, LabelledPDBDataset
        else:
            # maps to task='binary_classifier'
            return ClassifierModel, ClassifierTrainer, LabelledPDBDataset
    elif task.task_type == "multilabel":
        # maps to task='multilabel_classifier'
        return MultiLabelClassifierModel, MultiClassClassifierTrainer, MultiClassLabelledPDBDataset
    else:
        raise ValueError(f"Unknown task_type: {task.task_type}")

def get_atomica_embeddings(task: RNAGlibTask):
    batch_size = 2
    results = []
    Model, Trainer, Dataset = get_model_trainer_dataset(task)
    dataset = Dataset(task.get_atomica_dataset_path())
    model = Model.load_from_config_and_weights(task.atomica_model_config, task.atomica_model_weights)
    model.eval()
    model.to("cuda")
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size, desc=f"Processing {task.task_name}"):
        with torch.no_grad():
            batch_items = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            batch = Dataset.collate_fn(batch_items)
            batch = Trainer.to_device(batch, "cuda")
            _, output = model.infer(batch, extra_info=True)
            # Use the actual batch size from the output (number of samples in this batch)
            actual_batch_size = output.graph_repr.shape[0]
            if not task.residue_level:
                for j in range(actual_batch_size):
                    results.append({
                        'id': dataset.indexes[i+j],
                        'atomica_embedding': output.graph_repr[j].detach().cpu().numpy(),
                    })
            elif task.residue_level:
                curr_block_start = 0
                for j in range(actual_batch_size):
                    for block_idx, block_id in dataset.data[i+j]['block_to_pdb_indexes'].items():
                        results.append({
                            'id': f'{dataset.indexes[i+j]}_{block_id}',
                            'atomica_embedding': output.block_repr[curr_block_start+block_idx].detach().cpu().numpy(),
                        })
                curr_block_start += len(dataset.data[i+j]['data']['B'])
    results = pd.DataFrame(results)
    return results


def get_rnafm_embeddings(task: RNAGlibTask) -> pd.DataFrame:
    """
    Get embeddings from the penultimate layer of the RNAFM MLP classifier
    defined in `multiclass_classifier.py`.
    
    Parameters
    ----------
    task : RNAGlibTask
        Task object containing all necessary paths and information.
    """
    # Load data as in `multiclass_classifier.py`
    embeddings = np.load(task.get_rnafm_embeddings_path())
    labels = np.load(task.get_rnafm_labels_path())
    input_dim = embeddings.shape[1]
    
    if task.task_type == "multiclass":
        num_classes = int(np.max(labels) + 1)
    elif task.task_type == "binary":
        num_classes = 1
    else:  # multilabel
        num_classes = labels.shape[1]
    
    # Load classifier checkpoint
    checkpoint = torch.load(task.rnafm_model_checkpoint, map_location="cpu")
    model_cfg = checkpoint.get("model_config", {})
    
    # Prefer stored config if available, otherwise fall back to inferred dims
    input_dim_ckpt = model_cfg.get("input_dim", input_dim)
    num_classes_ckpt = model_cfg.get("num_classes", num_classes)
    task_type_ckpt = model_cfg.get("task_type", task.task_type)
    hidden_dim = model_cfg.get("hidden_dim", 512)
    dropout = model_cfg.get("dropout", 0.0)
    
    model = multiclass_classifier.MLPClassifier(
        input_dim=input_dim_ckpt,
        num_classes=num_classes_ckpt,
        task_type=task_type_ckpt,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cuda")
    model.eval()
    
    # Simple dataset over raw embeddings
    class EmbeddingOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, emb, lbl):
            self.emb = torch.FloatTensor(emb)
            self.lbl = torch.from_numpy(lbl)
        
        def __len__(self):
            return len(self.emb)
        
        def __getitem__(self, idx):
            return self.emb[idx], self.lbl[idx]
    
    dataset = EmbeddingOnlyDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch_emb, batch_lbl in loader:
            batch_emb = batch_emb.to("cuda")
            emb = model.get_embedding(batch_emb)
            all_embeddings.append(emb.detach().cpu().numpy())
            all_labels.append(batch_lbl.numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    rnafm_dataset = task.get_rnafm_dataset()
    if task.residue_level:
        item_ids = []
        for x in rnafm_dataset.itertuples():
            for i in x.pdb_indexes:
                item_ids.append(x.pdb_id + '_' + i)
    else:
        item_ids = rnafm_dataset['pdb_id'].tolist()
    
    results = pd.DataFrame({
        'id': item_ids,
        'sequence_embedding': all_embeddings.tolist(),
        'label': all_labels.tolist(),
    })
    return results



if __name__ == "__main__":
    for rnaglib_task in [RNA_GO]: # RNA_PROTEIN,  RNA_SITE, RNA_LIGAND, 
        for split in ['train', 'val', 'test']:
            rnaglib_task.split = split
            atomica_embeddings = get_atomica_embeddings(rnaglib_task)
            rnafm_embeddings = get_rnafm_embeddings(rnaglib_task)
            if not len(atomica_embeddings.id) == len(rnafm_embeddings.id) or not np.all(atomica_embeddings.id == rnafm_embeddings.id):
                mismatched_ids = set(atomica_embeddings.id) ^ set(rnafm_embeddings.id)
                print(f"Mismatched IDs: {mismatched_ids}")
            results = pd.merge(atomica_embeddings, rnafm_embeddings, on='id', how='left')
            output_dir = f"{DATA_DIR}/atomica_ensemble_embeddings/{rnaglib_task.task_name}"
            os.makedirs(output_dir, exist_ok=True)
            results.to_parquet(f"{output_dir}/atomica_{rnaglib_task.rnafm_name}_{rnaglib_task.task_name}_{split}_embeddings_v2.parquet")