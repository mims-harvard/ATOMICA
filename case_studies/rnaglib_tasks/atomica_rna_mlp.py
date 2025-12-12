#!/usr/bin/env python3
"""
Training script for Atomica + RNA-model (e.g. RNAFM/RINALMO) embeddings using an MLP.

This is analogous to `atomica_esm_mlp.py`, but tailored to the RNAGlib tasks and
their three different setups (binary, multiclass, multilabel) and parquet datasets
produced by `atomica_rna_embeddings.py`.

Expected parquet schema (per split: train / val / test):
- 'id'                  : identifier string
- 'atomica_embedding'   : 1D array-like (list, np.ndarray, or Tensor)
- 'sequence_embedding'  : 1D array-like (list, np.ndarray, or Tensor) from RNA model
- 'label'               : 
    * binary     : scalar 0/1
    * multiclass : scalar int in [0, num_classes)
    * multilabel : 1D array-like of {0,1} with length = num_classes
"""

import os
import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import average_precision_score

from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")

import wandb

# Import metrics modules (same as multiclass_classifier.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiclass_metrics import compute_multiclass_metrics, MetricsResult as MulticlassMetricsResult
from multilabel_metrics import compute_multilabel_metrics, MultilabelMetricsResult

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from dataclasses import dataclass, asdict


@dataclass
class BinaryMetricsResult:
    """Metrics result for binary classification."""
    # Global metrics
    accuracy: float
    balanced_accuracy: float
    
    # ROC AUC
    roc_auc: float
    auprc: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> BinaryMetricsResult:
    """
    Compute binary classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        True binary labels (0 or 1)
    y_pred : np.ndarray of shape (N,)
        Predicted binary labels (0 or 1)
    y_proba : np.ndarray of shape (N, 2), optional
        Class probabilities for both classes [prob_class0, prob_class1]
        If not provided, ROC AUC will be None
    
    Returns
    -------
    BinaryMetricsResult
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Ensure 1D arrays
    if y_true.ndim > 1:
        y_true = y_true.squeeze()
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays for binary classification")
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    
    # Ensure binary labels
    y_true = (y_true > 0).astype(int)
    y_pred = (y_pred > 0).astype(int)
    
    # Basic metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    
    # ROC AUC and AUPRC
    roc_auc = None
    auprc = None
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        # If probabilities are shape (N, 2), use class 1 probabilities
        # If shape (N, 1), assume it's class 1 probability
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            prob_class1 = y_proba[:, 1]
        elif y_proba.ndim == 2 and y_proba.shape[1] == 1:
            prob_class1 = y_proba.squeeze()
        elif y_proba.ndim == 1:
            prob_class1 = y_proba
        else:
            raise ValueError(f"y_proba must have shape (N,) or (N, 1) or (N, 2), got {y_proba.shape}")
        
        # ROC AUC requires both classes to be present
        roc_auc = float(roc_auc_score(y_true, prob_class1))
        # AUPRC
        precision, recall, thresholds = precision_recall_curve(y_true, prob_class1)
        auprc = auc(recall, precision)
    
    return BinaryMetricsResult(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        roc_auc=roc_auc,
        auprc=auprc,
    )


# Import RNAGlib task configs to reuse DATA_DIR and rnafm_name
from atomica_rna_embeddings import (
    RNA_SITE,
    RNA_LIGAND,
    RNA_GO,
    DATA_DIR as RNAGLIB_DATA_DIR,
)


DATA_DIR = RNAGLIB_DATA_DIR


class TaskConfig:
    """
    Configuration for a single RNAGlib task.

    This mirrors the task types defined in `atomica_rna_embeddings.py`:
    - RNA_Site   : binary, residue-level
    - RNA_Ligand : multiclass, graph-level
    - RNAGo      : multilabel, graph-level
    """

    def __init__(self, name: str, task_type: str):
        assert task_type in {"binary", "multiclass", "multilabel"}
        self.name = name
        self.task_type = task_type


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "RNA_Site": TaskConfig("RNA_Site", "binary"),
    "RNA_Ligand": TaskConfig("RNA_Ligand", "multiclass"),
    "RNAGo": TaskConfig("RNAGo", "multilabel"),
}

# Map to the RNAGlibTask objects defined in `atomica_rna_embeddings.py`
RNAGLIB_TASKS = {
    "RNA_Site": RNA_SITE,
    "RNA_Ligand": RNA_LIGAND,
    "RNAGo": RNA_GO,
}


class AtomicaRNADataset(Dataset):
    """
    Dataset for loading Atomica and RNA-model (sequence) embeddings from parquet files.
    """

    def __init__(self, parquet_path: str, task_type: str):
        """
        Args:
            parquet_path: Path to parquet file produced by `atomica_rna_embeddings.py`
                          (or equivalent) with columns:
                          'id', 'atomica_embedding', 'sequence_embedding', 'label'.
            task_type: One of {"binary", "multiclass", "multilabel"}.
        """
        super().__init__()
        assert task_type in {"binary", "multiclass", "multilabel"}
        self.task_type = task_type

        self.data = pd.read_parquet(parquet_path)

        self.atomica_embeddings: List[np.ndarray] = []
        self.sequence_embeddings: List[np.ndarray] = []
        self.labels_raw = self.data["label"].values

        for _, row in self.data.iterrows():
            # Atomica
            atomica_emb = row["atomica_embedding"]
            if isinstance(atomica_emb, torch.Tensor):
                atomica_emb = atomica_emb.cpu().numpy()
            elif isinstance(atomica_emb, list):
                atomica_emb = np.array(atomica_emb)
            else:
                atomica_emb = np.array(atomica_emb)

            # RNA sequence embedding
            seq_emb = row["sequence_embedding"]
            if isinstance(seq_emb, torch.Tensor):
                seq_emb = seq_emb.cpu().numpy()
            elif isinstance(seq_emb, list):
                seq_emb = np.array(seq_emb)
            else:
                seq_emb = np.array(seq_emb)

            self.atomica_embeddings.append(atomica_emb)
            self.sequence_embeddings.append(seq_emb)

        self.atomica_embeddings = np.array(self.atomica_embeddings)
        self.sequence_embeddings = np.array(self.sequence_embeddings)

        # Standardize labels depending on task_type
        labels_list: List[np.ndarray] = []
        for y in self.labels_raw:
            y_arr = np.array(y)
            labels_list.append(y_arr)

        labels = np.array(labels_list, dtype=object)

        if task_type == "multiclass":
            # Expect scalar per sample
            labels = np.array([int(np.array(l).item()) for l in labels])
            self.labels = labels.astype(np.int64)
            self.num_classes = int(self.labels.max() + 1)
        elif task_type == "binary":
            # Expect scalar (0/1) per sample, stored as float32 for BCEWithLogits
            labels = np.array([float(np.array(l).item()) for l in labels])
            self.labels = labels.astype(np.float32).reshape(-1, 1)
            self.num_classes = 1
        else:  # multilabel
            # Expect vector per sample
            labels = np.stack([np.array(l, dtype=np.float32) for l in labels], axis=0)
            self.labels = labels
            self.num_classes = self.labels.shape[1]

        print(f"Loaded {len(self.data)} samples from {parquet_path}")
        print(f"Task type: {self.task_type}, num_classes: {self.num_classes}")
        print(f"Atomica embedding shape: {self.atomica_embeddings.shape}")
        print(f"Sequence embedding shape: {self.sequence_embeddings.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        atomica_emb = torch.FloatTensor(self.atomica_embeddings[idx])
        seq_emb = torch.FloatTensor(self.sequence_embeddings[idx])

        if self.task_type == "multiclass":
            label = torch.LongTensor([self.labels[idx]]).squeeze(0)
        else:  # binary or multilabel
            label = torch.FloatTensor(self.labels[idx])

        return {
            "atomica_embedding": atomica_emb,
            "sequence_embedding": seq_emb,
            "label": label,
            "id": self.data.iloc[idx]["id"],
        }


class AtomicaRNAMLP(nn.Module):
    """
    MLP for classification using concatenated Atomica and RNA-model embeddings.

    Supports optional L2 normalization, layer normalization, and projection layers
    exactly like `AtomicaESMMLP` but is agnostic to task type.
    """

    def __init__(
        self,
        atomica_embedding_dim: int,
        sequence_embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        final_hidden_dim: int = 32,
        normalize_l2: bool = True,
        use_layer_norm: bool = True,
        use_projection: bool = False,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()

        self.atomica_embedding_dim = atomica_embedding_dim
        self.sequence_embedding_dim = sequence_embedding_dim
        self.normalize_l2 = normalize_l2
        self.use_layer_norm = use_layer_norm
        self.use_projection = use_projection

        if use_projection:
            if projection_dim is None:
                projection_dim = (atomica_embedding_dim + sequence_embedding_dim) // 2
            self.atomica_projection = nn.Linear(atomica_embedding_dim, projection_dim)
            self.sequence_projection = nn.Linear(sequence_embedding_dim, projection_dim)
            input_dim = 2 * projection_dim
            print(
                f"Using projection layers: Atomica {atomica_embedding_dim}->{projection_dim}, "
                f"Sequence {sequence_embedding_dim}->{projection_dim}"
            )
        else:
            input_dim = atomica_embedding_dim + sequence_embedding_dim

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        else:
            self.layer_norm = None

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, num_classes),
        )

        if normalize_l2:
            print("Using L2 normalization for embeddings")
        if use_layer_norm:
            print("Using LayerNorm after concatenation")

    def forward(self, atomica_embedding: torch.Tensor, sequence_embedding: torch.Tensor) -> torch.Tensor:
        if self.normalize_l2:
            atomica_embedding = nn.functional.normalize(atomica_embedding, p=2, dim=1, eps=1e-8)
            sequence_embedding = nn.functional.normalize(sequence_embedding, p=2, dim=1, eps=1e-8)

        if self.use_projection:
            atomica_embedding = self.atomica_projection(atomica_embedding)
            sequence_embedding = self.sequence_projection(sequence_embedding)

        x = torch.cat([atomica_embedding, sequence_embedding], dim=1)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return self.classifier(x)


class EarlyStopping:
    """Early stopping based on a monitored validation metric (e.g., AUPRC)."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model: nn.Module) -> None:
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


def calculate_auprc_multitask(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    task_type: str,
) -> float:
    """
    Compute macro AUPRC appropriate to the task type.
    """
    if task_type == "multiclass":
        # y_true: (N,) with integer labels
        num_classes = y_scores.shape[1]
        y_true_onehot = np.zeros((y_true.shape[0], num_classes), dtype=np.float32)
        y_true_onehot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1.0
        return float(average_precision_score(y_true_onehot, y_scores, average="macro"))
    elif task_type == "binary":
        # y_true: (N,) or (N,1) binary; y_scores: (N,) or (N,1) probability of class 1
        y_true_flat = y_true.reshape(-1)
        scores_flat = y_scores.reshape(-1)
        return float(average_precision_score(y_true_flat, scores_flat))
    else:  # multilabel
        # y_true, y_scores: (N, C)
        return float(average_precision_score(y_true, y_scores, average="macro"))


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_type: str,
    threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and return loss, predictions, probabilities, and true labels.
    
    Returns
    -------
    avg_loss : float
    y_pred : np.ndarray
        - For multilabel/binary: shape (N, num_classes) with {0, 1} predictions
        - For multiclass: shape (N,) with class indices
    y_proba : np.ndarray
        - For multilabel/binary: shape (N, num_classes) with probabilities
        - For multiclass: shape (N, num_classes) with class probabilities
    y_true : np.ndarray
        Same shape as y_pred
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    if task_type in ["multilabel", "binary"]:
        criterion = nn.BCEWithLogitsLoss()
    else:  # multiclass
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            atomica_emb = batch["atomica_embedding"].to(device)
            seq_emb = batch["sequence_embedding"].to(device)
            labels = batch["label"].to(device)

            logits = model(atomica_emb, seq_emb)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Apply activation to get probabilities
            if task_type == "binary":
                # Binary: sigmoid gives (batch, 1) probability of class 1
                prob_class1 = torch.sigmoid(logits).cpu().numpy()  # (batch, 1)
                # Convert to 2-class format: [prob_class0, prob_class1]
                prob_class0 = 1.0 - prob_class1
                probabilities = np.concatenate([prob_class0, prob_class1], axis=1)  # (batch, 2)
                # Predictions: class index (0 or 1)
                predictions = (prob_class1.squeeze() >= threshold).astype(int)  # (batch,)
                # Labels: squeeze to (batch,)
                labels_np = labels.cpu().numpy()
                if labels_np.ndim == 2:
                    labels_np = labels_np.squeeze(1)
                all_labels.append(labels_np)
            elif task_type == "multilabel":
                probabilities = torch.sigmoid(logits).cpu().numpy()  # (batch, num_classes)
                predictions = (probabilities >= threshold).astype(int)  # (batch, num_classes)
                all_labels.append(labels.cpu().numpy())  # (batch, num_classes)
            else:  # multiclass
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()  # (batch, num_classes)
                predictions = np.argmax(probabilities, axis=1)  # (batch,)
                all_labels.append(labels.cpu().numpy())  # (batch,)
            
            all_probabilities.append(probabilities)
            all_predictions.append(predictions)
    
    avg_loss = total_loss / num_batches
    y_proba = np.vstack(all_probabilities)
    if task_type == "binary" or task_type == "multiclass":
        y_pred = np.concatenate(all_predictions)  # (N,)
        y_true = np.concatenate(all_labels)  # (N,)
    else:  # multilabel
        y_pred = np.vstack(all_predictions)  # (N, num_classes)
        y_true = np.vstack(all_labels)  # (N, num_classes)
    
    return avg_loss, y_pred, y_proba, y_true


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    task_type: str,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Training"):
        atomica_emb = batch["atomica_embedding"].to(device)
        seq_emb = batch["sequence_embedding"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(atomica_emb, seq_emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            if task_type == "multiclass":
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                lbl = labels.cpu().numpy().reshape(-1)
            elif task_type == "binary":
                probs_pos = torch.sigmoid(logits).view(-1)
                preds = (probs_pos >= 0.5).long().cpu().numpy()
                lbl = labels.cpu().numpy().reshape(-1)
            else:  # multilabel
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu().numpy()
                lbl = labels.cpu().numpy()

        all_preds.append(preds)
        all_labels.append(lbl)

    if task_type in {"multiclass", "binary"}:
        preds_cat = np.concatenate(all_preds, axis=0)
        labels_cat = np.concatenate(all_labels, axis=0)
        accuracy = float((preds_cat == labels_cat).mean())
    else:
        preds_cat = np.concatenate(all_preds, axis=0)
        labels_cat = np.concatenate(all_labels, axis=0)
        accuracy = float((preds_cat == labels_cat).all(axis=1).mean())

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
    }


def infer_default_parquet_paths(task_name: str) -> Tuple[str, str, str]:
    """
    Infer standard RNAGlib Atomica+RNA parquet paths from DATA_DIR and rnafm_name.

    We follow a convention consistent with `atomica_rna_embeddings.py`:
    - Base directory: {DATA_DIR}/atomica_ensemble_embeddings/{task_name}
    - Filenames:      atomica_{rnafm_name}_{task_name}_{split}_embeddings.parquet
      where split in {train, val, test}

    This assumes you have generated the corresponding parquet files using the
    same naming convention.
    """
    if task_name not in RNAGLIB_TASKS:
        raise ValueError(f"Unknown task_name for RNAGlib defaults: {task_name}")

    rnaglib_task = RNAGLIB_TASKS[task_name]
    rnafm_name = rnaglib_task.rnafm_name

    base_dir = os.path.join(DATA_DIR, "atomica_ensemble_embeddings", task_name)
    if task_name == "RNAGo":
        suffix = "_v2"
    else:
        suffix = ""
    train_path = os.path.join(
        base_dir, f"atomica_{rnafm_name}_{task_name}_train_embeddings{suffix}.parquet"
    )
    val_path = os.path.join(
        base_dir, f"atomica_{rnafm_name}_{task_name}_val_embeddings{suffix}.parquet"
    )
    test_path = os.path.join(
        base_dir, f"atomica_{rnafm_name}_{task_name}_test_embeddings{suffix}.parquet"
    )

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Atomica+RNA-model-based MLP for RNAGlib tasks"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        choices=list(TASK_CONFIGS.keys()),
        required=True,
        help="RNAGlib task name (determines task_type and default logging prefixes)",
    )
    parser.add_argument(
        "--train_parquet",
        type=str,
        default=None,
        help=(
            "Path to training parquet file with combined embeddings. "
            "If not provided, inferred from DATA_DIR and rnafm_name using the "
            "RNAGlib convention."
        ),
    )
    parser.add_argument(
        "--val_parquet",
        type=str,
        default=None,
        help=(
            "Path to validation parquet file with combined embeddings. "
            "If not provided, inferred from DATA_DIR and rnafm_name using the "
            "RNAGlib convention."
        ),
    )
    parser.add_argument(
        "--test_parquet",
        type=str,
        default=None,
        help=(
            "Path to test parquet file with combined embeddings. "
            "If not provided, inferred from DATA_DIR and rnafm_name using the "
            "RNAGlib convention."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for MLP",
    )
    parser.add_argument(
        "--final_hidden_dim",
        type=int,
        default=32,
        help="Final hidden dimension before output layer",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (based on validation AUPRC)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_benchmark/atomica_ensemble/",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="atomica-rna-mlp",
        help="Wandb project name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training",
    )
    parser.add_argument(
        "--normalize_l2",
        action="store_true",
        default=True,
        help="L2 normalize each embedding type before concatenation",
    )
    parser.add_argument(
        "--no_normalize_l2",
        dest="normalize_l2",
        action="store_false",
        help="Disable L2 normalization",
    )
    parser.add_argument(
        "--use_layer_norm",
        action="store_true",
        default=True,
        help="Use LayerNorm after concatenating embeddings",
    )
    parser.add_argument(
        "--no_layer_norm",
        dest="use_layer_norm",
        action="store_false",
        help="Disable LayerNorm after concatenation",
    )
    parser.add_argument(
        "--use_projection",
        action="store_true",
        default=False,
        help="Use learnable projection layers to align embedding spaces",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=None,
        help="Dimension for projection layers (default: average of embedding dims)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting probabilities to binary predictions (for multilabel/binary)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task_cfg = TASK_CONFIGS[args.task_name]
    task_type = task_cfg.task_type

    # Infer default parquet paths if not provided
    if args.train_parquet is None or args.val_parquet is None or args.test_parquet is None:
        inferred_train, inferred_val, inferred_test = infer_default_parquet_paths(args.task_name)
        if args.train_parquet is None:
            args.train_parquet = inferred_train
        if args.val_parquet is None:
            args.val_parquet = inferred_val
        if args.test_parquet is None:
            args.test_parquet = inferred_test

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"atomica_rna_mlp_{args.task_name}",
        dir=os.path.join(args.output_dir, "wandb"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task: {args.task_name}, task_type: {task_type}")

    # Load datasets
    print("Loading datasets...")
    print(f"Train parquet: {args.train_parquet}")
    print(f"Val parquet:   {args.val_parquet}")
    print(f"Test parquet:  {args.test_parquet}")

    train_dataset = AtomicaRNADataset(args.train_parquet, task_type=task_type)
    val_dataset = AtomicaRNADataset(args.val_parquet, task_type=task_type)
    test_dataset = AtomicaRNADataset(args.test_parquet, task_type=task_type)

    assert train_dataset.num_classes == val_dataset.num_classes == test_dataset.num_classes, (
        "Mismatch in num_classes across splits; check label processing."
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    atomica_dim = train_dataset.atomica_embeddings.shape[1]
    seq_dim = train_dataset.sequence_embeddings.shape[1]
    num_classes = train_dataset.num_classes

    model = AtomicaRNAMLP(
        atomica_embedding_dim=atomica_dim,
        sequence_embedding_dim=seq_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        final_hidden_dim=args.final_hidden_dim,
        normalize_l2=args.normalize_l2,
        use_layer_norm=args.use_layer_norm,
        use_projection=args.use_projection,
        projection_dim=args.projection_dim,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Atomica embedding dim: {atomica_dim}")
    print(f"Sequence embedding dim: {seq_dim}")
    print(f"Num classes: {num_classes}, task_type: {task_type}")

    # Loss function
    if task_type in {"binary", "multilabel"}:
        criterion = nn.BCEWithLogitsLoss()
    else:  # multiclass
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

    best_val_f1 = -1.0
    best_val_auprc = 0.0
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    val_f1_scores = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            task_type=task_type,
        )
        train_losses.append(train_metrics["loss"])

        val_loss, val_pred, val_proba, val_true = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            task_type=task_type,
            threshold=args.threshold,
        )
        val_losses.append(val_loss)

        # Compute validation metrics
        if task_type == "multiclass":
            val_metrics = compute_multiclass_metrics(val_true, val_pred, y_proba=val_proba)
            val_f1_macro = val_metrics.f1_macro
            val_auprc = calculate_auprc_multitask(val_true, val_proba, task_type="multiclass")
        elif task_type == "multilabel":
            val_metrics = compute_multilabel_metrics(
                val_true, y_pred=val_pred, y_proba=val_proba, threshold=args.threshold
            )
            val_f1_macro = val_metrics.f1_macro
            val_auprc = calculate_auprc_multitask(val_true, val_proba, task_type="multilabel")
        else:  # binary
            val_metrics = compute_binary_metrics(val_true, val_pred, y_proba=val_proba)
            val_f1_macro = val_metrics.auprc  # use AUPRC for binary classification
            val_auprc = val_metrics.auprc
        
        val_f1_scores.append(val_f1_macro)

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_f1_macro": val_f1_macro,
            "val_auprc": val_auprc,
        }
        if task_type in {"multiclass", "binary"}:
            log_dict["val_accuracy"] = val_metrics.accuracy
        elif task_type == "multilabel":
            log_dict["val_subset_accuracy"] = val_metrics.subset_accuracy

        wandb.log(log_dict)

        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1 Macro: {val_f1_macro:.4f}, Val AUPRC: {val_auprc:.4f}")

        # Check for best model (using F1 macro for multiclass/multilabel, AUPRC for binary)
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_val_auprc = val_auprc
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1_macro": val_f1_macro,
                "val_auprc": val_auprc,
                "val_loss": val_loss,
                "task_name": args.task_name,
                "task_type": task_type,
                "model_config": {
                    "atomica_embedding_dim": atomica_dim,
                    "sequence_embedding_dim": seq_dim,
                    "num_classes": num_classes,
                    "task_type": task_type,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "final_hidden_dim": args.final_hidden_dim,
                },
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            print(f"Saved best model (Val F1 Macro: {best_val_f1:.4f}, Val AUPRC: {best_val_auprc:.4f})")

        # Early stopping (using AUPRC)
        if early_stopping(val_auprc, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"\nTraining completed. Best validation F1 Macro: {best_val_f1:.4f}, AUPRC: {best_val_auprc:.4f} at epoch {best_epoch}")

    # Load best model for testing
    print(f"\nLoading best model from epoch {best_epoch} (Val F1 Macro: {best_val_f1:.4f}, Val AUPRC: {best_val_auprc:.4f})")
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_pred, test_proba, test_true = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        task_type=task_type,
        threshold=args.threshold,
    )

    # Compute test metrics
    print("\nComputing test metrics...")
    if task_type == "multiclass":
        test_metrics = compute_multiclass_metrics(test_true, test_pred, y_proba=test_proba)
        
        # Print metrics
        print("\n" + "="*80)
        print("TEST SET METRICS")
        print("="*80)
        print(f"Accuracy: {test_metrics.accuracy:.4f}")
        print(f"Balanced Accuracy: {test_metrics.balanced_accuracy:.4f}")
        print(f"\nF1 Scores:")
        print(f"  Macro:   {test_metrics.f1_macro:.4f}")
        print(f"  Micro:   {test_metrics.f1_micro:.4f}")
        print(f"  Weighted: {test_metrics.f1_weighted:.4f}")
        print(f"\nJaccard Index:")
        print(f"  Macro:   {test_metrics.jaccard_macro:.4f}")
        print(f"  Micro:   {test_metrics.jaccard_micro:.4f}")
        print(f"  Weighted: {test_metrics.jaccard_weighted:.4f}")
        
        if test_metrics.roc_auc_ovr_macro is not None:
            print(f"\nROC AUC (OvR):")
            print(f"  Macro:   {test_metrics.roc_auc_ovr_macro:.4f}")
            if test_metrics.roc_auc_ovr_weighted is not None:
                print(f"  Weighted: {test_metrics.roc_auc_ovr_weighted:.4f}")
        
        if test_metrics.roc_auc_ovo_macro is not None:
            print(f"\nROC AUC (OvO):")
            print(f"  Macro:   {test_metrics.roc_auc_ovo_macro:.4f}")
            if test_metrics.roc_auc_ovo_weighted is not None:
                print(f"  Weighted: {test_metrics.roc_auc_ovo_weighted:.4f}")
        
        print(f"\nPer-class metrics:")
        for label_name, metrics_dict in test_metrics.per_class.items():
            print(f"  Class {label_name}:")
            print(f"    Precision: {metrics_dict['precision']:.4f}")
            print(f"    Recall:    {metrics_dict['recall']:.4f}")
            print(f"    F1:        {metrics_dict['f1']:.4f}")
            print(f"    Jaccard:   {metrics_dict['jaccard']:.4f}")
            print(f"    Support:   {metrics_dict['support']:.0f}")
            if test_metrics.per_class_ovr_auc and label_name in test_metrics.per_class_ovr_auc:
                auc_val = test_metrics.per_class_ovr_auc[label_name]
                if auc_val is not None:
                    print(f"    ROC AUC:   {auc_val:.4f}")
        print("="*80)
        
    elif task_type == "binary":
        test_metrics = compute_binary_metrics(test_true, test_pred, y_proba=test_proba)
        
        # Print metrics
        print("\n" + "="*80)
        print("TEST SET METRICS (BINARY CLASSIFICATION)")
        print("="*80)
        print(f"Accuracy: {test_metrics.accuracy:.4f}")
        print(f"Balanced Accuracy: {test_metrics.balanced_accuracy:.4f}")
        print(f"\nROC AUC: {test_metrics.roc_auc:.4f}")
        print(f"AUPRC: {test_metrics.auprc:.4f}")
        print("="*80)
        
    else:  # multilabel
        test_metrics = compute_multilabel_metrics(
            test_true, y_pred=test_pred, y_proba=test_proba, threshold=args.threshold
        )
        
        # Print metrics
        print("\n" + "="*80)
        print("TEST SET METRICS")
        print("="*80)
        print(f"Subset Accuracy: {test_metrics.subset_accuracy:.4f}")
        print(f"\nF1 Scores:")
        print(f"  Macro:   {test_metrics.f1_macro:.4f}")
        print(f"  Micro:   {test_metrics.f1_micro:.4f}")
        print(f"  Weighted: {test_metrics.f1_weighted:.4f}")
        print(f"  Samples: {test_metrics.f1_samples:.4f}")
        print(f"\nJaccard Index:")
        print(f"  Macro:   {test_metrics.jaccard_macro:.4f}")
        print(f"  Micro:   {test_metrics.jaccard_micro:.4f}")
        print(f"  Weighted: {test_metrics.jaccard_weighted:.4f}")
        print(f"  Samples: {test_metrics.jaccard_samples:.4f}")
        
        if test_metrics.roc_auc_ovr_macro is not None:
            print(f"\nROC AUC (OvR):")
            print(f"  Macro:   {test_metrics.roc_auc_ovr_macro:.4f}")
            if test_metrics.roc_auc_ovr_micro is not None:
                print(f"  Micro:   {test_metrics.roc_auc_ovr_micro:.4f}")
            if test_metrics.roc_auc_ovr_weighted is not None:
                print(f"  Weighted: {test_metrics.roc_auc_ovr_weighted:.4f}")
        
        print(f"\nPer-label metrics:")
        for label_name, metrics_dict in test_metrics.per_label.items():
            print(f"  Label {label_name}:")
            print(f"    Precision: {metrics_dict['precision']:.4f}")
            print(f"    Recall:    {metrics_dict['recall']:.4f}")
            print(f"    F1:        {metrics_dict['f1']:.4f}")
            print(f"    Jaccard:   {metrics_dict['jaccard']:.4f}")
            print(f"    Support:   {metrics_dict['support']:.0f}")
            if test_metrics.per_label_ovr_auc and label_name in test_metrics.per_label_ovr_auc:
                auc_val = test_metrics.per_label_ovr_auc[label_name]
                if auc_val is not None:
                    print(f"    ROC AUC:   {auc_val:.4f}")
        print("="*80)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'test_predictions.npy')
    probabilities_path = os.path.join(args.output_dir, 'test_probabilities.npy')
    np.save(predictions_path, test_pred)
    np.save(probabilities_path, test_proba)
    print(f"\nSaved test predictions to: {predictions_path}")
    print(f"Saved test probabilities to: {probabilities_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics.to_dict(), f, indent=2)
    print(f"Saved test metrics to: {metrics_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'best_val_auprc': best_val_auprc,
    }
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    # Log final test metrics to wandb
    log_test = {
        "test_loss": test_loss,
        "best_epoch": best_epoch,
    }
    if task_type == "multiclass":
        log_test.update({
            "test_accuracy": test_metrics.accuracy,
            "test_balanced_accuracy": test_metrics.balanced_accuracy,
            "test_f1_macro": test_metrics.f1_macro,
            "test_f1_micro": test_metrics.f1_micro,
            "test_f1_weighted": test_metrics.f1_weighted,
        })
        if test_metrics.roc_auc_ovr_macro is not None:
            log_test["test_roc_auc_ovr_macro"] = test_metrics.roc_auc_ovr_macro
    elif task_type == "binary":
        log_test.update({
            "test_accuracy": test_metrics.accuracy,
            "test_balanced_accuracy": test_metrics.balanced_accuracy,
            "test_roc_auc": test_metrics.roc_auc,
            "test_auprc": test_metrics.auprc,
        })
    else:  # multilabel
        log_test.update({
            "test_subset_accuracy": test_metrics.subset_accuracy,
            "test_f1_macro": test_metrics.f1_macro,
            "test_f1_micro": test_metrics.f1_micro,
            "test_f1_weighted": test_metrics.f1_weighted,
        })
        if test_metrics.roc_auc_ovr_macro is not None:
            log_test["test_roc_auc_ovr_macro"] = test_metrics.roc_auc_ovr_macro
    wandb.log(log_test)

    wandb.finish()
    print(f"\nModel saved to: {os.path.join(args.output_dir, 'best_model.pt')}")
    print(f"All outputs saved to: {args.output_dir}")
    print("Training completed!")


if __name__ == "__main__":
    main()


