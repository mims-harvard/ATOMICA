#!/usr/bin/env python3
"""
Classification with 3-Layer MLP

Supports:
- Multilabel classification (each item can have multiple labels)
- Binary classification (single label with 2 classes)
- Multiclass classification (each item has exactly one label)

Trains a 3-layer MLP on embeddings and saves the best model checkpoint based on validation performance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
import random
from typing import Optional, Tuple, Literal, Dict, Any
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path to import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multilabel_metrics import compute_multilabel_metrics, MultilabelMetricsResult
from multiclass_metrics import compute_multiclass_metrics, MetricsResult as MulticlassMetricsResult

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from collections import Counter


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    For multiclass: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    For multilabel: Applied per-label with BCEWithLogitsLoss base

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values focus more on hard examples. Default: 2.0
    alpha : float, list, or torch.Tensor, optional
        Class weighting. Can be:
        - float: same weight for all classes
        - list/tensor: per-class weights
        - None: no weighting
    task_type : str
        One of "multiclass" or "multilabel"
    """
    def __init__(self, gamma: float = 2.0, alpha=None, task_type: str = "multiclass"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.task_type = task_type

        # Handle alpha (class weights)
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                raise TypeError(f"alpha must be float, list, or tensor, got {type(alpha)}")
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Logits from model (before activation)
            - Multiclass: shape (N, num_classes)
            - Multilabel/Binary: shape (N, num_classes) or (N, 1)
        targets : torch.Tensor
            True labels
            - Multiclass: shape (N,) with class indices
            - Multilabel/Binary: shape (N, num_classes) or (N, 1) with {0, 1}
        """
        if self.task_type == "multiclass":
            # Multiclass focal loss
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            p = torch.exp(-ce_loss)  # p_t
            focal_loss = (1 - p) ** self.gamma * ce_loss

            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                # Apply per-class weights
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss

            return focal_loss.mean()

        elif self.task_type == "multilabel":
            # Multilabel focal loss (per-label binary focal loss)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            p = torch.sigmoid(inputs)
            p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1, 1-p if y=0
            focal_loss = (1 - p_t) ** self.gamma * bce_loss

            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                # Apply per-class weights (alpha for positive class)
                alpha_t = self.alpha.unsqueeze(0) * targets + (1 - targets)
                focal_loss = alpha_t * focal_loss

            return focal_loss.mean()

        else:
            raise ValueError(f"task_type must be 'multiclass' or 'multilabel', got {self.task_type}")


def setup_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class BinaryMetricsResult:
    """Metrics result for binary classification."""
    # Global metrics
    accuracy: float
    balanced_accuracy: float
    
    # ROC AUC
    roc_auc: float
    auprc: float

    def to_dict(self) -> Dict[str, Any]:
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

TaskType = Literal["multilabel", "binary", "multiclass"]


class EmbeddingDataset(Dataset):
    """Dataset for embeddings and labels."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, task_type: TaskType):
        """
        Parameters
        ----------
        embeddings : np.ndarray of shape (N, embedding_dim)
        labels : np.ndarray
            - For multilabel: shape (N, num_classes) with {0, 1} indicators
            - For binary: shape (N, 1) or (N,) with {0, 1} 
            - For multiclass: shape (N,) with class indices
        task_type : str
            One of "multilabel", "binary", "multiclass"
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.task_type = task_type
        
        # Handle labels based on task type
        labels = np.asarray(labels)
        if task_type == "binary":
            # Ensure binary labels are shape (N, 1) for consistency
            if labels.ndim == 1:
                labels = labels.reshape(-1, 1)
            self.labels = torch.FloatTensor(labels)
        elif task_type == "multilabel":
            # Multilabel: (N, num_classes) with {0, 1}
            if labels.ndim == 1:
                raise ValueError("Multilabel labels must be 2D array (N, num_classes)")
            self.labels = torch.FloatTensor(labels)
        else:  # multiclass
            # Multiclass: (N,) with class indices
            if labels.ndim != 1:
                raise ValueError("Multiclass labels must be 1D array (N,)")
            self.labels = torch.LongTensor(labels)
        
        assert len(self.embeddings) == len(self.labels), "Embeddings and labels must have same length"
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """3-layer MLP for classification (supports multilabel, binary, and multiclass)."""
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        task_type: TaskType,
        hidden_dim: int = 512, 
        final_hidden_dim: int = 32,
        dropout: float = 0.3
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input embeddings
        num_classes : int
            Number of output classes
        task_type : str
            One of "multilabel", "binary", "multiclass"
        hidden_dim : int
            Hidden layer dimension (default: 512)
        final_hidden_dim : int
            Final hidden layer dimension (default: 32)
        dropout : float
            Dropout probability (default: 0.3)
        """
        super(MLPClassifier, self).__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim, final_hidden_dim)
        self.bn3 = nn.BatchNorm1d(final_hidden_dim)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(final_hidden_dim, num_classes)
        
    def forward(self, x, apply_activation: bool = False):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings
        apply_activation : bool
            If True, apply sigmoid (multilabel/binary) or softmax (multiclass)
            If False, return logits (default for training)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Final hidden layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        
        # Optional activation for inference
        if apply_activation:
            if self.task_type in ["multilabel", "binary"]:
                x = torch.sigmoid(x)
            else:  # multiclass
                x = torch.softmax(x, dim=1)
        
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding at the final hidden layer (before the classification layer).
        
        This is typically used to extract a fixed-size representation for downstream tasks.
        """
        self.eval()
        with torch.no_grad():
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            
            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.dropout3(x)
        
        return x


def load_data(embeddings_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from given paths."""
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)

    print(f"Loaded: embeddings shape {embeddings.shape}, labels shape {labels.shape}")

    return embeddings, labels


def compute_class_weights(
    labels: np.ndarray,
    task_type: TaskType,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute class weights for weighted loss. For multiclass: use 1/count instead of (total-count)/count

    Parameters
    ----------
    labels : np.ndarray
        Labels array
        - For multiclass: shape (N,) with class indices
        - For multilabel: shape (N, num_classes) with {0, 1}
    task_type : str
        One of "multiclass" or "multilabel"
    num_classes : int
        Number of classes

    Returns
    -------
    torch.Tensor
        Class weights
        - For multiclass: shape (num_classes,) - weights for each class
        - For multilabel: shape (num_classes,) - pos_weight for each class
    """
    if task_type == "multilabel":
        # For multilabel: calculate pos_weight (weight for positive examples relative to negative)
        # pos_weight[i] = num_negatives[i] / num_positives[i] for each class i
        labels_array = np.asarray(labels)

        if labels_array.ndim != 2:
            raise ValueError(f"Multilabel labels must be 2D array (N, num_classes), got shape {labels_array.shape}")

        if labels_array.shape[1] != num_classes:
            print(f"Warning: labels have {labels_array.shape[1]} classes but num_classes={num_classes}")

        class_weights = torch.zeros(num_classes, dtype=torch.float32)
        num_samples = labels_array.shape[0]

        for class_idx in range(num_classes):
            if labels_array.shape[1] > class_idx:
                positives = labels_array[:, class_idx].sum()
                negatives = num_samples - positives

                if positives > 0:
                    class_weights[class_idx] = float(negatives) / float(positives)
                else:
                    # If no positive examples, set weight to 1.0 (no weighting)
                    class_weights[class_idx] = 1.0
            else:
                class_weights[class_idx] = 1.0

        # Log per-class statistics
        print(f'Pos weights for multilabel weighted loss: {class_weights.tolist()}')
        for class_idx in range(num_classes):
            if labels_array.shape[1] > class_idx:
                positives = int(labels_array[:, class_idx].sum())
                negatives = int(num_samples - positives)
                print(f'Class {class_idx}: {positives} positives, {negatives} negatives, pos_weight={class_weights[class_idx]:.4f}')

    else:  # multiclass
        # For multiclass: calculate inverse frequency weights
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"Multiclass labels must be 1D array (N,), got shape {labels.shape}")

        label_counts = Counter(labels)
        class_weights = torch.zeros(num_classes, dtype=torch.float32)
        total_samples = len(labels)

        for class_idx in range(num_classes):
            if class_idx in label_counts:
                class_weights[class_idx] = 1.0 / label_counts[class_idx]
            else:
                class_weights[class_idx] = 0.0

        # Normalize weights so average weight is 1
        total_weight = class_weights.sum()
        if total_weight > 0:
            class_weights = class_weights / total_weight * num_classes
        else:
            class_weights = torch.ones(num_classes, dtype=torch.float32)

        print(f'Class weights for weighted loss: {class_weights.tolist()}')
        print(f'Class distribution: {dict(label_counts)}')

    return class_weights


def get_next_version_number(output_dir: str) -> int:
    """
    Get the next version number for a new model directory.
    
    Parameters
    ----------
    output_dir : str
        Base output directory
        
    Returns
    -------
    int
        Next version number (e.g., 0, 1, 2, ...)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all existing version directories
    existing_versions = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("version_") and os.path.isdir(os.path.join(output_dir, item)):
                try:
                    version_num = int(item.split("_")[1])
                    existing_versions.append(version_num)
                except (ValueError, IndexError):
                    continue
    
    # Return the next version number
    if existing_versions:
        return max(existing_versions) + 1
    else:
        return 0


def train_epoch(model, train_loader, criterion, optimizer, device, task_type: TaskType):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for embeddings, labels in tqdm(train_loader, desc="Training", leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings, apply_activation=False)  # Get logits for loss computation

        # For binary classification, reshape labels to match outputs shape
        if task_type == "binary" and labels.ndim == 1:
            labels = labels.unsqueeze(1).float()  # (batch,) -> (batch, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(
    model, 
    data_loader, 
    device, 
    task_type: TaskType,
    threshold: float = 0.5
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and return predictions.
    
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
        criterion = nn.BCEWithLogitsLoss()  # Use logits version
    else:  # multiclass
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for embeddings, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings, apply_activation=False)

            # For binary classification, reshape labels to match logits shape
            if task_type == "binary" and labels.ndim == 1:
                labels = labels.unsqueeze(1).float()  # (batch,) -> (batch, 1)

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
                predictions = (prob_class1.squeeze(1) >= threshold).astype(int)  # (batch,)
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


def main(
    train_embeddings_path: str,
    train_labels_path: str,
    val_embeddings_path: str,
    val_labels_path: str,
    test_embeddings_path: str,
    test_labels_path: str,
    task_type: TaskType,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    threshold: float = 0.5,
    patience: int = 10,
    device: Optional[str] = None,
    output_dir: str = "checkpoints",
    seed: int = 42,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[list] = None,
    weighted_loss: bool = False,
):
    """
    Main training function.

    Parameters
    ----------
    train_embeddings_path : str
        Path to training embeddings .npy file
    train_labels_path : str
        Path to training labels .npy file
    val_embeddings_path : str
        Path to validation embeddings .npy file
    val_labels_path : str
        Path to validation labels .npy file
    test_embeddings_path : str
        Path to test embeddings .npy file
    test_labels_path : str
        Path to test labels .npy file
    task_type : str
        One of "multilabel", "binary", "multiclass"
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate
    num_epochs : int
        Maximum number of epochs
    hidden_dim : int
        Hidden layer dimension
    dropout : float
        Dropout probability
    threshold : float
        Threshold for converting probabilities to binary predictions (for multilabel/binary)
    patience : int
        Early stopping patience
    device : str
        Device to use (cuda/cpu). If None, auto-detect.
    output_dir : str
        Directory to save model and predictions
    use_focal_loss : bool
        Use focal loss instead of cross-entropy (default: False)
    focal_gamma : float
        Focusing parameter for focal loss (default: 2.0)
    focal_alpha : list, optional
        Class weighting for focal loss
    weighted_loss : bool
        Use weighted cross-entropy/BCE loss (default: False)
    """
    setup_seed(seed)
    # Validate task type
    if task_type not in ["multilabel", "binary", "multiclass"]:
        raise ValueError(f"task_type must be one of 'multilabel', 'binary', 'multiclass', got '{task_type}'")
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Task type: {task_type}")
    
    # Create versioned directory
    # version_num = get_next_version_number(output_dir)
    version_dir = output_dir
    os.makedirs(version_dir, exist_ok=True)
    print(f"Created version directory: {version_dir}")
    
    # Load data
    print("Loading data...")
    train_embeddings, train_labels = load_data(train_embeddings_path, train_labels_path)
    val_embeddings, val_labels = load_data(val_embeddings_path, val_labels_path)
    test_embeddings, test_labels = load_data(test_embeddings_path, test_labels_path)
    
    # Determine input and output dimensions
    input_dim = train_embeddings.shape[1]
    
    if task_type == "multiclass":
        # Multiclass: labels are (N,) with class indices
        num_classes = int(np.max(train_labels) + 1)
        print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    elif task_type == "binary":
        # Binary: labels are (N, 1) or (N,)
        num_classes = 1
        print(f"Input dimension: {input_dim}, Binary classification (num_classes=1)")
    else:  # multilabel
        # Multilabel: labels are (N, num_classes)
        num_classes = train_labels.shape[1]
        print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    
    # Compute class weights if requested
    class_weights = None
    if weighted_loss or use_focal_loss:
        if task_type == "binary":
            print("Warning: weighted_loss and focal_loss are not supported for binary classification in this script")
        else:
            print("Computing class weights...")
            class_weights = compute_class_weights(
                train_labels,
                task_type,
                num_classes,
            )

    # Determine loss function name for logging
    if use_focal_loss:
        loss_name = "FocalLoss"
    elif weighted_loss:
        if task_type == "multilabel":
            loss_name = "WeightedBCEWithLogitsLoss"
        else:  # multiclass
            loss_name = "WeightedCrossEntropyLoss"
    else:
        loss_name = "BCEWithLogitsLoss" if task_type in ["multilabel", "binary"] else "CrossEntropyLoss"

    # Save hyperparameters
    hyperparameters = {
        "task_type": task_type,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "threshold": threshold,
        "patience": patience,
        "device": device,
        "optimizer": "Adam",
        "loss": loss_name,
        "use_focal_loss": use_focal_loss,
        "focal_gamma": focal_gamma if use_focal_loss else None,
        "focal_alpha": focal_alpha if use_focal_loss else None,
        "weighted_loss": weighted_loss,
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "train_size": int(len(train_embeddings)),
        "val_size": int(len(val_embeddings)),
        "test_size": int(len(test_embeddings)),
        "train_embeddings_path": train_embeddings_path,
        "train_labels_path": train_labels_path,
        "val_embeddings_path": val_embeddings_path,
        "val_labels_path": val_labels_path,
        "test_embeddings_path": test_embeddings_path,
        "test_labels_path": test_labels_path,
    }
    hyperparams_path = os.path.join(version_dir, "hyperparameters.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparameters, f, indent=2)
    print(f"Saved hyperparameters to: {hyperparams_path}")
    
    # Create datasets and data loaders
    train_dataset = EmbeddingDataset(train_embeddings, train_labels, task_type)
    val_dataset = EmbeddingDataset(val_embeddings, val_labels, task_type)
    test_dataset = EmbeddingDataset(test_embeddings, test_labels, task_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MLPClassifier(
        input_dim=input_dim, 
        num_classes=num_classes, 
        task_type=task_type,
        hidden_dim=hidden_dim, 
        dropout=dropout
    )
    model = model.to(device)

    # Loss and optimizer
    if use_focal_loss:
        # Use focal loss
        if focal_alpha is not None:
            alpha = focal_alpha
        elif class_weights is not None:
            alpha = class_weights
        else:
            alpha = None

        criterion = FocalLoss(
            gamma=focal_gamma,
            alpha=alpha,
            task_type="multilabel" if task_type == "multilabel" else "multiclass"
        )
        print(f"Using Focal Loss with gamma={focal_gamma}, alpha={'custom' if alpha is not None else 'none'}")
    elif weighted_loss and class_weights is not None:
        # Use weighted loss
        if task_type == "multilabel":
            # For multilabel, use pos_weight parameter
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
            print(f"Using Weighted BCEWithLogitsLoss with pos_weight")
        else:  # multiclass
            # For multiclass, use weight parameter
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"Using Weighted CrossEntropyLoss")
    else:
        # Standard loss
        if task_type in ["multilabel", "binary"]:
            criterion = nn.BCEWithLogitsLoss()
        else:  # multiclass
            criterion = nn.CrossEntropyLoss()
        print(f"Using standard {'BCEWithLogitsLoss' if task_type in ['multilabel', 'binary'] else 'CrossEntropyLoss'}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_f1_scores = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, task_type)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_pred, val_proba, val_true = evaluate(model, val_loader, device, task_type, threshold)
        val_losses.append(val_loss)
        
        # Compute validation metrics
        if task_type == "multiclass":
            val_metrics = compute_multiclass_metrics(val_true, val_pred, y_proba=val_proba)
            val_f1_macro = val_metrics.f1_macro
        elif task_type == "multilabel":
            val_metrics = compute_multilabel_metrics(
                val_true, y_pred=val_pred, y_proba=val_proba, threshold=threshold
            )
            val_f1_macro = val_metrics.f1_macro
        else: # binary
            val_metrics = compute_binary_metrics(val_true, val_pred, y_proba=val_proba)
            val_f1_macro = val_metrics.auprc # use AUPRC for binary classification
        
        val_f1_scores.append(val_f1_macro)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1 Macro: {val_f1_macro:.4f}")
        
        # Save best model
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1_macro': val_f1_macro,
                'val_loss': val_loss,
                'model_config': {
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'task_type': task_type,
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                }
            }
            torch.save(checkpoint, os.path.join(version_dir, 'best_model.pt'))
            print(f"Saved best model (Val F1 Macro: {best_val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (patience: {patience})")
            break
    
    # Load best model for testing
    print(f"\nLoading best model from epoch {best_epoch + 1} (Val F1 Macro: {best_val_f1:.4f})")
    checkpoint = torch.load(os.path.join(version_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_pred, test_proba, test_true = evaluate(model, test_loader, device, task_type, threshold)
    
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
        print(f"\nROC AUC: {test_metrics.roc_auc:.4f}")
        print(f"\nROC AUPRC: {test_metrics.auprc:.4f}")
        print("="*80)
    else:  # multilabel
        test_metrics = compute_multilabel_metrics(
            test_true, y_pred=test_pred, y_proba=test_proba, threshold=threshold
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
    predictions_path = os.path.join(version_dir, 'test_predictions.npy')
    probabilities_path = os.path.join(version_dir, 'test_probabilities.npy')
    np.save(predictions_path, test_pred)
    np.save(probabilities_path, test_proba)
    print(f"\nSaved test predictions to: {predictions_path}")
    print(f"Saved test probabilities to: {probabilities_path}")
    
    # Save metrics
    metrics_path = os.path.join(version_dir, 'test_metrics.json')
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
    }
    history_path = os.path.join(version_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")
    
    print(f"\nModel saved to: {os.path.join(version_dir, 'best_model.pt')}")
    print(f"All outputs saved to: {version_dir}")
    print("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 3-layer MLP for classification (multilabel/binary/multiclass)")
    parser.add_argument("--train-embeddings", type=str, required=True, help="Path to training embeddings .npy file")
    parser.add_argument("--train-labels", type=str, required=True, help="Path to training labels .npy file")
    parser.add_argument("--val-embeddings", type=str, required=True, help="Path to validation embeddings .npy file")
    parser.add_argument("--val-labels", type=str, required=True, help="Path to validation labels .npy file")
    parser.add_argument("--test-embeddings", type=str, required=True, help="Path to test embeddings .npy file")
    parser.add_argument("--test-labels", type=str, required=True, help="Path to test labels .npy file")
    parser.add_argument("--task-type", type=str, required=True, choices=["multilabel", "binary", "multiclass"],
                       help="Task type: multilabel, binary, or multiclass")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary predictions (multilabel/binary only)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Focal loss arguments
    parser.add_argument("--use-focal-loss", action="store_true", default=False,
                       help="Use focal loss instead of cross-entropy for classification tasks. Helps with class imbalance.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focusing parameter for focal loss. Higher values focus more on hard examples. Default: 2.0")
    parser.add_argument("--focal-alpha", type=float, nargs='*', default=None,
                       help="Class weighting for focal loss. Can be a single value or per-class weights. If not specified, computed weights are used if weighted-loss is also enabled.")

    # Weighted loss arguments
    parser.add_argument("--weighted-loss", action="store_true", default=False,
                       help="Use weighted cross-entropy/BCE loss for classification tasks. Helps with class imbalance.")

    args = parser.parse_args()
    
    main(
        train_embeddings_path=args.train_embeddings,
        train_labels_path=args.train_labels,
        val_embeddings_path=args.val_embeddings,
        val_labels_path=args.val_labels,
        test_embeddings_path=args.test_embeddings,
        test_labels_path=args.test_labels,
        task_type=args.task_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        threshold=args.threshold,
        patience=args.patience,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir,
        seed=args.seed,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        weighted_loss=args.weighted_loss,
    )
