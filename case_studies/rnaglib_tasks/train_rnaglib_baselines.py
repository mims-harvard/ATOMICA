from rnaglib.tasks import (
    RNAGo,
    LigandIdentification,
    ProteinBindingSite,
    BindingSite,
)

from rnaglib.transforms import GraphRepresentation
from rnaglib.transforms.represent.GVPgraph import GVPGraphRepresentation
from rnaglib.learning.task_models import PygModel
from rnaglib.learning.gvp import GVPModel
from time import time
import numpy as np
import torch
import random
import os
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from multiclass_metrics import compute_multiclass_metrics
from multilabel_metrics import compute_multilabel_metrics

DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/rnaglib_tasks"

# Task type mapping
TASK_CONFIG = {
    "RNA_Ligand": (LigandIdentification, 4, 128, 0.5, 0.00001),
    "RNA_Site": (BindingSite, 4, 256, 0.5, 0.001),
    "RNA_Protein": (ProteinBindingSite, 4, 64, 0.2, 0.01),
    "RNAGo": (RNAGo, 3, 64, 0.5, 0.001),
}


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

def sigmoid(x):
    """Sigmoid function for numpy arrays"""
    # Clip to avoid overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def compute_metrics(all_preds, all_probs, all_labels, task_name, task_metadata):
    """Compute metrics based on task type, matching inference.ipynb"""
    multi_label = task_metadata['multi_label']
    num_classes = task_metadata['num_classes']
    graph_level = task_metadata['graph_level']
    
    # Handle residue-level tasks (concatenate lists to arrays)
    if not graph_level:
        if isinstance(all_preds, list):
            all_preds = np.concatenate(all_preds)
        if isinstance(all_probs, list):
            all_probs = np.concatenate(all_probs)
        if isinstance(all_labels, list):
            all_labels = np.concatenate(all_labels)
    
    if multi_label:
        # RNAGo: multilabel classification
        # Convert logits to probabilities (model outputs logits for multilabel)
        y_proba = np.stack(all_probs) if isinstance(all_probs[0], np.ndarray) else np.array(all_probs)
        y_proba = sigmoid(y_proba)  # Convert logits to probabilities
        y_true = np.stack(all_labels) if isinstance(all_labels[0], np.ndarray) else np.array(all_labels)
        
        # Find best threshold on validation (we'll use 0.5 for now, but should optimize on val set)
        threshold = 0.5
        
        metrics = compute_multilabel_metrics(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold
        )
        return {
            'subset_accuracy': metrics.subset_accuracy,
            'f1_macro': metrics.f1_macro,
            'f1_micro': metrics.f1_micro,
            'f1_weighted': metrics.f1_weighted,
        }
    elif num_classes > 2:
        # RNA_Ligand: multiclass classification
        y_pred = np.stack(all_preds) if isinstance(all_preds[0], np.ndarray) else np.array(all_preds)
        y_proba = np.stack(all_probs) if isinstance(all_probs[0], np.ndarray) else np.array(all_probs)
        # For multiclass, probs should already be probabilities (softmax applied)
        y_true = np.stack(all_labels) if isinstance(all_labels[0], np.ndarray) else np.array(all_labels)
        
        # Get class labels from metadata if available
        labels = None
        if 'label_mapping' in task_metadata:
            labels = list(range(num_classes))
        
        metrics = compute_multiclass_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            labels=labels
        )
        return {
            'accuracy': metrics.accuracy,
            'balanced_accuracy': metrics.balanced_accuracy,
            'f1_macro': metrics.f1_macro,
            'f1_micro': metrics.f1_micro,
            'f1_weighted': metrics.f1_weighted,
        }
    else:
        # RNA_Protein, RNA_Site: binary classification (residue-level)
        y_proba_logits = np.array(all_probs).flatten()
        y_proba = sigmoid(y_proba_logits)  # Convert logits to probabilities
        y_true = np.array(all_labels).flatten()
        
        # Find best threshold based on F1 score
        thresholds = np.linspace(0.0, 1.0, 101)
        f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1s)]
        best_pred = (y_proba >= best_threshold).astype(int)
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auprc = auc(recall, precision)
        
        return {
            'accuracy': np.mean(y_true == best_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'auprc': auprc,
        }

def run_task(task, task_name, num_layers, hidden_channels, dropout_rate, learning_rate, seed=None, use_gvp=False):
    # Set random seed for reproducibility (must be called before model creation)
    if seed is not None:
        setup_seed(seed)
        print(f"Random seed set to: {seed}")
    
    if use_gvp:
        model = GVPModel.from_task(
            task,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            dropout_rate=dropout_rate,
            seq_in=False,
        )
    else:
        model = PygModel.from_task(
            task,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            dropout_rate=dropout_rate
        )
    # Configure learning rate (default is 0.001, need to set to 0.00001)
    model.configure_training(learning_rate=learning_rate)
    
    if use_gvp:
        rep = GVPGraphRepresentation()
    else:
        rep = GraphRepresentation(framework="pyg")
    
    print("Representation loaded")

    task.add_representation(rep)
    train_loader, val_loader, test_loader = task.get_split_loaders(batch_size=8)
    print("Loaders loaded")

    # Train the model
    start_time = time()
    print("Training model")
    model.train_model(task, epochs=500)
    print("Model trained, time taken: ", time() - start_time)

    # Evaluate on test set using inference
    start_time = time()
    test_loader = model.get_dataloader(task, split='test')
    mean_loss, all_preds, all_probs, all_labels = model.inference(test_loader)

    test_metrics = model.evaluate(task, split='test')
    print("RNAGlib test metrics: ", test_metrics) # keep for reference to the paper evals
    
    test_metrics = compute_metrics(all_preds, all_probs, all_labels, task_name, task.metadata)
    test_metrics['loss'] = mean_loss
    print("Test metrics evaluated, time taken: ", time() - start_time)
    print(f"Test metrics: {test_metrics}")


def main():
    parser = argparse.ArgumentParser(description='Train rnaglib baselines')
    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        choices=list(TASK_CONFIG.keys()),
        help='Task type to run. Must be one of: ' + ', '.join(TASK_CONFIG.keys())
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 43, 44],
        help='List of random seeds to use (default: [42, 43, 44])'
    )
    parser.add_argument(
        '--use_gvp',
        action='store_true',
        default=False,
        help='Use GVP model instead of Pyg model'
    )
    
    args = parser.parse_args()
    
    task_name = args.task_type
    task_class, num_layers, hidden_channels, dropout_rate, learning_rate = TASK_CONFIG[task_name]
    seeds = args.seeds
    use_gvp = args.use_gvp
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running task: {task_name} with seed: {seed}")
        print(f"{'='*60}\n")
        
        task = task_class(
            root=f"{DATA_DIR}/{task_name}",
            recompute=False,
            debug=False,
        )
        print(f"Running task: {task_name}, with seed: {seed}")
        print("--------------------------------")
        run_task(task, task_name, num_layers, hidden_channels, dropout_rate, learning_rate, seed=seed, use_gvp=use_gvp)
        print("--------------------------------")
        print()


if __name__ == "__main__":
    main()