#!/usr/bin/env python3
"""
Training script for Atomica+ESM-based multiclass ligand classification using an MLP.
Features:
- MLP architecture combining Atomica and ESM embeddings
- Wandb logging
- Validation AUPRC-based checkpointing
- Early stopping
- Multiclass classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import wandb
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


DATA_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/atomica_ligand/"


class AtomicaESMEmbeddingDataset(Dataset):
    """Dataset class for loading Atomica and ESM embeddings from parquet files."""
    
    def __init__(self, parquet_path: str, label_encoder: Optional[LabelEncoder] = None):
        """
        Args:
            parquet_path: Path to parquet file containing Atomica and ESM embeddings
            label_encoder: Optional pre-fitted label encoder
        """
        self.data = pd.read_parquet(parquet_path)
        
        # Extract embeddings
        self.atomica_embeddings = []
        self.esm_embeddings = []
        
        for _, row in self.data.iterrows():
            # Convert atomica embeddings to numpy arrays
            if isinstance(row['atomica_embedding'], torch.Tensor):
                atomica_emb = row['atomica_embedding'].cpu().numpy()
            elif isinstance(row['atomica_embedding'], list):
                atomica_emb = np.array(row['atomica_embedding'])
            else:
                atomica_emb = np.array(row['atomica_embedding'])
            
            # Convert ESM embeddings to numpy arrays
            if isinstance(row['esm_embedding'], torch.Tensor):
                esm_emb = row['esm_embedding'].cpu().numpy()
            elif isinstance(row['esm_embedding'], list):
                esm_emb = np.array(row['esm_embedding'])
            else:
                esm_emb = np.array(row['esm_embedding'])
            
            self.atomica_embeddings.append(atomica_emb)
            self.esm_embeddings.append(esm_emb)
        
        # Convert to numpy arrays
        self.atomica_embeddings = np.array(self.atomica_embeddings)
        self.esm_embeddings = np.array(self.esm_embeddings)
        
        # Handle labels
        self.labels = self.data['label'].values
        
        # Fit or use provided label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(self.labels)
        
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Loaded {len(self.data)} samples with {self.num_classes} classes")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Atomica embedding shape: {self.atomica_embeddings.shape}")
        print(f"ESM embedding shape: {self.esm_embeddings.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'atomica_embedding': torch.FloatTensor(self.atomica_embeddings[idx]),
            'esm_embedding': torch.FloatTensor(self.esm_embeddings[idx]),
            'label': torch.LongTensor([self.encoded_labels[idx]]).squeeze(),
            'id': self.data.iloc[idx]['id']
        }


class AtomicaESMMLP(nn.Module):
    """MLP for multiclass classification using Atomica and ESM embeddings.
    
    Supports normalization strategies for better training when combining
    embeddings from different models:
    - L2 normalization: Normalizes each embedding type separately before concatenation
    - Layer normalization: Normalizes the concatenated embeddings
    - Projection layers: Optional learnable linear layers to align embedding spaces
    """
    
    def __init__(self, atomica_embedding_dim: int, esm_embedding_dim: int, 
                 hidden_dim: int = 512, num_classes: int = 7, dropout: float = 0.0,
                 final_hidden_dim: int = 32, 
                 normalize_l2: bool = True,
                 use_layer_norm: bool = True,
                 use_projection: bool = False,
                 projection_dim: Optional[int] = None):
        super(AtomicaESMMLP, self).__init__()
        self.atomica_embedding_dim = atomica_embedding_dim
        self.esm_embedding_dim = esm_embedding_dim
        self.normalize_l2 = normalize_l2
        self.use_layer_norm = use_layer_norm
        self.use_projection = use_projection
        
        # Optional projection layers to align embedding spaces
        if use_projection:
            if projection_dim is None:
                # Default: project both to the average of their dimensions
                projection_dim = (atomica_embedding_dim + esm_embedding_dim) // 2
            self.atomica_projection = nn.Linear(atomica_embedding_dim, projection_dim)
            self.esm_projection = nn.Linear(esm_embedding_dim, projection_dim)
            input_dim = 2 * projection_dim
            print(f"Using projection layers: Atomica {atomica_embedding_dim}->{projection_dim}, "
                  f"ESM {esm_embedding_dim}->{projection_dim}")
        else:
            input_dim = atomica_embedding_dim + esm_embedding_dim
        
        # Layer normalization after concatenation
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
            
            nn.Linear(final_hidden_dim, num_classes)
        )
        
        if normalize_l2:
            print("Using L2 normalization for embeddings")
        if use_layer_norm:
            print("Using LayerNorm after concatenation")

    def forward(self, atomica_embedding, esm_embedding):
        # L2 normalize each embedding separately
        if self.normalize_l2:
            # Add small epsilon to avoid division by zero
            atomica_embedding = nn.functional.normalize(atomica_embedding, p=2, dim=1, eps=1e-8)
            esm_embedding = nn.functional.normalize(esm_embedding, p=2, dim=1, eps=1e-8)
        
        # Optional projection layers
        if self.use_projection:
            atomica_embedding = self.atomica_projection(atomica_embedding)
            esm_embedding = self.esm_projection(esm_embedding)
        
        # Concatenate embeddings
        x = torch.cat([atomica_embedding, esm_embedding], dim=1)
        
        # Optional layer normalization after concatenation
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        return self.classifier(x)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()


def calculate_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate AUPRC for multiclass classification."""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_scores, average='macro')


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            atomica_emb = batch['atomica_embedding'].to(device)
            esm_emb = batch['esm_embedding'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(atomica_emb, esm_emb)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate AUPRC
    auprc = calculate_auprc(all_labels, all_probabilities)
    
    # Calculate accuracy
    accuracy = (all_predictions == all_labels).mean()
    
    return {
        'auprc': float(auprc),
        'accuracy': float(accuracy),
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probabilities.tolist(),
    }


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        atomica_emb = batch['atomica_embedding'].to(device)
        esm_emb = batch['esm_embedding'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(atomica_emb, esm_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Atomica+ESM-based MLP for ligand classification')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                       help='Directory containing parquet files')
    parser.add_argument('--dist_th', type=int, default=8,
                       help='Distance threshold for pocket definition')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed for embedding file selection')
    parser.add_argument('--model_type', type=str, default='esm2',
                       help='ESM model type (esm2 or esm3)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for MLP')
    parser.add_argument('--final_hidden_dim', type=int, default=32,
                       help='Final hidden dimension before output layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--wandb_project', type=str, default='atomica-esm-ligand-classification',
                       help='Wandb project name')
    parser.add_argument('--seed_train', type=int, default=42,
                       help='Random seed for training')
    parser.add_argument('--normalize_l2', action='store_true', default=True,
                       help='L2 normalize each embedding type separately before concatenation')
    parser.add_argument('--no_normalize_l2', dest='normalize_l2', action='store_false',
                       help='Disable L2 normalization')
    parser.add_argument('--use_layer_norm', action='store_true', default=True,
                       help='Use LayerNorm after concatenating embeddings')
    parser.add_argument('--no_layer_norm', dest='use_layer_norm', action='store_false',
                       help='Disable LayerNorm after concatenation')
    parser.add_argument('--use_projection', action='store_true', default=False,
                       help='Use learnable projection layers to align embedding spaces')
    parser.add_argument('--projection_dim', type=int, default=None,
                       help='Dimension for projection layers (default: average of embedding dims)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed_train)
    np.random.seed(args.seed_train)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=f"atomica_esm_ligand_classification",
        config=vars(args),
        name=f"atomica_{args.model_type}_mlp_dist{args.dist_th}_seed{args.seed}_lr{args.learning_rate}"
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    embeddings_dir = f"{args.data_dir}/atomica_{args.model_type}_embeddings"
    train_path = f"{embeddings_dir}/masif_ligand_pdbs_{args.dist_th}A_pocket_only_train_atomica_{args.model_type}_seed{args.seed}.parquet"
    val_path = f"{embeddings_dir}/masif_ligand_pdbs_{args.dist_th}A_pocket_only_val_atomica_{args.model_type}_seed{args.seed}.parquet"
    test_path = f"{embeddings_dir}/masif_ligand_pdbs_{args.dist_th}A_pocket_only_test_atomica_{args.model_type}_seed{args.seed}.parquet"
    
    print("Loading datasets...")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")
    print(f"Test path: {test_path}")
    
    train_dataset = AtomicaESMEmbeddingDataset(train_path)
    val_dataset = AtomicaESMEmbeddingDataset(val_path, train_dataset.label_encoder)
    test_dataset = AtomicaESMEmbeddingDataset(test_path, train_dataset.label_encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    atomica_embedding_dim = train_dataset.atomica_embeddings.shape[1]
    esm_embedding_dim = train_dataset.esm_embeddings.shape[1]
    
    model = AtomicaESMMLP(
        atomica_embedding_dim=atomica_embedding_dim,
        esm_embedding_dim=esm_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=train_dataset.num_classes,
        dropout=args.dropout,
        final_hidden_dim=args.final_hidden_dim,
        normalize_l2=args.normalize_l2,
        use_layer_norm=args.use_layer_norm,
        use_projection=args.use_projection,
        projection_dim=args.projection_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Atomica embedding dim: {atomica_embedding_dim}")
    print(f"ESM embedding dim: {esm_embedding_dim}")
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Training loop
    best_val_auprc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_auprc': val_metrics['auprc'],
            'val_accuracy': val_metrics['accuracy']
        })
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val AUPRC: {val_metrics['auprc']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Check for best model
        if val_metrics['auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            best_epoch = epoch + 1
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auprc': val_metrics['auprc'],
                'val_accuracy': val_metrics['accuracy'],
                'label_encoder': train_dataset.label_encoder,
                'args': vars(args)
            }
            torch.save(checkpoint, f"{args.output_dir}/best_model.pt")
            print(f"New best model saved with AUPRC: {best_val_auprc:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['auprc'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed. Best validation AUPRC: {best_val_auprc:.4f} at epoch {best_epoch}")
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(f"{args.output_dir}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"Test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Log final test metrics
    wandb.log({
        'test_auprc': test_metrics['auprc'],
        'test_accuracy': test_metrics['accuracy'],
        'best_epoch': best_epoch
    })
    
    # Save final results
    results = {
        'test_auprc': test_metrics['auprc'],
        'test_accuracy': test_metrics['accuracy'],
        'best_val_auprc': best_val_auprc,
        'best_epoch': best_epoch,
    }
    
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f"{args.output_dir}/test_predictions.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    wandb.finish()
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
