import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/MASIF/dpocket/dpocket_out_4A_exp_with_split.csv"
OUT_DIR = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ATOMICA/baselines/MASIF/dpocket"


class MLP(nn.Module):
    """Simple 3-layer MLP for multiclass classification."""

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int = 7, dropout: float = 0.0):
        super().__init__()
        assert len(hidden_dims) == 2, "Expecting two hidden layers for a 3-layer MLP (2 hidden + output)."

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add small dropout for regularization
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes),
        )
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_dataloader(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_map: dict,
    split: str,
    batch_size: int = 128,
    feature_mean: np.ndarray = None,
    feature_std: np.ndarray = None,
) -> DataLoader:
    split_df = df[df["split"] == split].reset_index(drop=True)

    x = split_df[feature_cols].to_numpy(dtype=np.float32)
    y = split_df["lig"].map(label_map).to_numpy(dtype=np.int64)

    # Normalize features if statistics provided
    if feature_mean is not None and feature_std is not None:
        x = (x - feature_mean) / (feature_std + 1e-8)  # Add small epsilon to avoid division by zero

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(x_tensor, y_tensor)
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected, skipping batch")
            continue
        
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: List[str],
    label_map: dict,
    device: torch.device,
    feature_mean: np.ndarray = None,
    feature_std: np.ndarray = None,
) -> pd.DataFrame:
    """Collect predictions for all rows in df and return a DataFrame."""
    inv_label_map = {v: k for k, v in label_map.items()}

    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_true = df["lig"].map(label_map).to_numpy(dtype=np.int64)
    ids = df["id"].to_numpy()
    splits = df["split"].to_numpy()

    # Normalize features if statistics provided
    if feature_mean is not None and feature_std is not None:
        x = (x - feature_mean) / (feature_std + 1e-8)

    x_tensor = torch.from_numpy(x).to(device)

    model.eval()
    logits = model(x_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    # Store probability vector (length-7) in a single column
    prob_list = [probs[i] for i in range(probs.shape[0])]

    pred_labels = [inv_label_map[int(p)] for p in preds]
    true_labels = [inv_label_map[int(t)] for t in y_true]

    out_df = pd.DataFrame(
        {
            "id": ids,
            "pred_probability": prob_list,
            "pred_label": pred_labels,
            "true_label": true_labels,
            "split": splits,
        }
    )
    return out_df


def main(seed: int = 42):
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(DATA_PATH)
    # df = df.drop(columns=[
    #     'ALA',
    #     'ARG',
    #     'ASN',
    #     'ASP',
    #     'CYS',
    #     'GLN',
    #     'GLU',
    #     'GLY',
    #     'HIS',
    #     'ILE',
    #     'LEU',
    #     'LYS',
    #     'MET',
    #     'PHE',
    #     'PRO',
    #     'SER',
    #     'THR',
    #     'TRP',
    #     'TYR',
    #     'VAL'
    # ])

    # Determine feature columns: all except 'id', 'lig', 'split'
    exclude_cols = {"id", "lig", "split"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Build label mapping from lig values to integer classes [0..6]
    lig_values = sorted(df["lig"].unique())
    label_map = {lig: i for i, lig in enumerate(lig_values)}
    num_classes = len(lig_values)
    assert num_classes == 7, f"Expected 7 unique lig values, got {num_classes}"

    input_dim = len(feature_cols)

    # Check for NaN/inf in features
    train_df = df[df["split"] == "train"]
    train_features = train_df[feature_cols].to_numpy(dtype=np.float32)
    
    print(f"Checking data quality...")
    print(f"  Feature shape: {train_features.shape}")
    print(f"  NaN count: {np.isnan(train_features).sum()}")
    print(f"  Inf count: {np.isinf(train_features).sum()}")
    print(f"  Feature min: {train_features.min():.4f}, max: {train_features.max():.4f}")
    print(f"  Feature mean: {train_features.mean():.4f}, std: {train_features.std():.4f}")
    
    # Replace NaN and Inf with 0 (or median/mean)
    if np.isnan(train_features).any() or np.isinf(train_features).any():
        print("  Warning: Found NaN/Inf values, replacing with 0")
        train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
        # Update dataframe
        for i, col in enumerate(feature_cols):
            df.loc[df["split"] == "train", col] = train_features[:, i]
    
    # Compute normalization statistics from training set only
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
    
    print(f"  Normalized feature mean: {feature_mean.mean():.4f}, std: {feature_std.mean():.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders with normalization
    train_loader = build_dataloader(df, feature_cols, label_map, split="train", batch_size=256, 
                                    feature_mean=feature_mean, feature_std=feature_std)
    val_loader = build_dataloader(df, feature_cols, label_map, split="val", batch_size=256,
                                 feature_mean=feature_mean, feature_std=feature_std)
    test_loader = build_dataloader(df, feature_cols, label_map, split="test", batch_size=256,
                                   feature_mean=feature_mean, feature_std=feature_std)

    # Learning rates to sweep
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    
    # Track best across all learning rates
    best_lr = None
    best_val_loss_overall = float("inf")
    best_state_dict_overall = None
    best_model_overall = None

    # Model: 3-layer MLP (2 hidden + output)
    hidden_dims = [256, 128]
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100

    # Sweep over learning rates
    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Training with learning rate: {lr}")
        print(f"{'='*60}")
        
        # Initialize fresh model for each learning rate
        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # Track best for this learning rate
        best_val_loss = float("inf")
        best_state_dict = None

        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0)
            val_loss = eval_epoch(model, val_loader, criterion, device)

            # Check for NaN and stop early if detected
            if np.isnan(train_loss) or np.isnan(val_loss) or np.isinf(train_loss) or np.isinf(val_loss):
                print(f"  Stopping early due to NaN/Inf loss at epoch {epoch}")
                break

            if val_loss < best_val_loss and not (np.isnan(val_loss) or np.isinf(val_loss)):
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        print(f"Best validation loss for lr={lr}: {best_val_loss:.4f}")

        # Check if this is the best learning rate so far (skip if NaN/Inf)
        if not (np.isnan(best_val_loss) or np.isinf(best_val_loss)) and best_val_loss < best_val_loss_overall:
            best_val_loss_overall = best_val_loss
            best_lr = lr
            best_state_dict_overall = best_state_dict
            # Load best weights into model for this learning rate
            model.load_state_dict(best_state_dict)
            best_model_overall = model

    print(f"\n{'='*60}")
    print(f"Best learning rate: {best_lr} with validation loss: {best_val_loss_overall:.4f}")
    print(f"{'='*60}\n")

    # Load best model weights
    if best_model_overall is not None:
        model = best_model_overall
    else:
        # Fallback (shouldn't happen)
        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
        model.load_state_dict(best_state_dict_overall)

    # Save final model (only for best learning rate)
    model_out_path = f"{OUT_DIR}/seed{seed}/dpocket_mlp_model.pt"
    predictions_out_path = f"{OUT_DIR}/seed{seed}/dpocket_mlp_predictions.parquet"
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(predictions_out_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "num_classes": num_classes,
            "feature_cols": feature_cols,
            "label_map": label_map,
            "best_learning_rate": best_lr,
            "best_val_loss": best_val_loss_overall,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "seed": seed,
        },
        model_out_path,
    )
    print(f"Saved model to {model_out_path}")

    # Collect predictions for the entire dataset (train/val/test combined)
    preds_df = collect_predictions(model, df, feature_cols, label_map, device,
                                   feature_mean=feature_mean, feature_std=feature_std)

    os.makedirs(os.path.dirname(predictions_out_path), exist_ok=True)
    preds_df.to_parquet(predictions_out_path)
    print(f"Saved predictions to {predictions_out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train D-Pocket MLP for ligand classification')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for training')
    args = parser.parse_args()
    main(seed=args.seed)
