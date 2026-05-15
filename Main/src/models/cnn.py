"""Phase 2 CNN model: SmallTraceCNN and training utilities.

Architecture
------------
SmallTraceCNN:
    Conv1D(C → 32, k=5, pad='same') + BN + ReLU + Dropout
    Conv1D(32 → 64, k=5, pad='same') + BN + ReLU + Dropout
    GlobalAveragePooling (mean over time)
    Linear(64 → 32) + ReLU + Dropout
    Linear(32 → n_classes)     ← logits; softmax applied at evaluation time

Input shape : (batch, C, T)   C=1 for calcium-only, C>1 for multichannel
Output shape: (batch, n_classes) logits

Training contract
-----------------
- Class-weighted cross-entropy loss (weights computed on training neurons,
  not rows, to avoid over-weighting high-observation-count neurons).
- AdamW optimizer (lr=1e-3, weight_decay=1e-4).
- CosineAnnealingLR scheduler.
- Early stopping on validation neuron-level balanced accuracy (patience=7).
- Random seeds fixed for reproducibility.
- Train on rows, evaluate on neurons.

Public API
----------
SmallTraceCNN          — the PyTorch module
get_class_weights()    — compute balanced class weights from neuron labels
train_fold()           — train one fold / session split, returns best model
predict_proba()        — run inference, return (n_rows, n_classes) probabilities
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.eval.metrics import aggregate_probs_to_neuron, aggregate_y_to_neuron


# ---------------------------------------------------------------------------
# Label encoder helpers
# ---------------------------------------------------------------------------
LAYER_CLASSES = ["L2/3", "L4", "L5", "L6"]
LABEL2IDX: dict[str, int] = {c: i for i, c in enumerate(LAYER_CLASSES)}
IDX2LABEL: dict[int, str] = {i: c for c, i in LABEL2IDX.items()}


def encode_labels(y_str: np.ndarray) -> np.ndarray:
    """Map layer label strings to integer indices."""
    return np.array([LABEL2IDX[s] for s in y_str], dtype=np.int64)


def decode_labels(y_idx: np.ndarray) -> np.ndarray:
    """Map integer indices back to layer label strings."""
    return np.array([IDX2LABEL[int(i)] for i in y_idx])


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def get_class_weights(
    y_neuron_str: np.ndarray,
    classes: list[str] = LAYER_CLASSES,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute inverse-frequency class weights from per-neuron labels.

    weight[c] = n_neurons / (n_classes * n_neurons_in_class_c)

    This is computed on the unique set of training neurons so that neurons
    contributing many rows (e.g. those with all 116 oracle hashes) do not
    inflate the effective count.
    """
    n_classes = len(classes)
    weights = []
    total = len(y_neuron_str)
    for c in classes:
        n_c = int((y_neuron_str == c).sum())
        w = total / (n_classes * n_c) if n_c > 0 else 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_class_balanced_weights(
    y_neuron_str: np.ndarray,
    classes: list[str] = LAYER_CLASSES,
    beta: float = 0.999,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Class-balanced loss weights (Cui et al. 2019, "Class-Balanced Loss
    Based on Effective Number of Samples").

    Effective number of samples for class c with `n_c` examples:

        E_{n_c} = (1 - beta^{n_c}) / (1 - beta)

    Weight for class c (before normalisation):

        w_c ∝ 1 / E_{n_c} = (1 - beta) / (1 - beta^{n_c})

    The output is renormalised so that sum(weights) == n_classes, which keeps
    the average loss scale comparable to the inverse-frequency weighting used
    elsewhere in the codebase. This makes the two weightings directly
    interchangeable inside `train_*_fold` functions.

    Parameters
    ----------
    y_neuron_str : (N_neurons,) array of layer label strings — *training-fold
        only* per the planning rule that fold weights must not see the test
        fold (caller's responsibility, same as `get_class_weights`).
    beta : (default 0.999, paper's canonical value). beta = 0 reduces to
        uniform weighting; beta -> 1 approaches inverse-frequency.

    Notes
    -----
    For a 4-class problem with class counts ~ (151, 249, 189, 96) we get
    weight ratios roughly (1.0, 0.85, 0.92, 1.24) at beta=0.999, which is
    much milder than the inverse-frequency ratios (1.13, 0.69, 0.91, 1.78).
    The milder weighting reduces the over-prediction of rare classes (here
    L6) at the cost of less aggressive correction for under-represented
    classes (here L2/3).
    """
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0, 1); got {beta}")

    n_classes = len(classes)
    raw = np.empty(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        n_c = int((y_neuron_str == c).sum())
        if n_c == 0:
            raw[i] = 1.0
            continue
        raw[i] = (1.0 - beta) / (1.0 - beta ** n_c)

    # Renormalise so sum == n_classes, matching the convention of
    # `get_class_weights`, so the two functions are interchangeable.
    raw = raw * (n_classes / raw.sum())
    return torch.tensor(raw, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MaskedGlobalAvgPool(nn.Module):
    """Global average pooling that ignores zero-padded frames.

    Expects inputs (batch, C, T) and mask (batch, 1, T) where mask=1 for
    valid frames and mask=0 for padding.  If no mask is provided, falls back
    to standard mean over all T frames.
    """
    def forward(
        self,
        x: torch.Tensor,          # (B, C, T)
        mask: Optional[torch.Tensor] = None,  # (B, 1, T)
    ) -> torch.Tensor:            # (B, C)
        if mask is None:
            return x.mean(dim=2)
        # Expand mask to match channels, clamp denom to avoid /0
        m = mask.expand_as(x)                      # (B, C, T)
        denom = m.sum(dim=2).clamp(min=1.0)        # (B, C)
        return (x * m).sum(dim=2) / denom          # (B, C)


class SmallTraceCNN(nn.Module):
    """Shallow 1-D CNN for calcium trace classification.

    Parameters
    ----------
    in_channels : int   — number of input channels (C=1 for calcium-only)
    n_classes   : int   — number of output classes (4 for L2/3, L4, L5, L6)
    dropout     : float — dropout rate applied after each conv block and head
    kernel_size : int   — temporal convolution kernel size (default 5)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 4,
        dropout: float = 0.25,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = MaskedGlobalAvgPool()
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,                       # (batch, C, T)
        mask: Optional[torch.Tensor] = None,   # (batch, 1, T) or None
    ) -> torch.Tensor:                         # (batch, n_classes) logits
        x = self.block1(x)           # (batch, 32, T)
        x = self.block2(x)           # (batch, 64, T)
        x = self.pool(x, mask)       # (batch, 64)
        return self.head(x)          # (batch, n_classes)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _make_tensor_dataset(
    X: np.ndarray,                          # (n, T)
    y_str: np.ndarray,                      # (n,) layer labels
    groups: np.ndarray,                     # (n,) nucleus_ids
    device: torch.device,
    mask: Optional[np.ndarray] = None,      # (n, T) or None
) -> TensorDataset:
    """Pack numpy arrays into a TensorDataset (kept on CPU; move batch-wise).

    If mask is provided, it is stored as (n, 1, T) so each batch contains
    (X, y, groups, mask).  If mask is None the dataset is (X, y, groups) and
    the training loop passes mask=None to the model.
    """
    X_t = torch.from_numpy(X[:, None, :].astype(np.float32))  # (n,1,T)
    y_t = torch.from_numpy(encode_labels(y_str))               # (n,)
    g_t = torch.from_numpy(groups.astype(np.int64))            # (n,)
    if mask is not None:
        m_t = torch.from_numpy(mask[:, None, :].astype(np.float32))  # (n,1,T)
        return TensorDataset(X_t, y_t, g_t, m_t)
    return TensorDataset(X_t, y_t, g_t)


def _make_tensor_dataset_mc(
    X: np.ndarray,       # (n, C, T) already channel-first
    y_str: np.ndarray,
    groups: np.ndarray,
) -> TensorDataset:
    """For multichannel inputs already shaped (n, C, T)."""
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(encode_labels(y_str))
    g_t = torch.from_numpy(groups.astype(np.int64))
    return TensorDataset(X_t, y_t, g_t)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,                          # (n, T) or (n, C, T)
    mask: Optional[np.ndarray] = None,      # (n, T) validity mask, or None
    batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run the model and return softmax probabilities (n_rows, n_classes).

    X can be (n, T) for single-channel (channel dim added automatically)
    or (n, C, T) for multichannel.  mask is the padding validity mask from
    get_padded_arrays(); pass None for non-padded inputs.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if X.ndim == 2:
        X_t = torch.from_numpy(X[:, None, :].astype(np.float32))
    else:
        X_t = torch.from_numpy(X.astype(np.float32))

    mask_t = None
    if mask is not None:
        # shape (n, T) → (n, 1, T)
        mask_t = torch.from_numpy(mask[:, None, :].astype(np.float32))

    all_probs = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size].to(device)
        mb = mask_t[i : i + batch_size].to(device) if mask_t is not None else None
        logits = model(xb, mb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


# ---------------------------------------------------------------------------
# Neuron-level validation metric (used for early stopping)
# ---------------------------------------------------------------------------
def _val_balanced_acc(
    model: nn.Module,
    X_val: np.ndarray,
    y_val_str: np.ndarray,
    groups_val: np.ndarray,
    batch_size: int,
    device: torch.device,
    mask_val: Optional[np.ndarray] = None,   # (n_val, T) or None
) -> float:
    probs = predict_proba(model, X_val, mask=mask_val,
                          batch_size=batch_size, device=device)
    _, neuron_probs = aggregate_probs_to_neuron(probs, groups_val)
    _, y_neuron_true = aggregate_y_to_neuron(y_val_str, groups_val)
    y_neuron_pred = np.array(LAYER_CLASSES)[neuron_probs.argmax(axis=1)]
    return float(balanced_accuracy_score(y_neuron_true, y_neuron_pred))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_fold(
    X_train: np.ndarray,                    # (n_train, T)
    y_train_str: np.ndarray,                # (n_train,)
    groups_train: np.ndarray,               # (n_train,) nucleus_ids
    X_val: np.ndarray,                      # (n_val, T)
    y_val_str: np.ndarray,
    groups_val: np.ndarray,
    # ----- optional padding masks -------------------------------------------
    mask_train: Optional[np.ndarray] = None,  # (n_train, T) 1=valid, 0=pad
    mask_val:   Optional[np.ndarray] = None,  # (n_val,   T)
    # ----- model config -----
    in_channels: int = 1,
    n_classes: int = 4,
    dropout: float = 0.25,
    kernel_size: int = 5,
    # ----- training config -----
    batch_size: int = 256,
    max_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> tuple[SmallTraceCNN, list[dict]]:
    """Train SmallTraceCNN for one fold with early stopping.

    Class weights are computed on unique training neurons (not rows).

    Parameters
    ----------
    X_train, X_val  : (n, T) for single-channel or (n, C, T) multichannel.
                      Channel dimension is added automatically for 2-D input.
    y_*_str         : layer label strings ('L2/3', 'L4', 'L5', 'L6').
    groups_*        : nucleus_id per row (used only for neuron-level val metric).
    mask_train/val  : (n, T) float32 masks from get_padded_arrays().
                      Pass these so the model sees the same masked pooling
                      during training as at inference.  If None, standard mean
                      pooling is used (all frames treated as valid).

    Returns
    -------
    best_model : SmallTraceCNN — state dict at best validation epoch restored
    history    : list of dicts with 'epoch', 'train_loss', 'val_bal_acc'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- class weights on unique training neurons ----------------------------
    unique_nids = np.unique(groups_train)
    y_neuron_train = {
        nid: y_train_str[groups_train == nid][0] for nid in unique_nids
    }
    y_neuron_arr = np.array([y_neuron_train[n] for n in unique_nids])
    class_weights = get_class_weights(y_neuron_arr, device=device)

    # -- data loaders --------------------------------------------------------
    use_mask = mask_train is not None
    if X_train.ndim == 2:
        ds_train = _make_tensor_dataset(
            X_train, y_train_str, groups_train, device,
            mask=mask_train,
        )
    else:
        ds_train = _make_tensor_dataset_mc(X_train, y_train_str, groups_train)

    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    # -- model, loss, optimizer ----------------------------------------------
    model = SmallTraceCNN(
        in_channels=in_channels,
        n_classes=n_classes,
        dropout=dropout,
        kernel_size=kernel_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # -- training loop -------------------------------------------------------
    best_val_acc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            if use_mask:
                Xb, yb, _, mb = batch          # mb: (B, 1, T)
                Xb, yb, mb = (Xb.to(device), yb.to(device), mb.to(device))
            else:
                Xb, yb, _ = batch
                Xb, yb = Xb.to(device), yb.to(device)
                mb = None
            optimizer.zero_grad()
            logits = model(Xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_acc = _val_balanced_acc(
            model, X_val, y_val_str, groups_val, batch_size, device,
            mask_val=mask_val,
        )
        avg_loss = total_loss / max(n_batches, 1)
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_bal_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | loss {avg_loss:.4f} | "
                  f"val bal_acc {val_acc:.4f} | best {best_val_acc:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)
    return model, history
