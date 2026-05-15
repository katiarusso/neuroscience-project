"""Deep Sets (Zaheer et al. 2017) for Phase-1 Strategy 3.

A neuron is a *set* of rows (its A1 hash rows or A0 trial rows). Set order
is arbitrary — we want a function from set to layer probability that is
permutation-invariant. Deep Sets achieves this by:

    1. Per-row encoder φ: ℝᶠ → ℝᵈ (small MLP).
    2. Symmetric pooling ρ: mean across rows (with masking for variable set sizes).
    3. Classifier head ψ: ℝᵈ → ℝᴋ (logits over layer classes).

The full model is f(S) = ψ(ρ({φ(s) : s ∈ S})) — invariant by construction.

Why mean-pool and not sum: mean pooling is set-size-invariant (a neuron
with 280 rows produces the same-magnitude pooled embedding as one with 250
rows). For our problem all neurons see exactly 280 unique hashes, so the
distinction doesn't matter much, but mean-pool is the safer default.

Variable-size handling: pad each neuron's set to `max_set_size` and use a
mask tensor to zero out padding before pooling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSets(nn.Module):
    """A small Deep Sets classifier.

    Parameters
    ----------
    n_features  : per-row feature dimension (Tier A: 20).
    n_classes   : number of layer classes (4 for Phase 1).
    enc_hidden  : encoder hidden width (default 64).
    head_hidden : classifier head hidden width (default 64).
    dropout     : dropout probability between layers (default 0.1).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        enc_hidden: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, enc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, enc_hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(enc_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, set_features: torch.Tensor, set_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        set_features : (B, N_max, F) float tensor — per-neuron padded sets.
        set_mask : (B, N_max) float tensor — 1.0 for real rows, 0.0 for padding.

        Returns logits (B, n_classes).
        """
        h = self.encoder(set_features)            # (B, N_max, enc_hidden)
        h = h * set_mask.unsqueeze(-1)            # zero out padding
        denom = set_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_pooled = h.sum(dim=1) / denom           # mean-pool with mask
        return self.head(h_pooled)                # (B, n_classes)


def train_deep_sets(
    model: DeepSets,
    X_train: torch.Tensor, M_train: torch.Tensor, y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    M_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> dict:
    """Train one DeepSets model. Returns a small history dict.

    Class weights (one per class) are passed to `nn.CrossEntropyLoss` to
    handle the L2/3 ≫ L6 imbalance.
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    X_train = X_train.to(device); M_train = M_train.to(device); y_train = y_train.to(device)
    if X_val is not None:
        X_val = X_val.to(device); M_val = M_val.to(device); y_val = y_val.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    n = X_train.shape[0]
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, mb, yb = X_train[idx], M_train[idx], y_train[idx]
            optim.zero_grad()
            logits = model(xb, mb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        history["train_loss"].append(epoch_loss)

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                logits_v = model(X_val, M_val)
                val_loss = loss_fn(logits_v, y_val).item()
                val_pred = logits_v.argmax(dim=-1)
                val_acc = (val_pred == y_val).float().mean().item()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if verbose:
                print(f"  epoch {epoch:>2}: train_loss={epoch_loss:.3f} | val_loss={val_loss:.3f} | val_acc={val_acc:.3f}")
        elif verbose:
            print(f"  epoch {epoch:>2}: train_loss={epoch_loss:.3f}")

    return history
