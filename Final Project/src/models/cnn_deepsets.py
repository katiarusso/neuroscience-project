"""Phase 2 Design B — DeepSets-style trace CNN for E1.

Phase 2 §6.3 (Design B):

    e_{i,h} = Encoder(x_{i,h})        # encode each oracle trace separately
    e_i     = Pool_h(e_{i,h})         # permutation-invariant pool across hashes
    p_i(y)  = Classifier(e_i)         # one prediction per neuron

The headline difference from Design A (`src/models/cnn.py`) is the **training
objective**: Design A trains row-level cross-entropy and averages probabilities
at evaluation; Design B trains one cross-entropy *per neuron* on a pooled
embedding, so the encoder is optimised directly for the neuron-level
representation. By construction, the model is permutation-invariant in the
hash dimension and never averages traces across different hashes — pooling
happens after encoding (§2.3, §6.3 of `PHASE2_PLANNING.md`).

Public API
----------
TraceCNNEncoder           — Conv1d × 2 + masked GAP, returns (N, embed_dim)
DeepSetsTraceCNN          — full Design-B model: forward(x_set, mask_set) → (B, n_classes)
build_e1_neuron_tensors   — pack a session E1 DataFrame into (N_neurons, H, T) tensors
train_design_b_fold       — train one fold with class-weighted CE + early stopping on val bal-acc
predict_neuron_proba      — softmax probabilities at the neuron level (no aggregation needed)

Notes
-----
* The trace encoder reuses the same backbone as `SmallTraceCNN` (Conv1d 1→32→D
  with kernel 5, BatchNorm, ReLU, Dropout, masked GAP) so the architecture
  difference vs. Design A is purely the training objective and the pooling
  location.
* `build_e1_neuron_tensors` requires every neuron to expose the **same set of
  hashes in the same order**, which holds for the oracle-116 subset on a
  balanced session. We assert this contract.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import (
    LAYER_CLASSES, LABEL2IDX,
    MaskedGlobalAvgPool, encode_labels,
    get_class_weights, get_class_balanced_weights,
)
from src.features.raw_traces import MAX_TRACE_LEN, get_padded_arrays


# ---------------------------------------------------------------------------
# Encoder + DeepSets head
# ---------------------------------------------------------------------------
class TraceCNNEncoder(nn.Module):
    """Conv1d → BN → ReLU → Dropout (× 2) followed by masked global average pool.

    Returns a per-trace embedding of dimension `embed_dim`. Architecture is
    intentionally identical to the convolutional body of `SmallTraceCNN` so any
    Design-A vs Design-B difference is attributable to the training objective
    and the pool location, not the encoder family.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
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
            nn.Conv1d(32, embed_dim, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = MaskedGlobalAvgPool()

    def forward(
        self,
        x: torch.Tensor,                              # (N, C, T)
        mask: Optional[torch.Tensor] = None,          # (N, 1, T) or None
    ) -> torch.Tensor:                                # (N, embed_dim)
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x, mask)


class DeepSetsTraceCNN(nn.Module):
    """Design B for E1: encode each of the H traces, pool across hashes, classify.

    Parameters
    ----------
    in_channels : 1 for calcium-only.
    n_classes   : 4 (L2/3, L4, L5, L6).
    embed_dim   : per-trace embedding dimension (= encoder output channels).
    head_hidden : hidden width of the classifier MLP (operates on pooled embedding).
    dropout     : dropout probability inside the encoder and the head.
    kernel_size : temporal convolution kernel.
    pool        : 'mean' (default) or 'attention' (gated tanh-attention pool).

    Shapes
    ------
    Input  x    : (B, H, T)        — B neurons, H = 116 hashes, T = 113 frames
    Input  mask : (B, H, T) | None — 1 for valid frames, 0 for padding
    Output      : (B, n_classes)   — logits, neuron-level
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 4,
        embed_dim: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.25,
        kernel_size: int = 5,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool not in ("mean", "attention"):
            raise ValueError(f"pool must be 'mean' or 'attention', got {pool!r}")
        self.encoder = TraceCNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.pool_kind = pool
        if pool == "attention":
            self.attn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, 1),
            )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,                              # (B, H, T) or (B, H, C, T)
        mask: Optional[torch.Tensor] = None,          # (B, H, T) or None
    ) -> torch.Tensor:                                # (B, n_classes)
        # Backward-compatible multichannel support: accept either (B, H, T) for
        # single-channel calcium (notebooks 02 / 03 / ...) or (B, H, C, T) for
        # multichannel (notebook 09: calcium + stimulus + behaviour). The mask
        # is always (B, H, T) — masking is a property of the time axis, not
        # channels — and is broadcast to all channels inside the encoder.
        if x.dim() == 3:
            B, H, T = x.shape
            x_flat = x.reshape(B * H, 1, T)
        elif x.dim() == 4:
            B, H, C, T = x.shape
            x_flat = x.reshape(B * H, C, T)
        else:
            raise ValueError(
                f"expected x of shape (B, H, T) or (B, H, C, T); got {tuple(x.shape)}"
            )
        m_flat = mask.reshape(B * H, 1, T) if mask is not None else None

        e_flat = self.encoder(x_flat, m_flat)          # (B*H, D)
        e = e_flat.reshape(B, H, -1)                    # (B, H, D)

        if self.pool_kind == "mean":
            e_neuron = e.mean(dim=1)                    # (B, D)
        else:                                           # 'attention'
            scores  = self.attn(e)                      # (B, H, 1)
            weights = torch.softmax(scores, dim=1)
            e_neuron = (e * weights).sum(dim=1)         # (B, D)

        return self.head(e_neuron)                      # (B, n_classes)


# ---------------------------------------------------------------------------
# Data preparation — pack E1 rows into (N_neurons, H, T) tensors
# ---------------------------------------------------------------------------
def build_e1_neuron_tensors(
    df: pd.DataFrame,
    pad_to: int = MAX_TRACE_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack an E1 long-format DataFrame into Design-B set-tensors.

    Every neuron must have the same set of `condition_hash` values, in the
    same number — the oracle-116 subset on a balanced session satisfies this.

    Returns
    -------
    X     : (N_neurons, H, T) float32
    mask  : (N_neurons, H, T) float32 — 1 valid, 0 padded
    y_str : (N_neurons,) layer label strings — same order as the unique
            sorted nucleus_ids
    nids  : (N_neurons,) int64        — sorted unique nucleus_ids
    """
    df_sorted = df.sort_values(["nucleus_id", "condition_hash"]).reset_index(drop=True)

    # Contract: every neuron has identical hash count
    hash_counts = df_sorted.groupby("nucleus_id")["condition_hash"].nunique()
    H_set = set(hash_counts.tolist())
    if len(H_set) != 1:
        raise ValueError(
            f"non-uniform hash coverage across neurons: counts seen = {sorted(H_set)}"
        )
    H = H_set.pop()

    # Sanity: hash ordering identical across neurons
    expected_hashes = (
        df_sorted[df_sorted["nucleus_id"] == df_sorted["nucleus_id"].iloc[0]]
        ["condition_hash"].tolist()
    )
    for nid, grp in df_sorted.groupby("nucleus_id"):
        if grp["condition_hash"].tolist() != expected_hashes:
            raise ValueError(f"hash ordering differs for nucleus_id={nid}")

    X_rows, mask_rows, _, _ = get_padded_arrays(df_sorted, pad_to=pad_to)
    n_rows = len(X_rows)
    if n_rows % H != 0:
        raise ValueError(f"n_rows={n_rows} not divisible by H={H}")
    N = n_rows // H

    X    = X_rows.reshape(N, H, pad_to).astype(np.float32)
    mask = mask_rows.reshape(N, H, pad_to).astype(np.float32)

    nids = (
        df_sorted.drop_duplicates("nucleus_id")
                 .sort_values("nucleus_id")["nucleus_id"]
                 .to_numpy(dtype=np.int64)
    )
    y_str = (
        df_sorted.drop_duplicates("nucleus_id")
                 .sort_values("nucleus_id")["layer_label"]
                 .to_numpy()
    )
    if len(nids) != N or len(y_str) != N:
        raise RuntimeError("internal packing mismatch")
    return X, mask, y_str, nids


def build_e1_neuron_tensors_multichannel(
    df: pd.DataFrame,
    behaviour_map: dict,
    channel_names: list,                       # ordered behaviour channels to stack after calcium
    pad_to: int = MAX_TRACE_LEN,
    hash_vocab: Optional[list] = None,
) -> tuple:
    """Multichannel analogue of `build_e1_neuron_tensors_with_hashes`.

    Stacks the calcium response with one or more behaviour channels along a
    new channel axis, producing `X` of shape `(N_neurons, H, C, T)` where
    `C = 1 + len(channel_names)` (channel 0 is always calcium).

    Behaviour channels come from `load_behaviour_channels_e1`'s output dict
    keyed by `(session_key, condition_hash)`. Each neuron's set is filled by
    looking up that neuron's session and broadcasting the per-(session, hash)
    behaviour signal over the calcium hash dimension. Neurons in the *same*
    session viewing the *same* hash receive identical multichannel input
    (excluding their own calcium); the model can only differentiate them via
    channel 0.

    Returns
    -------
    X         : (N, H, C, T)  float32  — channel 0 is calcium, channels 1+ are behaviour
    mask      : (N, H, T)     float32  — single mask shared across channels
    y_str     : (N,)          str      — layer labels
    nids      : (N,)          int64    — sorted unique nucleus_ids
    hash_ids  : (H,)          int64    — hash → integer indices
    hash_vocab: list[str]     canonical hash ordering
    channel_names_out : list[str] — actual channel ordering used; first entry is 'calcium'
    """
    # Reuse the calcium-only tensor build for shape and ordering invariants
    Xc, mask, y_str, nids, hash_ids, hash_vocab_out = build_e1_neuron_tensors_with_hashes(
        df, pad_to=pad_to, hash_vocab=hash_vocab,
    )
    N, H, T = Xc.shape

    # We need the session_key per neuron — pull it from df
    if "session_key" not in df.columns:
        raise ValueError("df must include a 'session_key' column")
    session_lookup = (
        df.drop_duplicates("nucleus_id")
          .set_index("nucleus_id")["session_key"]
          .to_dict()
    )
    session_of = np.array([session_lookup[int(n)] for n in nids])

    if not channel_names:
        # No behaviour channels — return calcium as a 4D tensor with C=1 for symmetry
        X_mc = Xc[:, :, None, :].astype(np.float32)        # (N, H, 1, T)
        return X_mc, mask, y_str, nids, hash_ids, hash_vocab_out, ["calcium"]

    # Build behaviour stack: for each neuron i in session s_i, for each hash h
    # (in canonical order), look up behaviour_map[(s_i, hash_str)][channel].
    # Assemble (N, H, n_extra_channels, T).
    inv_hash = {idx: h for h, idx in zip(hash_vocab_out, range(len(hash_vocab_out)))}
    hash_strs = [inv_hash[int(h)] for h in hash_ids]

    n_extra = len(channel_names)
    X_extra = np.zeros((N, H, n_extra, T), dtype=np.float32)
    n_missing = 0
    for i, sk in enumerate(session_of):
        sk_str = str(sk)        # cast np.str_ → str for dict lookup safety
        for h_idx, h_str in enumerate(hash_strs):
            entry = behaviour_map.get((sk_str, h_str))
            if entry is None:
                n_missing += 1
                continue
            for c_idx, ch_name in enumerate(channel_names):
                X_extra[i, h_idx, c_idx, :] = entry[ch_name]

    if n_missing > 0:
        # Filling with zero is consistent with masked GAP behaviour — but we
        # warn so the caller knows.
        print(f"WARNING: {n_missing} (neuron, hash) pairs had no behaviour entry; "
              f"those slots are zero-filled.")

    # Stack: calcium first, then behaviour channels (alphabetical order
    # determined by channel_names).
    X_calcium = Xc[:, :, None, :]                          # (N, H, 1, T)
    X_mc      = np.concatenate([X_calcium, X_extra], axis=2).astype(np.float32)
    out_names = ["calcium"] + list(channel_names)
    return X_mc, mask, y_str, nids, hash_ids, hash_vocab_out, out_names


def apply_control_to_set_tensor(
    X: np.ndarray,
    control: str,
    seed: int,
) -> np.ndarray:
    """Apply a Stage-2.1 trace control to a set-tensor of shape (N, H, T).

    Controls are applied **per trace** (i.e. independently for every (neuron,
    hash) cell), exactly as in Design A — that keeps Design A vs Design B
    comparisons fair.
    """
    if control == "identity":
        return X
    if X.ndim != 3:
        raise ValueError(f"expected (N, H, T), got {X.shape}")
    N, H, T = X.shape
    if control == "time_shuffled":
        rng = np.random.default_rng(seed)
        Xc = X.copy()
        # shuffle each trace in place
        flat = Xc.reshape(N * H, T)
        for i in range(N * H):
            rng.shuffle(flat[i])
        return Xc
    if control == "amplitude_only":
        scalars = X.mean(axis=2, keepdims=True)         # (N, H, 1) per-trace mean
        return np.broadcast_to(scalars, X.shape).astype(np.float32).copy()
    raise ValueError(f"unknown control: {control!r}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _make_neuron_dataset(
    X: np.ndarray,
    mask: np.ndarray,
    y_str: np.ndarray,
) -> TensorDataset:
    X_t = torch.from_numpy(X.astype(np.float32))      # (N, H, T)
    m_t = torch.from_numpy(mask.astype(np.float32))   # (N, H, T)
    y_t = torch.from_numpy(encode_labels(y_str))      # (N,)
    return TensorDataset(X_t, m_t, y_t)


def _val_balanced_acc(
    model: nn.Module,
    X_val: np.ndarray, mask_val: np.ndarray, y_val_str: np.ndarray,
    batch_size: int, device: torch.device,
) -> float:
    probs = predict_neuron_proba(
        model, X_val, mask=mask_val, batch_size=batch_size, device=device,
    )
    y_pred_idx = probs.argmax(axis=1)
    y_pred = np.array(LAYER_CLASSES)[y_pred_idx]
    return float(balanced_accuracy_score(y_val_str, y_pred))


def train_design_b_fold(
    X_train: np.ndarray, mask_train: np.ndarray, y_train_str: np.ndarray,
    X_val:   np.ndarray, mask_val:   np.ndarray, y_val_str:   np.ndarray,
    *,
    in_channels: int = 1,
    n_classes: int = 4,
    embed_dim: int = 64,
    head_hidden: int = 64,
    dropout: float = 0.25,
    kernel_size: int = 5,
    pool: str = "mean",
    batch_size: int = 32,
    max_epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> tuple[DeepSetsTraceCNN, list[dict]]:
    """Train one Design-B fold with class-weighted CE and early stopping on val
    neuron-level balanced accuracy.

    `X_*` shape: (N_neurons, H, T). `y_*_str`: (N_neurons,) layer labels.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = get_class_weights(y_train_str, classes=LAYER_CLASSES, device=device)

    ds = _make_neuron_dataset(X_train, mask_train, y_train_str)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DeepSetsTraceCNN(
        in_channels=in_channels,
        n_classes=n_classes,
        embed_dim=embed_dim,
        head_hidden=head_hidden,
        dropout=dropout,
        kernel_size=kernel_size,
        pool=pool,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_acc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_acc = _val_balanced_acc(
            model, X_val, mask_val, y_val_str,
            batch_size=batch_size, device=device,
        )
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_bal_acc": val_acc})

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_neuron_proba(
    model: nn.Module,
    X: np.ndarray,                          # (N, H, T)
    mask: Optional[np.ndarray] = None,      # (N, H, T) or None
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return per-neuron softmax probabilities (N_neurons, n_classes).

    No row-level aggregation is needed — Design B already outputs one
    prediction per neuron.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X_t = torch.from_numpy(X.astype(np.float32))
    m_t = torch.from_numpy(mask.astype(np.float32)) if mask is not None else None

    out = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        mb = m_t[i:i + batch_size].to(device) if m_t is not None else None
        logits = model(xb, mb)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out)


# ---------------------------------------------------------------------------
# Convenience: neuron-level score wrapper
# ---------------------------------------------------------------------------
def neuron_score_design_b(
    y_neuron_true_str: np.ndarray,
    neuron_probs: np.ndarray,
    classes: Optional[np.ndarray] = None,
) -> dict:
    """Direct neuron-level scoring for Design B (no row aggregation).

    Mirrors the schema of `src.eval.metrics.neuron_level_score` so result rows
    can flow through the same `make_result_row` plumbing used for Design A.
    """
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, confusion_matrix,
    )
    if classes is None:
        classes = np.array(LAYER_CLASSES)

    y_pred = classes[neuron_probs.argmax(axis=1)]
    cm = confusion_matrix(y_neuron_true_str, y_pred, labels=classes)
    per_class_recall = {}
    for i, c in enumerate(classes):
        denom = cm[i].sum()
        per_class_recall[str(c)] = float(cm[i, i] / denom) if denom > 0 else float("nan")

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_neuron_true_str, y_pred)),
        "macro_f1":          float(f1_score(y_neuron_true_str, y_pred,
                                            average="macro", labels=classes,
                                            zero_division=0)),
        "per_class_recall":  per_class_recall,
        "confusion_matrix":  cm,
        "n_neurons":         int(len(y_neuron_true_str)),
        "y_neuron_true":     y_neuron_true_str,
        "y_neuron_pred":     y_pred,
        "neuron_probs":      neuron_probs,
        "classes":           classes,
    }


# ===========================================================================
# Stage 2.2 — Hash-aware Design B
# ===========================================================================
#
# Architecture (planning §Stage 2.2a):
#
#     e_{i,h}^trace = Encoder_CNN(x_{i,h})           # per-trace embedding
#     e_h^hash      = Embedding(condition_hash_h)    # learned per-hash embedding
#     z_{i,h}       = [e_{i,h}^trace, e_h^hash]      # concat
#     u_{i,h}       = MLP(z_{i,h})                   # stimulus-aware row embedding
#     u_i           = Pool_h(u_{i,h})                # mean (or attention) over hashes
#     p_i(y)        = Classifier(u_i)
#
# Notes
# -----
# * The hash embedding table is shared across neurons. A pure hash-only model
#   would be degenerate at the neuron level (every neuron would receive the
#   same pooled representation), so the trace must do real per-neuron work for
#   any improvement to materialise — this is by design (planning §Stage 2.2c
#   diagnostic).
# * GroupKFold splits by `nucleus_id`, never by `condition_hash`. Every oracle
#   hash appears in train and in held-out test (with disjoint neurons), so the
#   hash table generalises across folds as expected. This is correct, not
#   leakage — the held-out unit is the neuron.


class HashAwareDeepSetsTraceCNN(nn.Module):
    """Hash-aware Design B (planning §Stage 2.2a).

    Forward signature differs from `DeepSetsTraceCNN`: it additionally takes a
    per-hash integer index tensor `hash_ids` of shape `(H,)` (shared across the
    batch — hash ordering is consistent across neurons after sorting).

    Parameters
    ----------
    n_hashes      : size of the hash vocabulary (e.g. 116 oracle hashes).
    in_channels   : 1 for calcium-only.
    n_classes     : 4 (L2/3, L4, L5, L6).
    trace_embed   : trace encoder output dimension.
    hash_embed    : hash embedding dimension (default 16).
    row_hidden    : hidden width of the per-row MLP that fuses trace + hash.
    head_hidden   : hidden width of the neuron-level classifier.
    dropout       : dropout in the encoder, the row MLP, and the head.
    kernel_size   : temporal convolution kernel.
    pool          : 'mean' (Stage 2.2a) or 'attention' (Stage 2.2b).
    """

    def __init__(
        self,
        n_hashes: int,
        in_channels: int = 1,
        n_classes: int = 4,
        trace_embed: int = 64,
        hash_embed: int = 16,
        row_hidden: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.25,
        kernel_size: int = 5,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool not in ("mean", "attention"):
            raise ValueError(f"pool must be 'mean' or 'attention', got {pool!r}")

        self.encoder = TraceCNNEncoder(
            in_channels=in_channels,
            embed_dim=trace_embed,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.hash_table = nn.Embedding(num_embeddings=n_hashes, embedding_dim=hash_embed)
        self.row_mlp = nn.Sequential(
            nn.Linear(trace_embed + hash_embed, row_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(row_hidden, row_hidden),
        )
        self.pool_kind = pool
        if pool == "attention":
            self.attn = nn.Sequential(
                nn.Linear(row_hidden, row_hidden),
                nn.Tanh(),
                nn.Linear(row_hidden, 1),
            )
        self.head = nn.Sequential(
            nn.Linear(row_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,                                 # (B, H, T) or (B, H, C, T)
        hash_ids: torch.Tensor,                          # (H,) long; broadcast over B
        mask: Optional[torch.Tensor] = None,             # (B, H, T) or None
        return_attention: bool = False,                  # if True, also return pool weights
    ):
        """Forward.

        Accepts either:

        * `x` of shape `(B, H, T)`        — single-channel calcium input
          (notebooks 02 / 04 / 05 / 06 / 07 / 08).
        * `x` of shape `(B, H, C, T)`     — multichannel input (notebook 09+).
          The model must be instantiated with `in_channels=C` matching the
          channel count.

        The mask is always `(B, H, T)` regardless of channel count — masking is
        a property of the time axis, not the channels.

        Returns logits `(B, n_classes)` by default. If `return_attention=True`,
        returns `(logits, weights)` where `weights` is shape `(B, H)` (squeezed)
        for `pool='attention'`, or `None` for `pool='mean'`. Backward
        compatible: the training loop calls `forward(...)` without the flag
        and gets logits only.
        """
        if x.dim() == 3:
            B, H, T = x.shape
            x_flat = x.reshape(B * H, 1, T)
        elif x.dim() == 4:
            B, H, C, T = x.shape
            x_flat = x.reshape(B * H, C, T)
        else:
            raise ValueError(
                f"expected x of shape (B, H, T) or (B, H, C, T); got {tuple(x.shape)}"
            )
        if hash_ids.shape != (H,):
            raise ValueError(
                f"hash_ids must have shape (H,)=({H},); got {tuple(hash_ids.shape)}"
            )

        m_flat = mask.reshape(B * H, 1, T) if mask is not None else None
        e_trace = self.encoder(x_flat, m_flat)            # (B*H, D_trace)
        e_trace = e_trace.reshape(B, H, -1)               # (B, H, D_trace)

        e_hash = self.hash_table(hash_ids)                # (H, D_hash)
        e_hash = e_hash.unsqueeze(0).expand(B, -1, -1)    # (B, H, D_hash)

        z = torch.cat([e_trace, e_hash], dim=-1)          # (B, H, D_trace+D_hash)
        u = self.row_mlp(z)                                # (B, H, row_hidden)

        weights: Optional[torch.Tensor] = None
        if self.pool_kind == "mean":
            u_neuron = u.mean(dim=1)
        else:                                              # 'attention'
            scores  = self.attn(u)                         # (B, H, 1)
            weights = torch.softmax(scores, dim=1)         # (B, H, 1)
            u_neuron = (u * weights).sum(dim=1)            # (B, row_hidden)

        logits = self.head(u_neuron)                       # (B, n_classes)
        if return_attention:
            # Squeeze the trailing singleton so weights have shape (B, H)
            w_out = weights.squeeze(-1) if weights is not None else None
            return logits, w_out
        return logits


# ---------------------------------------------------------------------------
# Hash-aware data prep
# ---------------------------------------------------------------------------
def build_e1_neuron_tensors_with_hashes(
    df: pd.DataFrame,
    pad_to: int = MAX_TRACE_LEN,
    hash_vocab: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Like `build_e1_neuron_tensors` but also returns the per-hash integer
    indices and the hash vocabulary used.

    Hash ordering is consistent across neurons (we sort by
    `(nucleus_id, condition_hash)`), so a single `(H,)` int64 array of hash
    indices applies to every neuron in the batch.

    Parameters
    ----------
    df         : E1 long-format DataFrame.
    pad_to     : pad length (default MAX_TRACE_LEN = 113).
    hash_vocab : optional list giving the canonical hash → index mapping.
                 Useful when the same vocabulary must be shared between train
                 and test (it always is here because every fold contains every
                 hash). If None, the vocabulary is built from this DataFrame
                 in sorted-string order.

    Returns
    -------
    X         : (N_neurons, H, T) float32
    mask      : (N_neurons, H, T) float32
    y_str     : (N_neurons,) str labels
    nids      : (N_neurons,) int64 nucleus_ids
    hash_ids  : (H,)       int64 hash indices into `hash_vocab`
    hash_vocab: list[str]  the canonical ordering used for `hash_ids`
    """
    df_sorted = df.sort_values(["nucleus_id", "condition_hash"]).reset_index(drop=True)

    hash_counts = df_sorted.groupby("nucleus_id")["condition_hash"].nunique()
    H_set = set(hash_counts.tolist())
    if len(H_set) != 1:
        raise ValueError(
            f"non-uniform hash coverage across neurons: counts seen = {sorted(H_set)}"
        )
    H = H_set.pop()

    # Hash ordering identical across neurons (after the sort)
    expected_hashes = (
        df_sorted[df_sorted["nucleus_id"] == df_sorted["nucleus_id"].iloc[0]]
        ["condition_hash"].tolist()
    )
    for nid, grp in df_sorted.groupby("nucleus_id"):
        if grp["condition_hash"].tolist() != expected_hashes:
            raise ValueError(f"hash ordering differs for nucleus_id={nid}")

    # Hash vocabulary
    if hash_vocab is None:
        hash_vocab = sorted(set(expected_hashes))
    hash_to_idx = {h: i for i, h in enumerate(hash_vocab)}
    if not set(expected_hashes).issubset(hash_to_idx):
        missing = set(expected_hashes) - set(hash_to_idx)
        raise ValueError(f"hash_vocab is missing {len(missing)} hashes; first: {next(iter(missing))!r}")

    hash_ids = np.array([hash_to_idx[h] for h in expected_hashes], dtype=np.int64)

    X_rows, mask_rows, _, _ = get_padded_arrays(df_sorted, pad_to=pad_to)
    n_rows = len(X_rows)
    if n_rows % H != 0:
        raise ValueError(f"n_rows={n_rows} not divisible by H={H}")
    N = n_rows // H

    X    = X_rows.reshape(N, H, pad_to).astype(np.float32)
    mask = mask_rows.reshape(N, H, pad_to).astype(np.float32)
    nids = (
        df_sorted.drop_duplicates("nucleus_id")
                 .sort_values("nucleus_id")["nucleus_id"]
                 .to_numpy(dtype=np.int64)
    )
    y_str = (
        df_sorted.drop_duplicates("nucleus_id")
                 .sort_values("nucleus_id")["layer_label"]
                 .to_numpy()
    )
    return X, mask, y_str, nids, hash_ids, hash_vocab


# ---------------------------------------------------------------------------
# Hash-aware training / inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_neuron_proba_hash(
    model: nn.Module,
    X: np.ndarray,
    hash_ids: np.ndarray,
    mask: Optional[np.ndarray] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Inference for `HashAwareDeepSetsTraceCNN`. Returns (N_neurons, n_classes)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X_t  = torch.from_numpy(X.astype(np.float32))
    h_t  = torch.from_numpy(hash_ids.astype(np.int64)).to(device)
    m_t  = torch.from_numpy(mask.astype(np.float32)) if mask is not None else None

    out = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        mb = m_t[i:i + batch_size].to(device) if m_t is not None else None
        logits = model(xb, h_t, mb)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def predict_neuron_proba_hash_with_attention(
    model: "HashAwareDeepSetsTraceCNN",
    X: np.ndarray,
    hash_ids: np.ndarray,
    mask: Optional[np.ndarray] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Inference returning both probabilities and pool attention weights.

    Returns
    -------
    probs   : (N_neurons, n_classes) softmax probabilities.
    weights : (N_neurons, H) attention weights for pool='attention',
              or `None` for pool='mean' (no per-hash weighting exists).

    Used by Stage 2.2 diagnostics (notebook 05) to reason about which oracle
    hashes the attention pool concentrates on as a function of the true layer.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X_t = torch.from_numpy(X.astype(np.float32))
    h_t = torch.from_numpy(hash_ids.astype(np.int64)).to(device)
    m_t = torch.from_numpy(mask.astype(np.float32)) if mask is not None else None

    probs_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        mb = m_t[i:i + batch_size].to(device) if m_t is not None else None
        logits, w = model(xb, h_t, mb, return_attention=True)
        probs_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        if w is not None:
            weight_chunks.append(w.cpu().numpy())

    probs = np.vstack(probs_chunks)
    weights = np.vstack(weight_chunks) if weight_chunks else None
    return probs, weights


def train_design_b_hash_fold(
    X_train: np.ndarray, mask_train: np.ndarray, y_train_str: np.ndarray,
    X_val:   np.ndarray, mask_val:   np.ndarray, y_val_str:   np.ndarray,
    hash_ids: np.ndarray,
    *,
    n_hashes: int,
    in_channels: int = 1,
    n_classes: int = 4,
    trace_embed: int = 64,
    hash_embed: int = 16,
    row_hidden: int = 64,
    head_hidden: int = 64,
    dropout: float = 0.25,
    kernel_size: int = 5,
    pool: str = "mean",
    batch_size: int = 32,
    max_epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    # ----- loss configuration (added at Stage 2.2 follow-up) -----
    loss_kind: str = "inverse_freq",         # 'inverse_freq' | 'class_balanced'
    cb_beta:   float = 0.999,                # used iff loss_kind=='class_balanced'
) -> tuple[HashAwareDeepSetsTraceCNN, list[dict]]:
    """Train one hash-aware Design B fold with class-weighted CE and early
    stopping on val neuron-level balanced accuracy.

    Same overall protocol as `train_design_b_fold`; the only architectural
    difference is the model class and the addition of `hash_ids` (a length-H
    int64 array, shared across the batch).

    `loss_kind` selects the class-weighting scheme:
        'inverse_freq'     — n_total / (n_classes * n_c)   (original default)
        'class_balanced'   — Cui et al. 2019, beta=`cb_beta` (default 0.999)
    Defaults preserve the existing behaviour of all earlier callers
    (notebooks 02 / 04 / 05) bit-for-bit.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loss_kind == "inverse_freq":
        class_weights = get_class_weights(
            y_train_str, classes=LAYER_CLASSES, device=device,
        )
    elif loss_kind == "class_balanced":
        class_weights = get_class_balanced_weights(
            y_train_str, classes=LAYER_CLASSES, beta=cb_beta, device=device,
        )
    else:
        raise ValueError(f"unknown loss_kind: {loss_kind!r}; "
                         f"expected 'inverse_freq' or 'class_balanced'")

    ds = _make_neuron_dataset(X_train, mask_train, y_train_str)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = HashAwareDeepSetsTraceCNN(
        n_hashes=n_hashes,
        in_channels=in_channels,
        n_classes=n_classes,
        trace_embed=trace_embed,
        hash_embed=hash_embed,
        row_hidden=row_hidden,
        head_hidden=head_hidden,
        dropout=dropout,
        kernel_size=kernel_size,
        pool=pool,
    ).to(device)

    h_t = torch.from_numpy(hash_ids.astype(np.int64)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_acc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, h_t, mb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Val
        probs_v = predict_neuron_proba_hash(
            model, X_val, hash_ids, mask=mask_val,
            batch_size=batch_size, device=device,
        )
        y_pred_v = np.array(LAYER_CLASSES)[probs_v.argmax(axis=1)]
        val_acc = float(balanced_accuracy_score(y_val_str, y_pred_v))
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_bal_acc": val_acc})

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
