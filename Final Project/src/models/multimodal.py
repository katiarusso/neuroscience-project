"""Phase 2 multichannel CNN model.

Extends SmallTraceCNN to arbitrary C input channels.  Used for:
  - Model B: calcium + stimulus temporal channels
  - Model C: calcium + behaviour temporal channels
  - Model D: calcium + stimulus + behaviour (full multichannel)

Channel construction helpers
-----------------------------
For E1 (trial-averaged data), stimulus temporal channels can be extracted from
stimulus metadata.  Behaviour channels are trial-averaged behaviour signals.

Channel layout convention (channel-first, PyTorch standard)
-----------------------------------------------------------
  channel 0 : calcium trace (always first)
  channel 1+: additional channels in order defined by the caller

Helper functions
----------------
stack_channels(calcium_X, *extra_Xs) → X_mc (n, C, T)
    Concatenate any number of (n, T) arrays along a new channel axis.

Public API
----------
MultiChannelTraceCNN   — same as SmallTraceCNN but accepts C channels
build_mc_arrays()      — convenience wrapper that returns (X_mc, y_str, groups)
                         from a dict of {channel_name: (n, T) array}
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.cnn import (
    LAYER_CLASSES,
    LABEL2IDX,
    IDX2LABEL,
    SmallTraceCNN,
    get_class_weights,
    predict_proba,
    train_fold,
)

__all__ = [
    "MultiChannelTraceCNN",
    "stack_channels",
    "build_mc_arrays",
    "channel_ablation_mask",
]


# ---------------------------------------------------------------------------
# MultiChannelTraceCNN
# ---------------------------------------------------------------------------
class MultiChannelTraceCNN(SmallTraceCNN):
    """Multichannel 1-D CNN — identical architecture to SmallTraceCNN.

    SmallTraceCNN already accepts arbitrary `in_channels`, so this class is
    a named alias with a descriptive docstring.  It exists so notebooks can
    import `MultiChannelTraceCNN` to signal intent clearly.

    Parameters
    ----------
    in_channels : int  — total number of channels (1 = calcium only,
                         2+ = calcium + additional channels)
    n_classes   : int  — number of output classes
    dropout     : float
    kernel_size : int
    """
    pass  # inherits everything from SmallTraceCNN


# ---------------------------------------------------------------------------
# Channel stacking utilities
# ---------------------------------------------------------------------------
def stack_channels(*channel_arrays: np.ndarray) -> np.ndarray:
    """Stack (n, T) arrays along a new channel axis → (n, C, T).

    Parameters
    ----------
    *channel_arrays : any number of (n, T) float32 arrays, all same shape.

    Returns
    -------
    X_mc : (n, C, T) float32 ndarray
    """
    if not channel_arrays:
        raise ValueError("At least one channel array required")
    shapes = [a.shape for a in channel_arrays]
    if len(set(shapes)) > 1:
        raise ValueError(f"All channel arrays must have the same shape. Got: {shapes}")
    return np.stack(channel_arrays, axis=1).astype(np.float32)  # (n, C, T)


def build_mc_arrays(
    channel_dict: dict[str, np.ndarray],
    y_str: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build a multichannel (n, C, T) array from a dict of named channels.

    The `calcium` key (if present) is placed first; remaining keys are sorted
    alphabetically for reproducibility.

    Parameters
    ----------
    channel_dict : dict mapping channel name → (n, T) float32 array.
                   Must include at least 'calcium'.
    y_str        : (n,) layer labels
    groups       : (n,) nucleus_ids

    Returns
    -------
    X_mc         : (n, C, T) float32
    y_str        : unchanged (returned for convenience)
    groups       : unchanged
    channel_names: list of channel names in the order they appear in X_mc
    """
    ordered_names = []
    if "calcium" in channel_dict:
        ordered_names.append("calcium")
    ordered_names += sorted(k for k in channel_dict if k != "calcium")

    arrays = [channel_dict[k] for k in ordered_names]
    X_mc = stack_channels(*arrays)
    return X_mc, y_str, groups, ordered_names


# ---------------------------------------------------------------------------
# Channel ablation helper
# ---------------------------------------------------------------------------
def channel_ablation_mask(
    X_mc: np.ndarray,
    keep_channels: list[int],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Return a copy of X_mc with all channels NOT in keep_channels zeroed out.

    Used for channel importance ablation: set unwanted channels to the fill
    value (default 0, which is the zero-centred baseline for normalized traces)
    and observe performance change.

    Parameters
    ----------
    X_mc          : (n, C, T) float32
    keep_channels : list of channel indices to keep (0-indexed)
    fill_value    : value to substitute for ablated channels

    Returns
    -------
    X_ablated : (n, C, T) float32 — copy with ablated channels zeroed
    """
    X_abl = X_mc.copy()
    n_channels = X_mc.shape[1]
    for c in range(n_channels):
        if c not in keep_channels:
            X_abl[:, c, :] = fill_value
    return X_abl


# ---------------------------------------------------------------------------
# Convenience: re-export train_fold / predict_proba for multichannel use
# ---------------------------------------------------------------------------
# Multichannel training uses the same train_fold() from cnn.py.
# Caller passes X_train shaped (n, C, T) — train_fold detects ndim==3 and
# skips the channel-dim-unsqueeze step.
train_mc_fold = train_fold
predict_mc_proba = predict_proba
