"""Neuron-level evaluation metrics — the WORKFLOW §6.3 contract.

WORKFLOW §6.3 binding rules
---------------------------
- The label is neuron-level. Final metrics are computed at the **neuron
  level**, never at the row level.
- For long-row models: aggregate per-row predicted probabilities to per-neuron
  probability vectors (mean across the neuron's rows), then take argmax for
  the predicted label and compute metrics on those neuron-level predictions.
- Never aggregate hard labels across rows (mode-of-predictions is
  information-poor and unstable when probabilities are close).
- Mean-of-probabilities weights every row equally, so high-row-count neurons
  do not dominate per-neuron predictions; they are simply estimated more
  confidently.

This module is the single source of truth for those rules. Every modelling
notebook imports `neuron_level_score` from here.

Primary metric: balanced accuracy (handles class imbalance). Secondaries:
macro-F1, per-class recall, confusion matrix, and reference baselines
(majority-class, stratified-random) for calibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_probs_to_neuron(
    row_probs: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean per-row predicted probabilities into per-neuron probabilities.

    Parameters
    ----------
    row_probs : (n_rows, n_classes) float
    groups : (n_rows,) array of nucleus_id

    Returns
    -------
    neuron_ids : (n_neurons,) ordered array of unique nucleus_ids
    neuron_probs : (n_neurons, n_classes) float, ordered to match `neuron_ids`
    """
    if row_probs.ndim != 2:
        raise ValueError(f"row_probs must be 2-D, got {row_probs.shape}")
    if len(groups) != len(row_probs):
        raise ValueError("groups and row_probs must have the same length")

    df = pd.DataFrame(row_probs)
    df["__group__"] = groups
    out = df.groupby("__group__", sort=True).mean()
    return out.index.to_numpy(), out.to_numpy()


def aggregate_y_to_neuron(
    row_y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Take the per-neuron label as the first row's label (it is constant
    across the neuron's rows by construction; we just pick one)."""
    df = pd.DataFrame({"__y__": row_y, "__group__": groups})
    out = df.groupby("__group__", sort=True)["__y__"].first()
    return out.index.to_numpy(), out.to_numpy()


def aggregate_probs_hash_uniform(
    row_probs: np.ndarray,
    groups: np.ndarray,
    hashes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hash-uniform aggregation (planning §6.2).

    For E0:

        p_{i,h}(y) = (1 / R_h) * sum_r p(y | x_{i,h,r})       # avg trials within hash
        p_i(y)     = (1 / H_i) * sum_h p_{i,h}(y)             # avg hashes within neuron

    This corrects the voting imbalance that the row-uniform mean would
    introduce when some hashes have many more repeated trials than others.

    Parameters
    ----------
    row_probs : (n_rows, n_classes) per-row class probabilities.
    groups    : (n_rows,) nucleus_id per row.
    hashes    : (n_rows,) condition_hash per row.

    Returns
    -------
    neuron_ids   : (n_neurons,) sorted unique nucleus_ids.
    neuron_probs : (n_neurons, n_classes) hash-uniform neuron probabilities,
                   ordered to match `neuron_ids`.
    """
    if row_probs.ndim != 2:
        raise ValueError(f"row_probs must be 2-D, got {row_probs.shape}")
    if not (len(groups) == len(hashes) == len(row_probs)):
        raise ValueError("groups, hashes and row_probs must have the same length")

    df = pd.DataFrame(row_probs)
    df["__g__"] = groups
    df["__h__"] = hashes

    # Step 1: average within (group, hash) → one prob vector per (neuron, hash)
    by_hash = df.groupby(["__g__", "__h__"], sort=True).mean()
    # Step 2: average across hashes within group → one prob vector per neuron
    by_group = by_hash.groupby(level="__g__", sort=True).mean()
    return by_group.index.to_numpy(), by_group.to_numpy()


def neuron_level_score_hash_uniform(
    y_row_true: np.ndarray,
    y_row_proba: np.ndarray,
    groups: np.ndarray,
    hashes: np.ndarray,
    classes: np.ndarray,
) -> dict:
    """Hash-uniform analogue of `neuron_level_score` (planning §6.2).

    First averages probabilities within (neuron, hash), then within neuron;
    final neuron prediction is `argmax` on the hash-uniform probability vector.

    Returns the same keys as `neuron_level_score` so result rows can use the
    same downstream plumbing (`make_result_row`, parquet schemas, etc.).
    """
    classes = np.asarray(classes)
    nid_p, neuron_probs = aggregate_probs_hash_uniform(y_row_proba, groups, hashes)
    nid_y, y_neuron_true = aggregate_y_to_neuron(y_row_true, groups)
    if not np.array_equal(nid_p, nid_y):
        raise RuntimeError("group ordering mismatch between probs and labels")
    y_neuron_pred = classes[neuron_probs.argmax(axis=1)]

    cm = confusion_matrix(y_neuron_true, y_neuron_pred, labels=classes)
    per_class_recall = {}
    for i, c in enumerate(classes):
        denom = cm[i].sum()
        per_class_recall[str(c)] = float(cm[i, i] / denom) if denom > 0 else float("nan")

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_neuron_true, y_neuron_pred)),
        "macro_f1":          float(f1_score(y_neuron_true, y_neuron_pred,
                                            average="macro", labels=classes,
                                            zero_division=0)),
        "per_class_recall":  per_class_recall,
        "confusion_matrix":  cm,
        "n_neurons":         int(len(y_neuron_true)),
        "y_neuron_true":     y_neuron_true,
        "y_neuron_pred":     y_neuron_pred,
        "neuron_probs":      neuron_probs,
        "classes":           classes,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def neuron_level_score(
    y_row_true: np.ndarray,
    y_row_proba: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
) -> dict:
    """Aggregate row probabilities to neuron level, argmax to a label, and
    return the workflow's primary and secondary metrics in one dict.

    Parameters
    ----------
    y_row_true : (n_rows,) true class labels at the row level (will be
                 collapsed to neuron level via `aggregate_y_to_neuron`).
    y_row_proba : (n_rows, n_classes) per-row predicted probabilities.
    groups : (n_rows,) nucleus_id per row.
    classes : (n_classes,) array giving the class label for each column of
              `y_row_proba` (in the same order).

    Returns
    -------
    dict with keys:
      - balanced_accuracy : float (primary metric)
      - macro_f1          : float
      - per_class_recall  : dict[class -> recall]
      - confusion_matrix  : ndarray (n_classes, n_classes)
      - n_neurons         : int
      - y_neuron_true     : (n_neurons,) ndarray
      - y_neuron_pred     : (n_neurons,) ndarray
      - neuron_probs      : (n_neurons, n_classes) ndarray
      - classes           : the class label array
    """
    classes = np.asarray(classes)
    nid_p, neuron_probs = aggregate_probs_to_neuron(y_row_proba, groups)
    nid_y, y_neuron_true = aggregate_y_to_neuron(y_row_true, groups)
    if not np.array_equal(nid_p, nid_y):
        raise RuntimeError("group ordering mismatch between probs and labels")
    y_neuron_pred = classes[neuron_probs.argmax(axis=1)]

    cm = confusion_matrix(y_neuron_true, y_neuron_pred, labels=classes)
    per_class_recall = {}
    for i, c in enumerate(classes):
        denom = cm[i].sum()
        per_class_recall[str(c)] = float(cm[i, i] / denom) if denom > 0 else float("nan")

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_neuron_true, y_neuron_pred)),
        "macro_f1":          float(f1_score(y_neuron_true, y_neuron_pred, average="macro", labels=classes, zero_division=0)),
        "per_class_recall":  per_class_recall,
        "confusion_matrix":  cm,
        "n_neurons":         int(len(y_neuron_true)),
        "y_neuron_true":     y_neuron_true,
        "y_neuron_pred":     y_neuron_pred,
        "neuron_probs":      neuron_probs,
        "classes":           classes,
    }


# ---------------------------------------------------------------------------
# Calibration baselines
# ---------------------------------------------------------------------------
def majority_class_baseline(y_neuron_true: np.ndarray) -> dict:
    """Predict the most-frequent class for every neuron."""
    classes, counts = np.unique(y_neuron_true, return_counts=True)
    majority = classes[counts.argmax()]
    y_pred = np.full_like(y_neuron_true, majority)
    return {
        "name": "majority-class",
        "balanced_accuracy": float(balanced_accuracy_score(y_neuron_true, y_pred)),
        "macro_f1":          float(f1_score(y_neuron_true, y_pred, average="macro", zero_division=0)),
    }


def stratified_random_baseline(
    y_neuron_true: np.ndarray,
    seed: int = 42,
    n_repeats: int = 100,
) -> dict:
    """Sample predictions from the empirical class distribution; average
    `n_repeats` runs to suppress sampling variance."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_neuron_true, return_counts=True)
    p = counts / counts.sum()
    bal_accs, f1s = [], []
    for _ in range(n_repeats):
        y_pred = rng.choice(classes, size=len(y_neuron_true), p=p)
        bal_accs.append(balanced_accuracy_score(y_neuron_true, y_pred))
        f1s.append(f1_score(y_neuron_true, y_pred, average="macro", zero_division=0))
    return {
        "name": f"stratified-random ({n_repeats}× avg)",
        "balanced_accuracy": float(np.mean(bal_accs)),
        "macro_f1":          float(np.mean(f1s)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
        "macro_f1_std":          float(np.std(f1s)),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def summarize_cv_runs(per_fold: list[dict]) -> dict:
    """Mean ± std across folds for the headline metrics."""
    keys = ["balanced_accuracy", "macro_f1"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in per_fold])
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    # Per-class recall: average across folds.
    classes = list(per_fold[0]["per_class_recall"].keys())
    pcr = {}
    for c in classes:
        vals = np.array([r["per_class_recall"][c] for r in per_fold])
        pcr[c] = float(np.nanmean(vals))
    out["per_class_recall"] = pcr
    out["n_folds"] = len(per_fold)
    return out
