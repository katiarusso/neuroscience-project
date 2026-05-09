"""Phase 2 CNN evaluation runners.

Implements the WORKFLOW §6.3 contract for CNN models:
  - GroupKFold-CV (5 folds, precomputed splits)
  - Leave-One-Session-Out (LOSO, 13 sessions)
  - Control runners (amplitude-only, time-shuffled)
  - Result saving to parquet

All functions operate family-by-family so family-specific CNNs are supported.
Neuron-level aggregation is always done by mean-probability (never hard vote).

Public API
----------
run_gkf_cv()           — 5-fold GKF evaluation for one model config / family
run_loso()             — LOSO evaluation for one model config / family
run_family_aggregate() — Combine per-family GKF results into one neuron-level score
make_result_row()      — Build a single result dict (for appending to results list)
save_results()         — Save list of result dicts to parquet
save_predictions()     — Save row-level predictions with nucleus_id, hash, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config import PROCESSED_RESULTS_DIR, PROCESSED_PREDICTIONS_DIR
from src.eval.metrics import (
    neuron_level_score,
    aggregate_probs_to_neuron,
    aggregate_y_to_neuron,
    summarize_cv_runs,
)
from src.features.raw_traces import (
    FAMILY_LENGTHS,
    build_e1_dataset,
    get_family_arrays,
    normalize_traces,
    split_train_test_rows,
    split_loso_rows,
    make_amplitude_only_traces,
    make_time_shuffled_traces,
)
from src.models.cnn import (
    LAYER_CLASSES,
    SmallTraceCNN,
    get_class_weights,
    train_fold,
    predict_proba,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _neuron_score_dict(
    y_row_true: np.ndarray,
    row_probs: np.ndarray,
    groups: np.ndarray,
) -> dict:
    return neuron_level_score(
        y_row_true, row_probs, groups, classes=np.array(LAYER_CLASSES)
    )


# ---------------------------------------------------------------------------
# GKF runner  — single family
# ---------------------------------------------------------------------------
def run_gkf_cv(
    df: pd.DataFrame,
    family: str,
    norm_method: str = "per_trace_robust",
    model_kwargs: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[dict], pd.DataFrame]:
    """Run 5-fold GroupKFold CV for one stimulus family.

    Parameters
    ----------
    df          : E1 DataFrame from build_e1_dataset()
    family      : 'Clip', 'Monet2', 'Trippy', or 'all'
    norm_method : normalization passed to normalize_traces()
    model_kwargs: kwargs forwarded to SmallTraceCNN (e.g. kernel_size=7)
    train_kwargs: kwargs forwarded to train_fold() (e.g. max_epochs=50)
    seed        : base random seed (fold i uses seed + i)

    Returns
    -------
    per_fold_scores : list of 5 neuron_level_score dicts
    pred_df         : DataFrame of row-level predictions with nucleus_id,
                      condition_hash, session_key, true_layer, prob_*, pred_row
    """
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    device = _get_device()

    per_fold_scores = []
    all_pred_rows = []

    for fold_k in range(5):
        val_fold = (fold_k + 1) % 5
        df_train, df_test, df_val = split_train_test_rows(df, fold_k, val_fold)

        df_train_n, df_test_n, df_val_n = normalize_traces(
            df_train, df_test, df_val, method=norm_method
        )

        X_train, y_train, g_train = get_family_arrays(df_train_n, family)
        X_test, y_test, g_test   = get_family_arrays(df_test_n,  family)
        X_val,  y_val,  g_val    = get_family_arrays(df_val_n,   family)

        if verbose:
            print(f"\n--- GKF fold {fold_k} | family={family} | "
                  f"train_rows={len(X_train)}, test_rows={len(X_test)}, "
                  f"val_rows={len(X_val)} ---")

        model, _ = train_fold(
            X_train, y_train, g_train,
            X_val,   y_val,   g_val,
            seed=seed + fold_k,
            device=device,
            verbose=verbose,
            **{**{"in_channels": 1}, **model_kwargs},
            **train_kwargs,
        )

        row_probs = predict_proba(model, X_test, device=device)
        fold_score = _neuron_score_dict(y_test, row_probs, g_test)
        per_fold_scores.append(fold_score)

        if verbose:
            ba = fold_score["balanced_accuracy"]
            f1 = fold_score["macro_f1"]
            print(f"  fold {fold_k} result → bal_acc={ba:.4f}, macro_f1={f1:.4f}")

        # collect row-level predictions
        sub_test = df_test_n if family == "all" else df_test_n[df_test_n["stim_type"] == family]
        sub_test = sub_test.copy().reset_index(drop=True)
        for cls_i, cls in enumerate(LAYER_CLASSES):
            sub_test[f"prob_{cls.replace('/', '')}"] = row_probs[:, cls_i]
        sub_test["pred_layer_row"] = np.array(LAYER_CLASSES)[row_probs.argmax(axis=1)]
        sub_test["gkf_test_fold"] = fold_k
        all_pred_rows.append(sub_test[
            ["nucleus_id", "condition_hash", "session_key", "layer_label",
             "prob_L23", "prob_L4", "prob_L5", "prob_L6",
             "pred_layer_row", "gkf_test_fold"]
        ])

    pred_df = pd.concat(all_pred_rows, ignore_index=True)
    return per_fold_scores, pred_df


# ---------------------------------------------------------------------------
# LOSO runner  — single family
# ---------------------------------------------------------------------------
def run_loso(
    df: pd.DataFrame,
    family: str,
    norm_method: str = "per_trace_robust",
    model_kwargs: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[dict], pd.DataFrame]:
    """Run Leave-One-Session-Out for one stimulus family.

    Returns
    -------
    per_session_scores : list of dicts (one per session), each is a
                         neuron_level_score dict with extra key 'session_key'
    pred_df            : row-level predictions with session info
    """
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    device = _get_device()

    sessions = sorted(df["loso_test_scan"].unique())
    per_session_scores = []
    all_pred_rows = []

    for i, test_sk in enumerate(sessions):
        df_train, df_test, df_val = split_loso_rows(df, test_sk, seed=seed + i)

        df_train_n, df_test_n, df_val_n = normalize_traces(
            df_train, df_test, df_val, method=norm_method
        )

        # Filter to family
        if family != "all":
            df_train_n = df_train_n[df_train_n["stim_type"] == family]
            df_test_n  = df_test_n [df_test_n ["stim_type"] == family]
            df_val_n   = df_val_n  [df_val_n  ["stim_type"] == family]

        if len(df_test_n) == 0:
            if verbose:
                print(f"  LOSO session {test_sk}: no test rows for family={family}, skipping")
            continue

        X_train, y_train, g_train = get_family_arrays(df_train_n, family)
        X_test,  y_test,  g_test  = get_family_arrays(df_test_n,  family)
        X_val,   y_val,   g_val   = get_family_arrays(df_val_n,   family)

        # Skip degenerate sessions (single class in test)
        n_classes_test = len(np.unique(y_test))
        if n_classes_test < 2:
            if verbose:
                print(f"  LOSO session {test_sk}: only {n_classes_test} class(es) in test, "
                      f"marking as degenerate")

        if verbose:
            print(f"\n--- LOSO {test_sk} ({i+1}/{len(sessions)}) | family={family} | "
                  f"train_rows={len(X_train)}, test_rows={len(X_test)} ---")

        model, _ = train_fold(
            X_train, y_train, g_train,
            X_val,   y_val,   g_val,
            seed=seed + i,
            device=device,
            verbose=verbose,
            **{**{"in_channels": 1}, **model_kwargs},
            **train_kwargs,
        )

        row_probs = predict_proba(model, X_test, device=device)
        sess_score = _neuron_score_dict(y_test, row_probs, g_test)
        sess_score["session_key"] = test_sk
        sess_score["n_classes_in_test"] = n_classes_test
        per_session_scores.append(sess_score)

        if verbose:
            ba = sess_score["balanced_accuracy"]
            print(f"  {test_sk} → bal_acc={ba:.4f}")

        sub_test = df_test_n.copy().reset_index(drop=True)
        for cls_i, cls in enumerate(LAYER_CLASSES):
            sub_test[f"prob_{cls.replace('/', '')}"] = row_probs[:, cls_i]
        sub_test["pred_layer_row"] = np.array(LAYER_CLASSES)[row_probs.argmax(axis=1)]
        sub_test["loso_test_session"] = test_sk
        all_pred_rows.append(sub_test[
            ["nucleus_id", "condition_hash", "session_key", "layer_label",
             "prob_L23", "prob_L4", "prob_L5", "prob_L6",
             "pred_layer_row", "loso_test_session"]
        ])

    pred_df = pd.concat(all_pred_rows, ignore_index=True) if all_pred_rows else pd.DataFrame()
    return per_session_scores, pred_df


# ---------------------------------------------------------------------------
# Family-aggregate neuron-level score
# ---------------------------------------------------------------------------
def aggregate_family_predictions(
    pred_dfs: dict[str, pd.DataFrame],
    split_col: str = "gkf_test_fold",
) -> dict:
    """Aggregate per-family row-level predictions to one neuron-level score.

    For each neuron, collect all its rows across all families, mean-average
    the softmax probabilities, then argmax.  This is the 'all-family
    aggregated' result.

    Parameters
    ----------
    pred_dfs : dict family -> prediction DataFrame (from run_gkf_cv or run_loso)
               Each must have columns: nucleus_id, layer_label, prob_L23, ..., prob_L6
    split_col: column to split on for per-fold/session reporting ('gkf_test_fold'
               for GKF, 'loso_test_session' for LOSO)

    Returns
    -------
    neuron_level_score dict (same format as metrics.neuron_level_score)
    """
    prob_cols = ["prob_L23", "prob_L4", "prob_L5", "prob_L6"]
    all_dfs = pd.concat(pred_dfs.values(), ignore_index=True)

    # Mean probabilities across all rows of the same neuron
    agg = all_dfs.groupby("nucleus_id").agg(
        layer_label=("layer_label", "first"),
        **{pc: (pc, "mean") for pc in prob_cols}
    ).reset_index()

    row_probs = agg[prob_cols].to_numpy()
    y_true = agg["layer_label"].to_numpy()
    y_pred = np.array(LAYER_CLASSES)[row_probs.argmax(axis=1)]

    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, confusion_matrix
    )
    classes = np.array(LAYER_CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    per_class_recall = {}
    for i, c in enumerate(classes):
        denom = cm[i].sum()
        per_class_recall[str(c)] = float(cm[i, i] / denom) if denom > 0 else float("nan")

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=classes, zero_division=0)),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
        "n_neurons": int(len(y_true)),
        "y_neuron_true": y_true,
        "y_neuron_pred": y_pred,
        "neuron_probs": row_probs,
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# Control runners
# ---------------------------------------------------------------------------
def run_control(
    df: pd.DataFrame,
    family: str,
    control: str,            # 'amplitude_only' | 'time_shuffled'
    norm_method: str = "per_trace_robust",
    model_kwargs: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
    amp_mode: str = "mean",  # for amplitude_only: 'mean', 'max', 'auc'
) -> tuple[list[dict], pd.DataFrame]:
    """Run GKF CV with a temporal control substitution.

    Applies the control *after* normalization (so the amplitude-only control
    uses the normalized trace's scalar, not the raw trace's).

    control='amplitude_only'
        Replace each trace with a constant trace carrying its mean/max/AUC.
        Tests whether temporal order is needed.

    control='time_shuffled'
        Shuffle timepoints within each trace before training.
        Preserves amplitude distribution but destroys temporal order.
    """
    if control not in ("amplitude_only", "time_shuffled"):
        raise ValueError(f"control must be 'amplitude_only' or 'time_shuffled', got {control!r}")

    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    device = _get_device()

    per_fold_scores = []
    all_pred_rows = []

    for fold_k in range(5):
        val_fold = (fold_k + 1) % 5
        df_train, df_test, df_val = split_train_test_rows(df, fold_k, val_fold)
        df_train_n, df_test_n, df_val_n = normalize_traces(
            df_train, df_test, df_val, method=norm_method
        )

        X_train, y_train, g_train = get_family_arrays(df_train_n, family)
        X_test,  y_test,  g_test  = get_family_arrays(df_test_n,  family)
        X_val,   y_val,   g_val   = get_family_arrays(df_val_n,   family)

        # Apply control
        if control == "amplitude_only":
            X_train = make_amplitude_only_traces(X_train, mode=amp_mode)
            X_test  = make_amplitude_only_traces(X_test,  mode=amp_mode)
            X_val   = make_amplitude_only_traces(X_val,   mode=amp_mode)
        elif control == "time_shuffled":
            X_train = make_time_shuffled_traces(X_train, seed=seed + fold_k)
            X_test  = make_time_shuffled_traces(X_test,  seed=seed + fold_k + 100)
            X_val   = make_time_shuffled_traces(X_val,   seed=seed + fold_k + 200)

        if verbose:
            print(f"\n--- Control={control} | fold {fold_k} | family={family} ---")

        model, _ = train_fold(
            X_train, y_train, g_train,
            X_val,   y_val,   g_val,
            seed=seed + fold_k,
            device=device,
            verbose=verbose,
            **{**{"in_channels": 1}, **model_kwargs},
            **train_kwargs,
        )

        row_probs = predict_proba(model, X_test, device=device)
        fold_score = _neuron_score_dict(y_test, row_probs, g_test)
        per_fold_scores.append(fold_score)

        if verbose:
            ba = fold_score["balanced_accuracy"]
            print(f"  fold {fold_k} control result → bal_acc={ba:.4f}")

        sub_test = (df_test_n if family == "all"
                    else df_test_n[df_test_n["stim_type"] == family]).copy().reset_index(drop=True)
        for cls_i, cls in enumerate(LAYER_CLASSES):
            sub_test[f"prob_{cls.replace('/', '')}"] = row_probs[:, cls_i]
        sub_test["pred_layer_row"] = np.array(LAYER_CLASSES)[row_probs.argmax(axis=1)]
        sub_test["gkf_test_fold"] = fold_k
        all_pred_rows.append(sub_test[
            ["nucleus_id", "condition_hash", "session_key", "layer_label",
             "prob_L23", "prob_L4", "prob_L5", "prob_L6",
             "pred_layer_row", "gkf_test_fold"]
        ])

    pred_df = pd.concat(all_pred_rows, ignore_index=True)
    return per_fold_scores, pred_df


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------
def make_result_row(
    model_name: str,
    input_mode: str,
    normalization: str,
    stimulus_family: str,
    split_protocol: str,            # 'GKF' or 'LOSO'
    fold_or_session: str,           # fold index (str) or session_key
    seed: int,
    score_dict: dict,
    notes: str = "",
) -> dict:
    """Build a flat result row dict compatible with save_results().

    score_dict is a neuron_level_score() output or similar.
    """
    pcr = score_dict.get("per_class_recall", {})
    return {
        "model_name":        model_name,
        "input_mode":        input_mode,
        "normalization":     normalization,
        "stimulus_family":   stimulus_family,
        "split_protocol":    split_protocol,
        "fold_or_session":   str(fold_or_session),
        "seed":              seed,
        "n_neurons":         score_dict.get("n_neurons", np.nan),
        "balanced_accuracy": score_dict.get("balanced_accuracy", np.nan),
        "macro_f1":          score_dict.get("macro_f1", np.nan),
        "recall_L23":        pcr.get("L2/3", np.nan),
        "recall_L4":         pcr.get("L4",   np.nan),
        "recall_L5":         pcr.get("L5",   np.nan),
        "recall_L6":         pcr.get("L6",   np.nan),
        "notes":             notes,
    }


def make_summary_result_rows(
    per_fold_or_session_scores: list[dict],
    model_name: str,
    input_mode: str,
    normalization: str,
    stimulus_family: str,
    split_protocol: str,
    seed: int,
    notes: str = "",
    session_keys: Optional[list[str]] = None,
) -> list[dict]:
    """Build one result row per fold/session + one summary (mean ± std) row."""
    rows = []
    for i, score in enumerate(per_fold_or_session_scores):
        fold_or_session = (session_keys[i] if session_keys
                           else score.get("session_key", str(i)))
        rows.append(make_result_row(
            model_name, input_mode, normalization, stimulus_family,
            split_protocol, fold_or_session, seed, score, notes
        ))

    # Summary row
    bas = [s["balanced_accuracy"] for s in per_fold_or_session_scores]
    f1s = [s["macro_f1"] for s in per_fold_or_session_scores]
    summary = {
        "model_name": model_name, "input_mode": input_mode,
        "normalization": normalization, "stimulus_family": stimulus_family,
        "split_protocol": split_protocol, "fold_or_session": "mean±std",
        "seed": seed,
        "n_neurons": np.mean([s.get("n_neurons", np.nan)
                               for s in per_fold_or_session_scores]),
        "balanced_accuracy": np.mean(bas),
        "macro_f1": np.mean(f1s),
        "recall_L23": np.nan, "recall_L4": np.nan,
        "recall_L5": np.nan,  "recall_L6": np.nan,
        "notes": f"{notes} | std_ba={np.std(bas):.4f} std_f1={np.std(f1s):.4f}",
    }
    rows.append(summary)
    return rows


def save_results(
    rows: list[dict],
    filename: str,
    results_dir: Path = PROCESSED_RESULTS_DIR,
) -> Path:
    """Append result rows to a parquet file (or create it).

    If the file already exists, new rows are appended and the file is
    re-written.  This allows incremental saving across notebook re-runs.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
    print(f"Saved {len(rows)} row(s) → {path}  (total: {len(combined)})")
    return path


def save_predictions(
    pred_df: pd.DataFrame,
    filename: str,
    predictions_dir: Path = PROCESSED_PREDICTIONS_DIR,
) -> Path:
    """Save row-level prediction DataFrame to parquet."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    path = predictions_dir / filename
    pred_df.to_parquet(path, index=False)
    print(f"Saved predictions → {path}  ({len(pred_df)} rows)")
    return path
