"""Modelling-ready data loaders.

The contract of every modelling notebook from 04 onward:

  X, y, groups, folds, df = build_modeling_table(level, blocks, label)

where:
  - X        : 2-D numpy float array (rows × features)
  - y        : 1-D numpy array of class labels (strings)
  - groups   : 1-D numpy array of `nucleus_id` per row (for GroupKFold)
  - folds    : 1-D numpy array of `gkf_fold` per row (precomputed in nb 02)
  - df       : the underlying merged DataFrame, in case extra columns are
               needed downstream (cc_abs, session_key, pt_position_y, …)

`level` is `'A0'` or `'A1'`; `blocks` is a list of feature-block names —
either `['amp']`, `['shape']`, or `['amp', 'shape']` for the merged variant.

This module is written defensively: it never silently joins a feature row
to a different neuron, and never drops rows from a table without printing
how many were dropped.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_TABLES_DIR,
    PROCESSED_FEATURES_DIR,
    PROCESSED_SPLITS_DIR,
)


# ---------------------------------------------------------------------------
# Single-table loaders
# ---------------------------------------------------------------------------
def load_working_pop() -> pd.DataFrame:
    """Cleaned working population (8,895 V1 / excitatory / matched / best-only,
    L1 dropped) with the `layer_label`, `celltype_label`, and `session_key`
    columns added in notebook 02."""
    return pd.read_parquet(PROCESSED_TABLES_DIR / "units_working.parquet")


def load_splits() -> pd.DataFrame:
    """Per-`nucleus_id` CV assignments: `gkf_fold` (StratifiedGroupKFold(5)
    stratified by layer) and `loso_test_scan` (= the neuron's session_key)."""
    return pd.read_parquet(PROCESSED_SPLITS_DIR / "cv_assignments.parquet")


def load_feature_block(level: str, block: str) -> pd.DataFrame:
    """Load one feature parquet (e.g. `level='A1'`, `block='amp'`)."""
    if level not in ("A0", "A1"):
        raise ValueError(f"level must be 'A0' or 'A1', got {level!r}")
    if block not in ("amp", "shape"):
        raise ValueError(f"block must be 'amp' or 'shape', got {block!r}")
    return pd.read_parquet(PROCESSED_FEATURES_DIR / f"{level}_{block}.parquet")


def load_features(level: str, blocks: Sequence[str]) -> pd.DataFrame:
    """Load and merge one or more feature blocks at the requested level.

    All blocks share the same key columns (nucleus_id + identifiers); the
    feature columns are unioned. Returns one DataFrame.
    """
    blocks = list(blocks)
    if not blocks:
        raise ValueError("at least one block required")

    first = load_feature_block(level, blocks[0])
    if len(blocks) == 1:
        return first

    # Identify the keys vs. the features.
    feat_prefixes = ("amp_", "shape_")
    keys = [c for c in first.columns if not c.startswith(feat_prefixes)]
    out = first.copy()
    for b in blocks[1:]:
        right = load_feature_block(level, b)
        right_feats = [c for c in right.columns if c.startswith(feat_prefixes)]
        out = pd.merge(out, right[keys + right_feats], on=keys, how="inner",
                       validate="one_to_one")
    return out


# ---------------------------------------------------------------------------
# Modelling-table assembly
# ---------------------------------------------------------------------------
def build_modeling_table(
    level: str,
    blocks: Sequence[str],
    label: str = "layer_label",
    extra_cols: Iterable[str] = ("cc_abs", "pt_position_y", "session_key"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Assemble (X, y, groups, folds, df) for a Tier-A Stage-1 model run.

    Parameters
    ----------
    level : 'A0' or 'A1'
    blocks : list of {'amp', 'shape'}
    label : column from `units_working.parquet` to use as y
            ('layer_label' for Phase 1, 'celltype_label' for Phase 3).
    extra_cols : columns from `units_working.parquet` to also expose in `df`
                 (used by the §6.5 confound baselines).

    Returns
    -------
    X : (n_rows, n_features) float ndarray
    y : (n_rows,) ndarray of label strings
    groups : (n_rows,) ndarray of nucleus_id (int)
    folds : (n_rows,) int8 ndarray of gkf_fold (precomputed)
    df : the merged underlying DataFrame
    """
    feats = load_features(level, blocks)
    units = load_working_pop()
    splits = load_splits()

    # Pull only the columns we need from units, AND drop any that already
    # exist in feats — otherwise pandas renames the conflict to `<col>_x` and
    # `<col>_y` on the merge, and downstream cells looking up the plain name
    # break with a KeyError. `nucleus_id` is the merge key so it's kept.
    needed_cols = ["nucleus_id", label] + list(extra_cols)
    feats_cols = set(feats.columns)
    units_keep = ["nucleus_id"] + [c for c in needed_cols
                                   if c != "nucleus_id" and c not in feats_cols]
    units_sub = units[units_keep].copy()

    n_feats0 = len(feats)
    df = (feats
          .merge(units_sub, on="nucleus_id", how="inner", validate="many_to_one")
          .merge(splits[["nucleus_id", "gkf_fold", "loso_test_scan"]],
                 on="nucleus_id", how="inner", validate="many_to_one"))
    if len(df) != n_feats0:
        # This is informational only — best-only guarantees one neuron per
        # nucleus_id in `units_working`, but feature parquets may have more
        # rows (per trial / per hash) than `units_working` — so the inner
        # merge can drop rows whose `nucleus_id` is missing from the working
        # population (would happen only if a feature was computed for a
        # neuron that was later dropped, e.g. the L1 drop in notebook 02).
        n_dropped = n_feats0 - len(df)
        print(f"[build_modeling_table] dropped {n_dropped} feature rows whose "
              f"nucleus_id is not in units_working (typically L1 drops).")

    feat_prefixes = ("amp_", "shape_")
    feat_cols = [c for c in df.columns if c.startswith(feat_prefixes)]

    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df[label].to_numpy()
    groups = df["nucleus_id"].to_numpy()
    folds = df["gkf_fold"].to_numpy()
    return X, y, groups, folds, df


# ---------------------------------------------------------------------------
# Single-row-per-neuron sampler — used by Strategy 1 (1a, random row per neuron)
# ---------------------------------------------------------------------------
def sample_one_row_per_neuron(
    df: pd.DataFrame,
    seed: int,
    by: str = "nucleus_id",
) -> pd.DataFrame:
    """Pick exactly one row per `nucleus_id` (or other group) at random.

    Used to build Strategy 1's diagnostic baselines. Pass different `seed`
    values to assess stochastic variability of the resulting score (the
    workflow recommends multi-seed averaging for this strategy).
    """
    rng = np.random.default_rng(seed)
    return (df.groupby(by, group_keys=False)
              .apply(lambda g: g.iloc[rng.integers(0, len(g))]))
