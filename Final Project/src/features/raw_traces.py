"""Phase 2 raw-trace dataset builder.

Builds the E1 dataset: one row per (nucleus_id × condition_hash),
restricted to the 116 oracle hashes, carrying the within-hash trial-averaged
calcium trace (`response_avg`).

Key public functions
--------------------
load_oracle_hashes()
    Returns (oracle_hash_list, oracle_by_family_counts).

build_e1_dataset()
    Full E1 DataFrame: nucleus_id × oracle_hash with layer_label, fold info,
    stim_type, n_frames, n_trials, response_avg.

--- Single-session probe functions (Phase 2 step 1) ---

remove_mismatch_neurons(units_df)
    Drop neurons where EM layer_label disagrees with celltype_label layer.
    Returns cleaned units DataFrame + mismatch report dict.

find_balanced_session(units_df, n_top)
    Rank sessions by layer-balance entropy.  Returns sorted DataFrame.

build_single_session_e1(session_key, units_df)
    E1 dataset restricted to one session and clean neurons.

stratified_neuron_split(df, train_frac, val_frac, seed)
    Stratified train/val/test split by layer, at the neuron level (no leakage).

baseline_subtract_traces(df, n_baseline_frames)
    Subtract per-trace pre-stimulus baseline (first n frames).

get_padded_arrays(df, pad_to)
    Return (X, mask, y_str, groups) with all families padded to pad_to=113.
    mask shape (n, T): 1 for valid frames, 0 for padding.

--- Full multi-session functions ---

split_train_test_rows(df, fold_k)
    Returns (df_train_rows, df_test_rows) for GKF fold k.

split_loso_rows(df, test_session_key)
    Returns (df_train_rows, df_test_rows) for one LOSO session.

normalize_traces(df_train, df_test, method, session_col)
    Fits normalization on df_train, applies to both.  Returns (df_train_norm,
    df_test_norm) — copies, never in-place.

get_family_arrays(df, family)
    Returns (X, y_str, groups) numpy arrays for one stimulus family,
    ready for CNN input.  X shape: (n_rows, T).

Notes
-----
- Traces are stored as Python lists in `response_avg`; we convert to np.ndarray
  on export.
- All normalization stats are fit on the TRAINING rows only (no leakage).
- family='all' in get_family_arrays pads shorter families to max_T=113 with
  zeros and returns a boolean mask column for masked pooling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DIR,
    PROCESSED_TABLES_DIR,
    PROCESSED_SPLITS_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRACES_AVG_PATH: Path = PROCESSED_TABLES_DIR / "traces_avg.parquet"
_TRIALS_META_PATH: Path = PROCESSED_TABLES_DIR / "trials_meta.parquet"
_ORACLE_HASHES_PATH: Path = PROCESSED_DIR / "oracle_hashes.pkl"

#: Behaviour columns available in trials_meta — per-trial time-series at the
#: same temporal resolution as calcium response (Clip: 75 frames, Monet2/Trippy:
#: 113 frames).
BEHAVIOUR_CHANNELS = (
    "treadmill",
    "pupil_dilation",
    "pupil_pos_x",
    "pupil_pos_y",
    "pupil_aux",
)

FAMILY_LENGTHS: dict[str, int] = {
    "Clip": 75,
    "Monet2": 113,
    "Trippy": 113,
}
MAX_TRACE_LEN: int = 113  # pad target for 'all' mode
EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Oracle hash helpers
# ---------------------------------------------------------------------------
def load_oracle_hashes() -> tuple[list[str], dict[str, int]]:
    """Load oracle_hashes.pkl.

    Returns
    -------
    oracle_hashes : list of 116 condition_hash strings
    oracle_by_family : dict mapping family name -> count (int)
                       e.g. {'Clip': 96, 'Monet2': 10, 'Trippy': 10}
    """
    with open(_ORACLE_HASHES_PATH, "rb") as f:
        oh = pickle.load(f)
    return oh["oracle_hashes"], oh["oracle_by_family"]


# ---------------------------------------------------------------------------
# E1 dataset builder
# ---------------------------------------------------------------------------
def build_e1_dataset(
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the Phase 2 E1 long-format dataset.

    One row per (nucleus_id × condition_hash), restricted to the 116 oracle
    hashes, for the 8,895 best-only V1 excitatory neurons in the working
    population.

    Expected row count: ~8,895 × 116 = ~1,031,820 rows.

    Columns in output
    -----------------
    nucleus_id        int64   — structural EM identity (cross-session stable)
    condition_hash    str     — oracle stimulus hash
    session_key       str     — the neuron's session scan (from units_working)
    stim_type         str     — 'Clip', 'Monet2', or 'Trippy'
    layer             str     — raw layer string
    layer_label       str     — 'L2/3', 'L4', 'L5', 'L6'
    n_frames          int64   — trace length in frames
    n_trials          int64   — number of trials averaged into response_avg
    response_avg      object  — list[float], the trial-averaged trace
    gkf_fold          int8    — precomputed GroupKFold fold (0-4)
    loso_test_scan    str     — session_key of the held-out LOSO scan
    """
    oracle_hashes, oracle_by_family = load_oracle_hashes()
    oracle_set = set(oracle_hashes)

    # -- load working population (8,895 neurons, one row per nucleus_id) -----
    units = pd.read_parquet(PROCESSED_TABLES_DIR / "units_working.parquet",
                            columns=["nucleus_id", "layer", "layer_label",
                                     "session_key"])

    # -- load CV splits -------------------------------------------------------
    splits = pd.read_parquet(PROCESSED_SPLITS_DIR / "cv_assignments.parquet",
                             columns=["nucleus_id", "gkf_fold", "loso_test_scan"])

    # -- load traces_avg, filter to oracle hashes ----------------------------
    if verbose:
        print("Loading traces_avg.parquet …")

    traces = pd.read_parquet(_TRACES_AVG_PATH)

    if verbose:
        print(f"  raw rows: {len(traces):,}")

    traces = traces[traces["condition_hash"].isin(oracle_set)].copy()

    if verbose:
        print(f"  oracle-filtered rows: {len(traces):,}")

    # -- keep only the neuron's own session -----------------------------------
    # traces_avg may have rows for a nucleus_id from multiple sessions
    # (if the neuron appeared in more than one scan). We restrict to the
    # session recorded in units_working (the best-only session).
    session_map = units.set_index("nucleus_id")["session_key"].to_dict()
    traces = traces[
        traces.apply(
            lambda r: session_map.get(r["nucleus_id"]) == r["session_key"],
            axis=1,
        )
    ].copy()

    if verbose:
        print(f"  after best-only session filter: {len(traces):,}")

    # -- join layer labels and fold assignments --------------------------------
    df = (
        traces
        .merge(units[["nucleus_id", "layer", "layer_label"]],
               on="nucleus_id", how="inner")
        .merge(splits, on="nucleus_id", how="inner")
    )

    # rename session_key from traces (already filtered above)
    # units.session_key = traces.session_key by construction after filter
    if "session_key_x" in df.columns:
        df = df.rename(columns={"session_key_x": "session_key"}).drop(
            columns=["session_key_y"], errors="ignore"
        )

    # -- cast types -----------------------------------------------------------
    df["gkf_fold"] = df["gkf_fold"].astype("int8")
    df = df.reset_index(drop=True)

    if verbose:
        print(f"\nE1 dataset shape: {df.shape}")
        print(f"Neurons: {df['nucleus_id'].nunique():,}")
        print(f"Hashes : {df['condition_hash'].nunique():,}")
        print(f"\nRows by stimulus family:")
        print(df.groupby("stim_type")["nucleus_id"].count().to_string())
        print(f"\nNeurons by layer:")
        print(df.drop_duplicates("nucleus_id")["layer_label"].value_counts().sort_index().to_string())

    return df


# ---------------------------------------------------------------------------
# Train / test splitting helpers
# ---------------------------------------------------------------------------
def split_train_test_rows(
    df: pd.DataFrame,
    fold_k: int,
    val_fold: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Split E1 DataFrame into train / (optional val) / test by GKF fold.

    Parameters
    ----------
    df : E1 DataFrame from build_e1_dataset()
    fold_k : int  — test fold (0–4)
    val_fold : int or None — if given, this fold is used for validation
               (must differ from fold_k). Typically (fold_k + 1) % 5.

    Returns
    -------
    df_train, df_test, df_val (df_val is None if val_fold is None)
    """
    mask_test = df["gkf_fold"] == fold_k
    if val_fold is not None:
        if val_fold == fold_k:
            raise ValueError("val_fold must differ from fold_k")
        mask_val = df["gkf_fold"] == val_fold
        mask_train = ~mask_test & ~mask_val
        return df[mask_train].copy(), df[mask_test].copy(), df[mask_val].copy()
    else:
        return df[~mask_test].copy(), df[mask_test].copy(), None


def split_loso_rows(
    df: pd.DataFrame,
    test_session_key: str,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split E1 DataFrame for one LOSO held-out session.

    Validation set: a grouped fraction of training neurons (no nucleus_id
    overlap with test).

    Returns
    -------
    df_train, df_test, df_val
    """
    mask_test = df["loso_test_scan"] == test_session_key
    df_test = df[mask_test].copy()
    df_trainval = df[~mask_test].copy()

    # pick validation neurons (grouped — no leakage)
    train_neurons = df_trainval["nucleus_id"].unique()
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(train_neurons) * val_fraction))
    val_neurons = set(rng.choice(train_neurons, size=n_val, replace=False))

    mask_val = df_trainval["nucleus_id"].isin(val_neurons)
    df_val = df_trainval[mask_val].copy()
    df_train = df_trainval[~mask_val].copy()
    return df_train, df_test, df_val


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _robust_norm_trace(trace: np.ndarray) -> np.ndarray:
    """Per-trace robust normalization: (x - median) / (1.4826 * MAD + eps)."""
    med = np.median(trace)
    mad = np.median(np.abs(trace - med))
    return (trace - med) / (1.4826 * mad + EPS)


def _session_stats(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute per-session (median, MAD) over all trace values in df.
    Returns dict: session_key -> (median, mad).
    """
    stats: dict[str, tuple[float, float]] = {}
    for sk, grp in df.groupby("session_key"):
        all_vals = np.concatenate(
            [np.asarray(t, dtype=np.float64) for t in grp["response_avg"]]
        )
        med = float(np.median(all_vals))
        mad = float(np.median(np.abs(all_vals - med)))
        stats[str(sk)] = (med, mad)
    return stats


def normalize_traces(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    method: Literal["raw", "per_trace_robust", "per_session_robust", "zscore"] = "per_trace_robust",
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Apply normalization to response_avg column.

    All stats are estimated on df_train only; df_test and df_val are
    transformed using the same stats (no leakage).

    Parameters
    ----------
    method : one of
        'raw'               — no normalization
        'per_trace_robust'  — per-trace (x - median) / (1.4826*MAD)
        'per_session_robust'— fit session median/MAD on train, apply everywhere
        'zscore'            — per-trace zero-mean unit-variance

    Returns
    -------
    (df_train_norm, df_test_norm, df_val_norm)  — copies
    """
    if method == "raw":
        out_val = df_val.copy() if df_val is not None else None
        return df_train.copy(), df_test.copy(), out_val

    def _apply_per_trace(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if method == "per_trace_robust":
            df["response_avg"] = df["response_avg"].apply(
                lambda t: _robust_norm_trace(np.asarray(t, dtype=np.float64)).tolist()
            )
        elif method == "zscore":
            def _z(t):
                t = np.asarray(t, dtype=np.float64)
                mu, sd = t.mean(), t.std()
                return ((t - mu) / (sd + EPS)).tolist()
            df["response_avg"] = df["response_avg"].apply(_z)
        return df

    def _apply_per_session(
        df: pd.DataFrame, stats: dict[str, tuple[float, float]]
    ) -> pd.DataFrame:
        df = df.copy()

        def _norm(row):
            sk = str(row["session_key"])
            med, mad = stats.get(sk, (0.0, 1.0))
            t = np.asarray(row["response_avg"], dtype=np.float64)
            return ((t - med) / (1.4826 * mad + EPS)).tolist()

        df["response_avg"] = df.apply(_norm, axis=1)
        return df

    if method in ("per_trace_robust", "zscore"):
        tr_norm = _apply_per_trace(df_train)
        te_norm = _apply_per_trace(df_test)
        va_norm = _apply_per_trace(df_val) if df_val is not None else None
    elif method == "per_session_robust":
        # fit stats on training rows only
        session_stats = _session_stats(df_train)
        tr_norm = _apply_per_session(df_train, session_stats)
        te_norm = _apply_per_session(df_test, session_stats)
        va_norm = (_apply_per_session(df_val, session_stats)
                   if df_val is not None else None)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    return tr_norm, te_norm, va_norm


# ---------------------------------------------------------------------------
# Array getters — ready for CNN
# ---------------------------------------------------------------------------
def get_family_arrays(
    df: pd.DataFrame,
    family: str,
    pad_to: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y_str, groups) numpy arrays for one stimulus family.

    Parameters
    ----------
    df : E1 DataFrame (already train/test split and normalized)
    family : 'Clip', 'Monet2', 'Trippy', or 'all'
    pad_to : if given, pad/truncate all traces to this length (required for
             'all' family or cross-family models).  If None, use the natural
             family length.

    Returns
    -------
    X       : (n_rows, T) float32 ndarray  — the raw traces
    y_str   : (n_rows,) str ndarray         — layer_label per row
    groups  : (n_rows,) int64 ndarray       — nucleus_id per row
    """
    if family == "all":
        sub = df.copy()
        if pad_to is None:
            pad_to = MAX_TRACE_LEN
    else:
        if family not in FAMILY_LENGTHS:
            raise ValueError(
                f"family must be one of {list(FAMILY_LENGTHS)} or 'all', got {family!r}"
            )
        sub = df[df["stim_type"] == family].copy()

    if len(sub) == 0:
        raise ValueError(f"No rows for family={family!r}")

    def _to_fixed(trace, T):
        t = np.asarray(trace, dtype=np.float32)
        if len(t) >= T:
            return t[:T]
        # zero-pad on the right
        out = np.zeros(T, dtype=np.float32)
        out[: len(t)] = t
        return out

    T = pad_to if pad_to is not None else FAMILY_LENGTHS[family]
    X = np.stack([_to_fixed(t, T) for t in sub["response_avg"]])  # (n, T)
    y_str = sub["layer_label"].to_numpy()
    groups = sub["nucleus_id"].to_numpy(dtype=np.int64)
    return X, y_str, groups


def get_all_families_arrays(
    df: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {family: (X, y_str, groups)} for all three families."""
    return {fam: get_family_arrays(df, fam) for fam in FAMILY_LENGTHS}


# ---------------------------------------------------------------------------
# Amplitude-only control helper
# ---------------------------------------------------------------------------
def make_amplitude_only_traces(
    X: np.ndarray,
    mode: Literal["mean", "max", "auc"] = "mean",
) -> np.ndarray:
    """Replace each trace with a constant trace carrying one amplitude scalar.

    This is the amplitude-only control: the CNN receives a flat trace, so any
    temporal convolution learns nothing from temporal order.  If the CNN
    matches the full-trace model on this control, temporal kinetics are not
    what drove performance.

    Parameters
    ----------
    X    : (n, T) float32
    mode : scalar summary to broadcast — 'mean', 'max', or 'auc' (trapezoid)

    Returns
    -------
    X_amp : (n, T) float32, each row = constant = scalar_summary(row)
    """
    if mode == "mean":
        scalars = X.mean(axis=1, keepdims=True)
    elif mode == "max":
        scalars = X.max(axis=1, keepdims=True)
    elif mode == "auc":
        scalars = np.trapz(X, axis=1, keepdims=True)  # type: ignore[call-overload]
        # normalise by length to keep same scale
        scalars = scalars / X.shape[1]
    else:
        raise ValueError(f"mode must be 'mean', 'max', or 'auc', got {mode!r}")
    return np.broadcast_to(scalars, X.shape).astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Mismatch neuron removal
# ---------------------------------------------------------------------------

#: Maps celltype_label → expected layer_label (from AIBS metamodel v661)
CELLTYPE_TO_LAYER: dict[str, str] = {
    "23P":   "L2/3",
    "4P":    "L4",
    "5P-IT": "L5",
    "5P-ET": "L5",
    "5P-NP": "L5",
    "6P-IT": "L6",
    "6P-CT": "L6",
}


def remove_mismatch_neurons(
    units_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Remove neurons where EM layer_label disagrees with celltype_label.

    A mismatch is defined as: the layer implied by the AIBS metamodel
    cell-type label ≠ the anatomical EM layer label.  These neurons sit at
    layer boundaries or have ambiguous identity and can act as noise for a
    layer-discrimination classifier.

    Parameters
    ----------
    units_df : DataFrame from load_working_pop() (must have layer_label and
               celltype_label columns)

    Returns
    -------
    clean_df : units_df with mismatch rows removed
    report   : dict with counts and breakdown
    """
    df = units_df.copy()
    df["_expected_layer"] = df["celltype_label"].map(CELLTYPE_TO_LAYER)
    df["_mismatch"] = df["layer_label"] != df["_expected_layer"]

    n_total    = len(df)
    n_mismatch = int(df["_mismatch"].sum())
    n_clean    = n_total - n_mismatch

    breakdown = (
        df[df["_mismatch"]]
        .groupby(["layer_label", "celltype_label"])
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    clean_df = df[~df["_mismatch"]].drop(
        columns=["_expected_layer", "_mismatch"]
    ).copy()

    report = {
        "n_total":    n_total,
        "n_mismatch": n_mismatch,
        "n_clean":    n_clean,
        "pct_removed": round(100 * n_mismatch / n_total, 2),
        "breakdown":  breakdown,
    }

    if verbose:
        print(f"Mismatch removal: {n_mismatch} / {n_total} neurons removed "
              f"({report['pct_removed']:.1f}%)")
        print(f"Remaining clean neurons: {n_clean}")
        print("\nLayer distribution after cleaning:")
        print(clean_df["layer_label"].value_counts().sort_index().to_string())

    return clean_df, report


# ---------------------------------------------------------------------------
# Session balance analysis
# ---------------------------------------------------------------------------

def find_balanced_session(
    units_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rank all sessions by layer-balance entropy (higher = more balanced).

    Uses normalised Shannon entropy (base 4 = n_classes) so 1.0 = perfectly
    uniform, 0.0 = single class.  Sessions with fewer than 4 layers present
    are flagged as degenerate.

    Parameters
    ----------
    units_df : clean units DataFrame (after remove_mismatch_neurons)

    Returns
    -------
    DataFrame sorted by entropy descending, with columns:
    session_key, total, L2/3, L4, L5, L6, entropy, min_max_ratio, degenerate
    """
    rows = []
    for sk, grp in units_df.groupby("session_key"):
        counts = grp["layer_label"].value_counts()
        total  = len(grp)
        l23 = int(counts.get("L2/3", 0))
        l4  = int(counts.get("L4",   0))
        l5  = int(counts.get("L5",   0))
        l6  = int(counts.get("L6",   0))

        p = np.array([l23, l4, l5, l6], dtype=float)
        p_norm = p / p.sum()
        p_nz   = p_norm[p_norm > 0]
        ent = float(-np.sum(p_nz * np.log(p_nz)) / np.log(4))
        min_max = float(p_nz.min() / p_nz.max()) if len(p_nz) > 1 else 0.0
        degen  = int((p == 0).sum()) > 0  # at least one layer missing

        rows.append({
            "session_key":   sk,
            "total":         total,
            "L2/3":          l23,
            "L4":            l4,
            "L5":            l5,
            "L6":            l6,
            "entropy":       round(ent, 3),
            "min_max_ratio": round(min_max, 3),
            "degenerate":    degen,
        })

    df_rank = pd.DataFrame(rows).sort_values("entropy", ascending=False).reset_index(drop=True)

    if verbose:
        print("Session balance ranking (clean neurons):")
        print(df_rank.to_string(index=False))
        best = df_rank[~df_rank["degenerate"]].iloc[0]
        print(f"\nBest balanced non-degenerate session: {best['session_key']} "
              f"(entropy={best['entropy']:.3f}, n={best['total']})")

    return df_rank


# ---------------------------------------------------------------------------
# Single-session E1 dataset builder
# ---------------------------------------------------------------------------

def build_single_session_e1(
    session_key: str,
    units_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the E1 dataset restricted to one session and the given neurons.

    Uses `units_df` as the population (apply remove_mismatch_neurons first).

    Returns a DataFrame with the same schema as build_e1_dataset() but for
    one session only:
    nucleus_id, condition_hash, session_key, stim_type, layer, layer_label,
    n_frames, n_trials, response_avg.
    """
    oracle_hashes, _ = load_oracle_hashes()
    oracle_set = set(oracle_hashes)

    # Neurons in this session
    session_units = units_df[units_df["session_key"] == session_key].copy()
    if len(session_units) == 0:
        raise ValueError(f"No neurons found for session_key={session_key!r}")

    if verbose:
        print(f"Session {session_key}: {len(session_units)} clean neurons")
        print(session_units["layer_label"].value_counts().sort_index().to_string())

    # Load traces for this session
    traces = pd.read_parquet(_TRACES_AVG_PATH)
    traces = traces[
        (traces["session_key"] == session_key) &
        (traces["condition_hash"].isin(oracle_set))
    ].copy()

    # Keep only nucleus_ids in our clean population
    valid_nids = set(session_units["nucleus_id"].unique())
    traces = traces[traces["nucleus_id"].isin(valid_nids)].copy()

    # Join layer labels
    label_map = session_units.set_index("nucleus_id")[["layer", "layer_label"]].to_dict("index")

    df = traces.copy()
    df["layer"]       = df["nucleus_id"].map(lambda n: label_map[n]["layer"])
    df["layer_label"] = df["nucleus_id"].map(lambda n: label_map[n]["layer_label"])
    df = df.reset_index(drop=True)

    if verbose:
        expected_rows = len(session_units) * len(oracle_hashes)
        print(f"\nE1 rows: {len(df):,}  (expected ~{expected_rows:,})")
        print(f"Neurons with data: {df['nucleus_id'].nunique():,}")
        print(f"Oracle hashes covered: {df['condition_hash'].nunique():,}")
        print(f"\nRows by family:")
        print(df.groupby("stim_type")["nucleus_id"].count().to_string())

    return df


# ---------------------------------------------------------------------------
# Pooled multi-session E1 builder (memory-efficient via predicate pushdown)
# ---------------------------------------------------------------------------

def build_pooled_e1(
    session_keys: list[str],
    units_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build E1 pooled across the given sessions, oracle-filtered, label-joined.

    Same schema as `build_single_session_e1` but covers multiple sessions in
    one pass. Reads `traces_avg.parquet` ONCE with pyarrow predicate pushdown
    (filter on `session_key` and `condition_hash`) so memory peaks at the
    filtered subset, not the full ~4M-row table.

    Parameters
    ----------
    session_keys : list of session keys to pool (e.g. the six balanced
                   sessions ['5_6','5_7','6_2','6_4','6_6','6_7']).
    units_df     : working population — typically the output of
                   `remove_mismatch_neurons` so only consistent-only neurons
                   contribute.

    Returns
    -------
    DataFrame with columns:
      nucleus_id, session_key, condition_hash, stim_type, n_frames, n_trials,
      response_avg, layer, layer_label.

    Notes
    -----
    Neurons in `units_df` whose session is not in `session_keys` are excluded.
    Neurons with incomplete oracle coverage in their own session are kept; the
    caller is responsible for any uniform-coverage filtering downstream
    (`build_e1_neuron_tensors_with_hashes` will assert uniform coverage).
    """
    oracle_hashes, _ = load_oracle_hashes()
    oracle_set = set(oracle_hashes)
    if not session_keys:
        raise ValueError("session_keys must be non-empty")

    # Restrict the working population to the requested sessions
    pop = units_df[units_df["session_key"].isin(session_keys)].copy()
    if len(pop) == 0:
        raise ValueError(f"No neurons found for any of session_keys={session_keys}")

    valid_nids = set(pop["nucleus_id"].unique())

    if verbose:
        print(f"Pooling {len(session_keys)} sessions: {session_keys}")
        print(f"  Population (mismatch-clean): {len(pop)} neurons")
        print(pop.groupby("session_key")["nucleus_id"].count().to_string())

    # Predicate pushdown — read ONLY the rows we will keep
    traces = pd.read_parquet(
        _TRACES_AVG_PATH,
        filters=[
            ("session_key",    "in", list(session_keys)),
            ("condition_hash", "in", list(oracle_set)),
        ],
    )
    traces = traces[traces["nucleus_id"].isin(valid_nids)].copy()

    # Join layer labels
    label_map = (
        pop.set_index("nucleus_id")[["layer", "layer_label"]].to_dict("index")
    )
    traces["layer"]       = traces["nucleus_id"].map(lambda n: label_map[n]["layer"])
    traces["layer_label"] = traces["nucleus_id"].map(lambda n: label_map[n]["layer_label"])
    traces = traces.reset_index(drop=True)

    if verbose:
        cov = traces.groupby("nucleus_id")["condition_hash"].nunique()
        print(f"\n  Pooled E1 rows: {len(traces):,}")
        print(f"  Neurons with data: {traces['nucleus_id'].nunique():,}")
        print(f"  Oracle coverage per neuron: min={cov.min()} median={cov.median():.0f} max={cov.max()}")
        if (cov < 116).any():
            print(f"  ⚠️  {(cov < 116).sum()} neurons have <116 oracles — drop before set-tensor build.")
        print("\n  Per-session row count:")
        print(traces.groupby("session_key").size().to_string())

    return traces


# ---------------------------------------------------------------------------
# Multichannel: per-(session, hash) behaviour signals (E1)
# ---------------------------------------------------------------------------

def load_behaviour_channels_e1(
    session_keys: list[str],
    channels: list[str] = ("treadmill", "pupil_dilation"),
    pad_to: int = MAX_TRACE_LEN,
    verbose: bool = True,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load and trial-average behaviour signals at the (session, hash) level.

    For each requested behaviour `channel`, gather all trials in a given
    `(session_key, condition_hash)` pair (oracle subset) and average them
    timepoint-wise. The result is the **same shape as the calcium
    `response_avg`** so multichannel CNN input has matching temporal extent.

    Behaviour signals are *shared* across all neurons in the same session
    viewing the same hash. Therefore multichannel input cannot help
    discriminate between two neurons in the same `(session, hash)`; it can
    only help via cross-session context (different sessions have different
    behaviour distributions) or via within-(session, hash) trial variability
    when the model is trained on E0 (planning §2.1).

    Parameters
    ----------
    session_keys : sessions to include (e.g. the six balanced sessions).
    channels     : behaviour columns from trials_meta.parquet to load. Each
                   must be one of `BEHAVIOUR_CHANNELS`. Default
                   ('treadmill', 'pupil_dilation').
    pad_to       : pad/truncate length (default MAX_TRACE_LEN = 113), so each
                   channel array has the same temporal extent as the calcium
                   response_avg used in `build_e1_neuron_tensors_with_hashes`.

    Returns
    -------
    dict keyed by `(session_key, condition_hash)` whose value is a dict
    mapping each channel name to a 1-D numpy array of length `pad_to`.
    Missing entries (no trials of that hash in that session) are absent
    from the dict — callers must handle KeyError or use .get().
    """
    invalid = [c for c in channels if c not in BEHAVIOUR_CHANNELS]
    if invalid:
        raise ValueError(
            f"Unknown behaviour channels {invalid!r}; "
            f"valid choices are {list(BEHAVIOUR_CHANNELS)}"
        )

    oracle_hashes, _ = load_oracle_hashes()
    oracle_set = set(oracle_hashes)

    cols = ["session_key", "condition_hash", "n_frames"] + list(channels)
    if verbose:
        print(f"Loading behaviour channels {list(channels)} for sessions={session_keys} ...")
    tm = pd.read_parquet(
        _TRIALS_META_PATH,
        columns=cols,
        filters=[
            ("session_key",    "in", list(session_keys)),
            ("condition_hash", "in", list(oracle_set)),
        ],
    )
    if verbose:
        print(f"  trials_meta rows: {len(tm):,}")

    def _pad_or_truncate(arr_like, target_T: int) -> np.ndarray:
        a = np.asarray(arr_like, dtype=np.float32)
        if len(a) >= target_T:
            return a[:target_T]
        out = np.zeros(target_T, dtype=np.float32)
        out[: len(a)] = a
        return out

    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (sk, ch), grp in tm.groupby(["session_key", "condition_hash"]):
        out[(sk, ch)] = {}
        for col in channels:
            # Stack all trials' signals (each up to MAX_TRACE_LEN), then mean.
            stacked = np.stack(
                [_pad_or_truncate(g, pad_to) for g in grp[col]], axis=0,
            )
            # Some behaviour samples may be NaN — use nanmean to be robust.
            avg = np.nanmean(stacked, axis=0)
            # If a frame is NaN across ALL trials, fill with 0 (treated as
            # 'no data' — the masked GAP will skip padded frames, but inner
            # frames default to 0).
            avg = np.where(np.isnan(avg), 0.0, avg).astype(np.float32)
            out[(sk, ch)][col] = avg

    if verbose:
        n_pairs = len(out)
        n_expected = sum(1 for _ in session_keys) * len(oracle_set)
        print(f"  trial-averaged (session, hash) pairs: {n_pairs:,} "
              f"(of {n_expected:,} possible)")
        # Quick value-range printout per channel
        for col in channels:
            vals = np.concatenate([d[col] for d in out.values()])
            print(f"  channel {col!r}: range [{np.nanmin(vals):.3f}, "
                  f"{np.nanmax(vals):.3f}], mean {np.nanmean(vals):.3f}")
    return out


# ---------------------------------------------------------------------------
# Multichannel: per-(session, hash) stimulus features (Tier D, broadcast)
# ---------------------------------------------------------------------------

#: Tier D scalar stimulus features computed at the `condition_hash` level
#: (see `src/features/tier_d.py`). They are broadcast as constant time channels
#: per (session, hash) so they share the same shape contract as the behaviour
#: channels produced by `load_behaviour_channels_e1`.
STIMULUS_CHANNELS_TIER_D: tuple[str, ...] = (
    "d_luminance_mean",
    "d_luminance_std",
    "d_contrast",
    "d_motion_energy",
    "d_sf_low",
    "d_sf_mid",
    "d_sf_high",
    "d_tf_low",
    "d_tf_mid",
    "d_tf_high",
)


def load_stimulus_channels_e1(
    session_keys: list[str],
    channels: list[str] = STIMULUS_CHANNELS_TIER_D,
    pad_to: int = MAX_TRACE_LEN,
    standardize: bool = True,
    verbose: bool = True,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load Tier-D stimulus descriptors and broadcast them as constant time channels.

    Stimulus features are *hash-level* by construction — every trial of the
    same `condition_hash` shows the same physical stimulus, so the feature is
    a scalar that does not vary across the response window. We therefore
    broadcast each scalar into a constant array of length `pad_to` so the
    model's 1-D convolution sees a stable channel that already encodes the
    per-hash stimulus identity (planning §8 trace-only ladder; see also the
    'multichannel without hash embeddings' design rationale).

    The output dict shape **matches `load_behaviour_channels_e1`** exactly,
    so the two dicts can be merged and fed into
    `build_e1_neuron_tensors_multichannel(channel_names=stim+behaviour)`.

    Parameters
    ----------
    session_keys : sessions to include — entries are duplicated for each
                   session because the stimulus is session-independent.
    channels     : Tier-D scalar columns to broadcast. Defaults to all 10.
    pad_to       : channel length (default 113 frames, matching the calcium
                   set-tensor and behaviour map).
    standardize  : if True, z-score each Tier-D scalar across the 116 oracle
                   hashes (using mean/std across hashes — never within a
                   single hash, which would be ill-defined). This is the
                   safer default because raw Tier-D values have very
                   different scales (e.g. d_luminance_mean ∈ [0, 1] vs
                   d_motion_energy in arbitrary units), and the encoder's
                   first conv layer will otherwise be dominated by whichever
                   channel happens to have the largest variance. Standardisation
                   is computed across ALL 116 oracle hashes — it does not see
                   neuron labels and is therefore not a label leak.

    Returns
    -------
    dict keyed by `(session_key, condition_hash)` whose value is a dict
    mapping each Tier-D channel name to a 1-D numpy array of length `pad_to`
    holding the scalar value broadcast across all timepoints.
    """
    from src.config import PROCESSED_FEATURES_DIR

    invalid = [c for c in channels if c not in STIMULUS_CHANNELS_TIER_D]
    if invalid:
        raise ValueError(
            f"Unknown stimulus channels {invalid!r}; "
            f"valid choices are {list(STIMULUS_CHANNELS_TIER_D)}"
        )

    oracle_hashes, _ = load_oracle_hashes()
    oracle_set = set(oracle_hashes)

    d_path = PROCESSED_FEATURES_DIR / "D_per_hash.parquet"
    if verbose:
        print(f"Loading Tier-D stimulus features {list(channels)} ...")
    d = pd.read_parquet(d_path)
    d = d[d["condition_hash"].isin(oracle_set)].copy()
    if verbose:
        print(f"  Tier-D rows on oracle subset: {len(d):,}")

    # Optional per-channel standardisation across the oracle hashes
    feats = d[list(channels)].astype(np.float32).values  # (n_hash, n_chan)
    if standardize:
        mu = np.nanmean(feats, axis=0, keepdims=True)
        sd = np.nanstd(feats, axis=0, keepdims=True)
        feats = (feats - mu) / np.where(sd > 1e-8, sd, 1.0)
        if verbose:
            print(
                f"  Standardised Tier-D channels (z-score across {len(d)} oracle hashes)."
            )

    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)

    hash_to_vec = {h: feats[i] for i, h in enumerate(d["condition_hash"].tolist())}

    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for sk in session_keys:
        for h in oracle_set:
            v = hash_to_vec.get(h)
            if v is None:
                continue  # hash absent from D_per_hash
            entry: dict[str, np.ndarray] = {}
            for c_idx, c_name in enumerate(channels):
                entry[c_name] = np.full(
                    (pad_to,), float(v[c_idx]), dtype=np.float32,
                )
            out[(sk, h)] = entry

    if verbose:
        print(
            f"  Stimulus map: {len(out):,} (session, hash) pairs across "
            f"{len(session_keys)} session(s)."
        )
    return out


def load_stimulus_channels_e1_timevarying(
    session_keys: list[str],
    pad_to: int = MAX_TRACE_LEN,
    standardize: bool = True,
    cache_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load per-frame (time-varying) stimulus channels per `(session, hash)`.

    Unlike `load_stimulus_channels_e1` (which broadcasts scalar Tier-D
    descriptors as constant-time channels), this function reads the raw oracle
    videos via `MicronsFunctionalReader` and computes the per-frame analogues
    of the Tier-D descriptors at the *native* video frame index. The result
    is zero-padded on the right to `pad_to = MAX_TRACE_LEN` exactly like the
    calcium `response_avg`, so the channel arrays are temporally aligned
    frame-by-frame with the calcium trace.

    Stimulus features depend only on the hash (not the session), but the
    output is keyed by `(session_key, condition_hash)` so it can be merged
    with `load_behaviour_channels_e1`'s output via `merge_channel_maps`.

    Parameters
    ----------
    session_keys : sessions to include — entries are duplicated across them
                   because the stimulus is session-invariant.
    pad_to       : length to pad/truncate to (default 113).
    standardize  : if True, z-score every channel across all `(hash, time)`
                   samples — *across hashes*, not within. This is *not* a
                   label leak (the layer label is not used). Behaviour
                   normalisation should NOT be performed here; for LOSO,
                   the caller must standardise behaviour using only the
                   training sessions of each fold (see
                   `zscore_behaviour_for_loso_fold`).
    cache_path   : optional path to a `.npz` cache. If the file exists it is
                   loaded; otherwise the per-frame extraction is run once
                   and the result is written to disk. Using a cache is
                   highly recommended because video extraction is slow.

    Returns
    -------
    dict keyed by `(session_key, condition_hash)`. Each value is a dict with
    one key per channel in `PER_FRAME_STIM_CHANNELS`, each pointing to a
    1-D `float32` array of length `pad_to`.
    """
    from src.features.tier_d import (
        PER_FRAME_STIM_CHANNELS, compute_per_frame_stim_map,
    )

    oracle_hashes, _ = load_oracle_hashes()
    oracle_set = set(oracle_hashes)

    # ---------- Cache hit ---------------------------------------------------
    per_hash_arrays: Optional[dict[str, dict[str, np.ndarray]]] = None
    if cache_path is not None and Path(cache_path).exists():
        if verbose:
            print(f"Loading cached per-frame stim features from {cache_path} ...")
        npz = np.load(cache_path, allow_pickle=False)
        # Cache layout: one (n_chan, pad_to) array per hash, plus a hashes/channels listing.
        cached_hashes   = list(npz["hashes"])
        cached_channels = list(npz["channels"])
        if list(cached_channels) != list(PER_FRAME_STIM_CHANNELS):
            print(
                "  cache channel order does not match PER_FRAME_STIM_CHANNELS — recomputing."
            )
        else:
            per_hash_arrays = {}
            for h in cached_hashes:
                arr = npz[f"feat__{h}"]              # (n_chan, pad_to)
                per_hash_arrays[h] = {
                    cached_channels[i]: arr[i].astype(np.float32)
                    for i in range(len(cached_channels))
                }

    # ---------- Cache miss → compute via MicronsFunctionalReader ----------
    if per_hash_arrays is None:
        if verbose:
            print(
                f"Computing per-frame stim features for {len(oracle_set)} oracle hashes "
                f"(this is slow on first run; subsequent calls hit the cache)..."
            )
        try:
            from microns_datacleaner.functionalreader import MicronsFunctionalReader
        except ImportError as e:
            raise ImportError(
                "microns_datacleaner is required to compute per-frame stim features. "
                "Install via `pip install microns_datacleaner` or set "
                "cache_path to a precomputed `.npz`."
            ) from e
        from src.config import DATA_DIR, FUNCTIONAL_H5
        reader = MicronsFunctionalReader(
            datadir=str(DATA_DIR), path=str(FUNCTIONAL_H5),
        )
        per_hash_arrays = compute_per_frame_stim_map(
            reader, sorted(oracle_set), pad_to=pad_to, progress_every=50,
        )
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            kwargs = {
                f"feat__{h}": np.stack(
                    [per_hash_arrays[h][c] for c in PER_FRAME_STIM_CHANNELS], axis=0,
                )
                for h in per_hash_arrays
            }
            np.savez_compressed(
                cache_path,
                hashes=np.array(list(per_hash_arrays.keys())),
                channels=np.array(list(PER_FRAME_STIM_CHANNELS)),
                **kwargs,
            )
            if verbose:
                print(f"  Cached → {cache_path}")

    # ---------- Optional standardisation across hashes -------------------
    if standardize:
        for ch in PER_FRAME_STIM_CHANNELS:
            stack = np.stack(
                [per_hash_arrays[h][ch] for h in per_hash_arrays], axis=0,
            )                                                    # (n_hash, pad_to)
            mu = float(np.nanmean(stack))
            sd = float(np.nanstd(stack))
            if not np.isfinite(sd) or sd < 1e-8:
                sd = 1.0
            for h in per_hash_arrays:
                per_hash_arrays[h][ch] = (
                    (per_hash_arrays[h][ch] - mu) / sd
                ).astype(np.float32)
        if verbose:
            print(
                f"  Standardised {len(PER_FRAME_STIM_CHANNELS)} per-frame stim channels "
                f"(z-score across hashes × time)."
            )

    # ---------- Lift to (session, hash) → channels ------------------------
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for sk in session_keys:
        for h in oracle_set:
            v = per_hash_arrays.get(h)
            if v is None:
                continue
            out[(sk, h)] = {ch: v[ch] for ch in PER_FRAME_STIM_CHANNELS}
    if verbose:
        print(
            f"  Per-frame stim map: {len(out):,} (session, hash) pairs across "
            f"{len(session_keys)} session(s); channels = {list(PER_FRAME_STIM_CHANNELS)}"
        )
    return out


def zscore_behaviour_for_loso_fold(
    chan_map: dict[tuple[str, str], dict[str, np.ndarray]],
    train_sessions: list[str],
    behaviour_channels: list[str],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Return a *new* channel map with behaviour z-scored using only training sessions.

    For each LOSO fold the held-out session is excluded from the z-score
    statistics. The function does **not** mutate the input map.

    Parameters
    ----------
    chan_map           : un-normalised behaviour map (output of
                         `load_behaviour_channels_e1`, optionally merged with
                         a stimulus map). Keys `(session_key, condition_hash)`.
    train_sessions     : the sessions available for training in this LOSO
                         fold (the held-out session is *excluded*).
    behaviour_channels : list of channels to standardise. Other channels
                         (e.g. stimulus) are left untouched.

    Returns
    -------
    A new dict with the same keys; the listed `behaviour_channels` are
    z-scored using mean/std computed over `(session, hash)` entries whose
    `session in train_sessions` only.
    """
    train_set = set(train_sessions)

    # Compute per-channel mean/std on training-session entries only
    stats: dict[str, tuple[float, float]] = {}
    for ch in behaviour_channels:
        vals = np.concatenate(
            [d[ch] for (sk, _h), d in chan_map.items() if sk in train_set]
        )
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        stats[ch] = (mu, sd)

    # Apply to *all* entries (train + held-out test)
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, d in chan_map.items():
        new_entry = dict(d)
        for ch in behaviour_channels:
            mu, sd = stats[ch]
            new_entry[ch] = ((d[ch] - mu) / sd).astype(np.float32)
        out[key] = new_entry
    return out


def merge_channel_maps(
    *maps: dict[tuple[str, str], dict[str, np.ndarray]],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Merge several channel-maps (e.g. stimulus + behaviour) into one.

    Each input has the shape returned by `load_stimulus_channels_e1` /
    `load_behaviour_channels_e1`: a dict keyed by `(session_key, condition_hash)`
    whose value is `{channel_name: ndarray of length pad_to}`. The output keeps
    every key that appears in *all* input maps (intersection-of-keys), so a
    `(session, hash)` cell is kept only if every map can fill its channels;
    anything else would silently zero-fill some channels and break per-channel
    reasoning.

    Returns
    -------
    merged : dict same shape, with the union of channel dicts at each key.
    """
    if not maps:
        return {}
    common = set(maps[0].keys())
    for m in maps[1:]:
        common &= set(m.keys())

    merged: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key in common:
        entry: dict[str, np.ndarray] = {}
        for m in maps:
            entry.update(m[key])
        merged[key] = entry
    return merged


# ---------------------------------------------------------------------------
# Single-session E0 dataset builder (trial-level)
# ---------------------------------------------------------------------------

_TRACES_TRIAL_PATH: Path = PROCESSED_TABLES_DIR / "traces.parquet"


def build_single_session_e0(
    session_key: str,
    units_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the E0 dataset (trial level) restricted to one session.

    Each row corresponds to one trial:

        E0 = (nucleus_id, condition_hash, trial_idx)

    Restricted to the 116 oracle hashes and to the neurons in `units_df`
    (apply `remove_mismatch_neurons` first).

    Returns a DataFrame with columns:
      nucleus_id, session_key, condition_hash, stim_type, trial_idx,
      n_frames, layer, layer_label, response  (list[float] — the per-trial
      calcium trace).

    Notes
    -----
    * Trial counts per (neuron, hash) are NOT uniform on this corpus: a small
      subset of "true oracle" hashes (e.g. 6 Clip oracles on 5_6) have ~10
      trials per neuron, while the remaining oracle hashes have ~2 trials.
      The two aggregation modes in §6.2 (row-uniform vs hash-uniform) are
      designed to expose / correct for this voting imbalance — see
      `src.eval.metrics.aggregate_probs_hash_uniform` for hash-uniform
      neuron-level scoring.
    * The `response` column holds **per-trial** traces (not within-hash
      averages). Pass `trace_col='response'` to `get_padded_arrays` and
      `baseline_subtract_traces` when consuming this DataFrame.
    """
    oracle_hashes, _ = load_oracle_hashes()
    oracle_set = set(oracle_hashes)

    session_units = units_df[units_df["session_key"] == session_key].copy()
    if len(session_units) == 0:
        raise ValueError(f"No neurons found for session_key={session_key!r}")

    if verbose:
        print(f"Session {session_key}: {len(session_units)} clean neurons")
        print(session_units["layer_label"].value_counts().sort_index().to_string())

    # Predicate-pushdown read so we never materialise the full 4.1M-row table
    # in memory; pyarrow filters at file scan time.
    traces = pd.read_parquet(
        _TRACES_TRIAL_PATH,
        filters=[
            ("session_key",    "==", session_key),
            ("condition_hash", "in", list(oracle_set)),
        ],
    )

    valid_nids = set(session_units["nucleus_id"].unique())
    traces = traces[traces["nucleus_id"].isin(valid_nids)].copy()

    label_map = (
        session_units.set_index("nucleus_id")[["layer", "layer_label"]]
        .to_dict("index")
    )
    traces["layer"]       = traces["nucleus_id"].map(lambda n: label_map[n]["layer"])
    traces["layer_label"] = traces["nucleus_id"].map(lambda n: label_map[n]["layer_label"])
    traces = traces.reset_index(drop=True)

    if verbose:
        n_neurons   = traces["nucleus_id"].nunique()
        n_hashes    = traces["condition_hash"].nunique()
        tph         = traces.groupby(["nucleus_id", "condition_hash"]).size()
        rows_per_n  = traces.groupby("nucleus_id").size()
        print(f"\nE0 rows: {len(traces):,}")
        print(f"Neurons: {n_neurons}   |   oracle hashes: {n_hashes}")
        print(f"Trials per (neuron, hash): "
              f"min={int(tph.min())} median={int(tph.median())} "
              f"max={int(tph.max())} mean={tph.mean():.2f}")
        print(f"Trial rows per neuron    : "
              f"min={int(rows_per_n.min())} median={int(rows_per_n.median())} "
              f"max={int(rows_per_n.max())} mean={rows_per_n.mean():.1f}")
        print("\nRows by family:")
        print(traces.groupby("stim_type")["nucleus_id"].count().to_string())

    return traces


# ---------------------------------------------------------------------------
# Stratified neuron-level train / val / test split (single-session)
# ---------------------------------------------------------------------------

def stratified_neuron_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split at the neuron level.

    Stratification is by layer_label so each split has proportional class
    representation.  All rows of a given nucleus_id land in exactly one split
    (no leakage).

    Returns
    -------
    df_train, df_val, df_test  — row subsets of df
    """
    rng = np.random.default_rng(seed)
    train_nids, val_nids, test_nids = [], [], []

    for layer, grp in df.drop_duplicates("nucleus_id").groupby("layer_label"):
        nids = grp["nucleus_id"].to_numpy()
        rng.shuffle(nids)
        n = len(nids)
        n_train = max(1, int(n * train_frac))
        n_val   = max(1, int(n * val_frac))
        train_nids.extend(nids[:n_train].tolist())
        val_nids  .extend(nids[n_train : n_train + n_val].tolist())
        test_nids .extend(nids[n_train + n_val :].tolist())

    train_set = set(train_nids)
    val_set   = set(val_nids)
    test_set  = set(test_nids)

    df_train = df[df["nucleus_id"].isin(train_set)].copy()
    df_val   = df[df["nucleus_id"].isin(val_set  )].copy()
    df_test  = df[df["nucleus_id"].isin(test_set )].copy()
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Baseline subtraction
# ---------------------------------------------------------------------------

def baseline_subtract_traces(
    df: pd.DataFrame,
    n_baseline_frames: int = 5,
    trace_col: str = "response_avg",
) -> pd.DataFrame:
    """Subtract per-trace pre-stimulus baseline.

    baseline(trace) = median of the first `n_baseline_frames` frames.
    x_bs(t) = x(t) - baseline

    This removes slow drift and session-level offset differences while
    preserving the stimulus-evoked temporal shape.

    Parameters
    ----------
    df               : DataFrame with `trace_col` column.
    n_baseline_frames: number of pre-stimulus frames to use as baseline.
                       Default 5 (~0.67 s at 7.5 Hz).  If the trace is
                       shorter than n_baseline_frames, all frames are used.
    trace_col        : column holding the trace (default 'response_avg' for
                       E1; pass 'response' for E0 per-trial traces).
    """
    df = df.copy()

    def _bs(trace):
        t = np.asarray(trace, dtype=np.float64)
        n_bl = min(n_baseline_frames, len(t))
        baseline = np.median(t[:n_bl])
        return (t - baseline).tolist()

    df[trace_col] = df[trace_col].apply(_bs)
    return df


# ---------------------------------------------------------------------------
# Padded all-family array getter (for single-CNN-on-all-families approach)
# ---------------------------------------------------------------------------

def get_padded_arrays(
    df: pd.DataFrame,
    pad_to: int = MAX_TRACE_LEN,
    trace_col: str = "response_avg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return all families padded to pad_to frames, with a validity mask.

    Zero-pads shorter traces (Clip at ~75 frames) on the right.
    The mask lets the CNN ignore padded frames in global-average pooling.

    Parameters
    ----------
    df         : DataFrame with one trace per row.
    pad_to     : pad length (default MAX_TRACE_LEN = 113).
    trace_col  : column holding the trace as a list/array of floats. Default
                 'response_avg' for E1 (within-hash mean); pass 'response'
                 for E0 (per-trial trace).

    Returns
    -------
    X      : (n, pad_to) float32  — padded traces
    mask   : (n, pad_to) float32  — 1 = valid frame, 0 = padding
    y_str  : (n,) str ndarray     — layer_label per row
    groups : (n,) int64 ndarray   — nucleus_id per row
    """
    def _pad(trace):
        t = np.asarray(trace, dtype=np.float32)
        T = len(t)
        if T >= pad_to:
            return t[:pad_to], np.ones(pad_to, dtype=np.float32)
        x = np.zeros(pad_to, dtype=np.float32)
        m = np.zeros(pad_to, dtype=np.float32)
        x[:T] = t
        m[:T] = 1.0
        return x, m

    results = [_pad(t) for t in df[trace_col]]
    X    = np.stack([r[0] for r in results])
    mask = np.stack([r[1] for r in results])
    y_str  = df["layer_label"].to_numpy()
    groups = df["nucleus_id"].to_numpy(dtype=np.int64)
    return X, mask, y_str, groups


def make_time_shuffled_traces(
    X: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Shuffle timepoints within each trace independently.

    Preserves the marginal amplitude distribution of every trace but destroys
    temporal order.  If the model is unchanged, it is NOT using kinetics.
    """
    rng = np.random.default_rng(seed)
    X_shuf = X.copy()
    for i in range(len(X_shuf)):
        rng.shuffle(X_shuf[i])
    return X_shuf
