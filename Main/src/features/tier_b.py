"""Tier B — trial-to-trial reliability features per (neuron, hash).

WORKFLOW.md §4 Tier B
=====================
"Statistics computed across the trials of that (neuron, hash) **before** the
within-hash average":
  - trial-to-trial variance and standard deviation,
  - coefficient of variation,
  - Fano-like factor (variance / mean of per-trial response amplitudes),
  - signal-to-noise ratio (mean / std),
  - within-hash trial-trial Pearson correlation of the full traces,
  - oracle correlation when the hash has enough trials (Sinz/Tolias-style),
  - state-conditional reliability when behaviour is partitioned (feeds Tier C1).

This module computes the **stimulus-only** subset (everything except state-
conditional reliability — that lives in Tier C1 since it requires behaviour).

Why "before the within-hash average": these features measure the variability
that within-hash averaging suppresses. A hash whose trials are highly
reliable is informative; a hash whose trials look like white noise around the
mean is not. Reliability differs between cell types independently of tuning
(L4 cells receive direct LGN input → most reliable; L6 → least), so this is
one of the few feature blocks predicted *a priori* to differ between layers.

Domain
======
Tier B requires `n_trials >= 2`. From the EDA: per neuron, ~136 of the 280
unique hashes meet this — 96 oracle Clips (with 2–10 repeats each), 20
Monet2 hashes (always 2 trials), 20 Trippy hashes (always 2 trials). The
total Tier B row count is therefore ~1.2M, not 2.5M.

Per-neuron summary
==================
The same 7 scalars can be averaged across all of a neuron's repeated hashes
to give a per-neuron reliability vector. WORKFLOW §4 Tier B explicitly
allows this. We expose `per_neuron_summary` to compute it.

Features (7)
============
  b_amp_var          : variance of per-trial response amplitudes (= trace mean per trial).
  b_amp_std          : sqrt of the variance.
  b_amp_cv           : std / mean (coefficient of variation); NaN if mean ≤ EPS_AMP_MEAN.
  b_amp_fano         : var / mean (Fano-like factor); NaN if mean ≤ EPS_AMP_MEAN.
  b_amp_snr          : mean / std (SNR); NaN if std ≤ EPS_AMP_STD.
  b_trace_corr_mean  : mean of pairwise Pearson correlations between the n_trials
                       full traces. With n=2 this is the single pair correlation.
  b_oracle_corr      : leave-one-trial-out correlation. For each trial i, correlate
                       it with the mean of the other (n-1) trials, then average
                       across i. Sinz/Tolias-style canonical reliability scalar.

For both correlation features, when a trial has zero variance (flat / all-zero
trace), `np.corrcoef` is undefined; we return NaN for that pair, then average
the well-defined pairs only. If *no* pair is well-defined the feature is NaN.

Numerical safety
================
Same per-quantity threshold scheme as Tier A. EPS_AMP_MEAN guards CV/Fano.
EPS_AMP_STD guards SNR.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import pandas as pd


# Per-quantity thresholds (in trace-mean units; calibrated against the data).
EPS_AMP_MEAN: Final[float] = 0.01   # mean across trials of per-trial means
EPS_AMP_STD:  Final[float] = 0.001  # std  across trials of per-trial means
EPS_TRACE_VAR: Final[float] = 1e-6  # per-trial trace variance for corr validity


B_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "b_amp_var",
    "b_amp_std",
    "b_amp_cv",
    "b_amp_fano",
    "b_amp_snr",
    "b_trace_corr_mean",
    "b_oracle_corr",
)


# ---------------------------------------------------------------------------
# Vectorized core: amp stats + correlations on a single (group_count, n_trials,
# n_frames) tensor where every group in `traces_3d` shares the same `n_trials`
# and `n_frames`. This is the fast path used by `compute_tier_b`.
# ---------------------------------------------------------------------------
def _safe_div_vec(num: np.ndarray, den: np.ndarray, eps: float) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = np.isfinite(den) & (np.abs(den) > eps)
    out[mask] = num[mask] / den[mask]
    out[~np.isfinite(out)] = np.nan
    return out


def _amp_features_block(per_trial_amp: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the five amp-stat features given a (G, n_trials) array of
    per-trial amplitudes (one amplitude per trial per group).
    """
    amp_mean = per_trial_amp.mean(axis=1)
    amp_var  = per_trial_amp.var(axis=1, ddof=0)
    amp_std  = np.sqrt(amp_var)
    cv   = _safe_div_vec(amp_std, amp_mean, eps=EPS_AMP_MEAN)
    fano = _safe_div_vec(amp_var, amp_mean, eps=EPS_AMP_MEAN)
    snr  = _safe_div_vec(amp_mean, amp_std, eps=EPS_AMP_STD)
    return {
        "b_amp_var":  amp_var,
        "b_amp_std":  amp_std,
        "b_amp_cv":   cv,
        "b_amp_fano": fano,
        "b_amp_snr":  snr,
    }


def _pairwise_pearson(traces: np.ndarray) -> np.ndarray:
    """For each group of shape (n_trials, n_frames), compute the mean of all
    C(n_trials, 2) pairwise Pearson correlations.

    Input  : (G, n_trials, n_frames) float64.
    Output : (G,) float64 — mean of pairwise correlations, NaN when no pair
             is well-defined (every trial flat).
    """
    G, T, F = traces.shape
    # Center each trial.
    centered = traces - traces.mean(axis=2, keepdims=True)
    # Per-trial std (used as denominator).
    var = (centered ** 2).mean(axis=2)            # (G, T)
    valid = var > EPS_TRACE_VAR                    # (G, T) — trials with non-zero variance
    norm = np.sqrt(var)                            # (G, T)

    # For each pair (i<j), compute corr = mean(centered_i * centered_j) / (norm_i * norm_j)
    # over frames. We accumulate the sum of valid corrs and the count of valid pairs.
    corr_sum = np.zeros(G, dtype=np.float64)
    pair_n   = np.zeros(G, dtype=np.int64)
    for i in range(T):
        for j in range(i + 1, T):
            cov = (centered[:, i, :] * centered[:, j, :]).mean(axis=1)  # (G,)
            den = norm[:, i] * norm[:, j]                                # (G,)
            ok = valid[:, i] & valid[:, j] & (den > 0)
            if ok.any():
                c = np.full(G, np.nan, dtype=np.float64)
                c[ok] = cov[ok] / den[ok]
                corr_sum = np.where(ok, corr_sum + c, corr_sum)
                pair_n   = np.where(ok, pair_n + 1, pair_n)
    out = np.full(G, np.nan, dtype=np.float64)
    nz = pair_n > 0
    out[nz] = corr_sum[nz] / pair_n[nz]
    return out


def _oracle_corr(traces: np.ndarray) -> np.ndarray:
    """Leave-one-trial-out correlation. For each trial i, correlate it with
    the mean of the other (n_trials - 1) trials; average across i.

    Input  : (G, n_trials, n_frames).
    Output : (G,).

    For n_trials = 2 this collapses to the single pair correlation (the LOO
    "average" is the other trial). Implementation handles n_trials >= 2.
    """
    G, T, F = traces.shape
    if T < 2:
        return np.full(G, np.nan, dtype=np.float64)

    # Sum across trials for fast LOO mean: leave_out_i_mean = (sum - trial_i) / (T - 1).
    total_sum = traces.sum(axis=1)  # (G, F)
    out = np.zeros(G, dtype=np.float64)
    pair_n = np.zeros(G, dtype=np.int64)
    for i in range(T):
        held = traces[:, i, :]                                 # (G, F)
        other_mean = (total_sum - held) / (T - 1)              # (G, F)
        # Pearson(held, other_mean), per group
        h_c = held - held.mean(axis=1, keepdims=True)
        o_c = other_mean - other_mean.mean(axis=1, keepdims=True)
        h_var = (h_c ** 2).mean(axis=1)
        o_var = (o_c ** 2).mean(axis=1)
        cov = (h_c * o_c).mean(axis=1)
        den = np.sqrt(h_var * o_var)
        ok = (h_var > EPS_TRACE_VAR) & (o_var > EPS_TRACE_VAR) & (den > 0)
        c = np.full(G, np.nan, dtype=np.float64)
        c[ok] = cov[ok] / den[ok]
        out = np.where(ok, out + c, out)
        pair_n = np.where(ok, pair_n + 1, pair_n)
    final = np.full(G, np.nan, dtype=np.float64)
    nz = pair_n > 0
    final[nz] = out[nz] / pair_n[nz]
    return final


def features_from_3d(traces_3d: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all 7 Tier B features given a (G, n_trials, n_frames) tensor
    where every group has the same (n_trials, n_frames) shape.
    """
    if traces_3d.ndim != 3:
        raise ValueError(f"expected 3-D, got {traces_3d.shape}")
    per_trial_amp = traces_3d.mean(axis=2)  # (G, n_trials)
    out = _amp_features_block(per_trial_amp)
    out["b_trace_corr_mean"] = _pairwise_pearson(traces_3d.astype(np.float64))
    out["b_oracle_corr"]     = _oracle_corr(traces_3d.astype(np.float64))
    return out


# ---------------------------------------------------------------------------
# Top-level: compute Tier B per (nucleus_id, condition_hash) from traces.parquet
# ---------------------------------------------------------------------------
def compute_tier_b(
    traces_df: pd.DataFrame,
    min_n_trials: int = 2,
    progress: bool = True,
) -> pd.DataFrame:
    """Compute Tier B features per (nucleus_id, condition_hash) from a long
    traces DataFrame.

    Strategy: group rows by (n_frames, n_trials_per_group), stack into a
    uniform 3-D tensor per shape, vectorize.

    Parameters
    ----------
    traces_df : DataFrame with columns nucleus_id, condition_hash, session_key,
                stim_type, n_frames, response (list[float]).
    min_n_trials : minimum trial count required for a (neuron, hash) pair to
                   appear in the output.
    progress : print per-shape progress if True.

    Returns
    -------
    DataFrame with one row per (nucleus_id, condition_hash) and columns:
      nucleus_id, condition_hash, session_key, stim_type, n_frames, n_trials,
      b_amp_var, b_amp_std, b_amp_cv, b_amp_fano, b_amp_snr,
      b_trace_corr_mean, b_oracle_corr.
    """
    keys = ["nucleus_id", "condition_hash", "session_key", "stim_type", "n_frames"]
    sizes = traces_df.groupby(keys, sort=False).size().rename("n_trials").reset_index()
    keep = sizes[sizes["n_trials"] >= min_n_trials].copy()
    if progress:
        print(f"[tier_b] {len(sizes):,} (neuron, hash) groups total; "
              f"{len(keep):,} retained with >= {min_n_trials} trials.")

    out_chunks: list[pd.DataFrame] = []
    # Group by (n_frames, n_trials) so every shape can be vectorized in one stack.
    for (n_frames, n_trials), shape_keys in keep.groupby(["n_frames", "n_trials"], sort=True):
        if progress:
            print(f"  shape (n_trials={n_trials}, n_frames={n_frames}): "
                  f"{len(shape_keys):,} groups", flush=True)

        # Subset traces_df to this shape.
        sub = traces_df.merge(shape_keys[keys], on=keys, how="inner")
        sub = sub.sort_values(keys + ["trial_idx"]).reset_index(drop=True)

        # Stack responses into (G, n_trials, n_frames). G = len(shape_keys).
        responses = np.array(sub["response"].tolist(), dtype=np.float32)  # (G * n_trials, n_frames)
        responses = responses.reshape(len(shape_keys), n_trials, n_frames)

        feats = features_from_3d(responses)
        # Build the chunk DataFrame.
        chunk = shape_keys[keys].reset_index(drop=True).copy()
        chunk["n_trials"] = int(n_trials)
        for k in B_FEATURE_NAMES:
            chunk[k] = feats[k]
        out_chunks.append(chunk)

    out = pd.concat(out_chunks, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Per-neuron summary
# ---------------------------------------------------------------------------
def per_neuron_summary(b_per_hash: pd.DataFrame) -> pd.DataFrame:
    """Average each Tier B feature across the neuron's repeated hashes.

    Returns a DataFrame with columns nucleus_id, b_summary_<feature> for each
    of the 7 features. Suffix `_summary` makes the per-neuron variant
    distinguishable from the per-hash original.
    """
    rename = {f: f"b_summary_{f.replace('b_', '')}" for f in B_FEATURE_NAMES}
    g = (b_per_hash.groupby("nucleus_id", sort=True)[list(B_FEATURE_NAMES)]
                   .mean()
                   .rename(columns=rename)
                   .reset_index())
    # Also store the count of (neuron, hash) rows that were averaged.
    n_hashes = (b_per_hash.groupby("nucleus_id", sort=True).size()
                .rename("b_summary_n_hashes").reset_index())
    return g.merge(n_hashes, on="nucleus_id")
