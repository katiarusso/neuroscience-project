"""Tier C — behavioural features.

WORKFLOW.md §4 Tier C and §3.4
==============================
Behaviour (treadmill, pupil) is **trial-level**: shared across all neurons
within a trial, but **varying across the repeated trials of a hash**. The
within-trial sharing means raw behaviour is not neuron-specific on its own —
its information toward a layer label only emerges through interaction with
the neural channel on the same row.

The workflow splits behaviour into two sub-tiers with very different status:

- **Tier C0 (trial-level scalars on top of A0).** Behavioural summaries
  computed per `(session, trial)` and broadcast onto every per-`(neuron,
  trial)` row of A0. Useful only via interaction with neural features —
  i.e. tree models can use C0 productively, linear models cannot without
  explicit interaction terms.
- **Tier C1 (behaviour-conditioned per-(neuron, hash) features on top of
  A1).** The workflow's preferred form. Per `(neuron, hash)`: partition the
  hash's trials by behavioural state (running vs still, high vs low pupil),
  compute neural statistics per state, take the modulation as the feature.
  Neuron-specific by construction; both linear and non-linear models can
  use it.

WORKFLOW §3.4 is the binding rule: "behavior-conditioned neural features
(Tier C1) are computed *before* the within-hash trial-averaging, not after".
We respect that here — C1 is computed from per-trial response amplitudes
and per-trial behavioural states, then summarised per `(neuron, hash)`.

C0 features (8 features, per `(session, trial)`)
================================================
  c0_run_mean       : mean treadmill speed over the trial
  c0_run_max        : max treadmill speed over the trial
  c0_run_std        : std of treadmill speed over the trial
  c0_pupil_dil_mean : mean pupil dilation
  c0_pupil_dil_std  : std of pupil dilation
  c0_pupil_dil_change : end-of-trial − start-of-trial pupil dilation (drift)
  c0_pupil_x_std    : std of pupil x-position (gaze stability)
  c0_pupil_y_std    : std of pupil y-position

C1 features (4 features, per `(nucleus_id, condition_hash)`)
============================================================
  c1_rmi               : running modulation index
                         = (mean_resp_running − mean_resp_still)
                         / (mean_resp_running + mean_resp_still + EPS)
                         in [−1, +1]; NaN if either state is empty.
  c1_pupil_resp_slope  : slope of per-trial response amplitude on per-trial
                         pupil dilation, within the hash. Closed-form OLS.
                         NaN if pupil variability is too small.
  c1_state_rel_diff    : reliability (b_amp_std-style) on the running subset
                         minus on the still subset. Captures whether trial-
                         to-trial reliability *itself* depends on behaviour.
                         NaN if either subset has < 2 trials.
  c1_arousal_gain_diff : same as RMI but for high-pupil vs low-pupil.

Behavioural state thresholds — per session, percentile-based
============================================================
For every session we compute the 25th and 75th percentile of
`c0_run_mean` and `c0_pupil_dil_mean` across that session's trials, and:

  - "running"     trial = c0_run_mean      ≥ session 75th percentile
  - "still"       trial = c0_run_mean      ≤ session 25th percentile
  - "high-pupil"  trial = c0_pupil_dil_mean ≥ session 75th percentile
  - "low-pupil"   trial = c0_pupil_dil_mean ≤ session 25th percentile

Per-session thresholds are robust to per-session offsets in treadmill /
pupil calibration (Stage-2's lesson). Trials that fall in the middle are
excluded from the running-vs-still partition (their behavioural state is
ambiguous).

Numerical safety
================
All ratios use EPS_DENOM_C1 floors on denominators. Modulation indices for
silent or near-silent neurons are NaN rather than 0/0 = NaN-from-numpy.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


EPS_DENOM_C1: Final[float] = 1e-3   # for modulation-index denominators
EPS_PUPIL_VAR: Final[float] = 1e-6  # for pupil-response slope denominator


# ---------------------------------------------------------------------------
# Tier C0 — per-(session, trial) behavioural scalars
# ---------------------------------------------------------------------------
C0_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "c0_run_mean",
    "c0_run_max",
    "c0_run_std",
    "c0_pupil_dil_mean",
    "c0_pupil_dil_std",
    "c0_pupil_dil_change",
    "c0_pupil_x_std",
    "c0_pupil_y_std",
)


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def compute_tier_c0(trials_meta_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(session_key, trial_idx) behavioural scalars from trials_meta.

    Input: DataFrame with columns session_key, trial_idx, treadmill,
           pupil_dilation, pupil_pos_x, pupil_pos_y (all list[float]).
    Output: DataFrame with key columns session_key, trial_idx + the 8
            C0 feature columns above.
    """
    out_rows: list[dict] = []
    for _, row in trials_meta_df.iterrows():
        tread = _arr(row["treadmill"]).reshape(-1)
        dil   = _arr(row["pupil_dilation"]).reshape(-1)
        px    = _arr(row["pupil_pos_x"]).reshape(-1)
        py    = _arr(row["pupil_pos_y"]).reshape(-1)
        out_rows.append({
            "session_key": row["session_key"],
            "trial_idx":   int(row["trial_idx"]),
            "c0_run_mean":         float(np.nanmean(tread)) if tread.size else np.nan,
            "c0_run_max":          float(np.nanmax(tread))  if tread.size else np.nan,
            "c0_run_std":          float(np.nanstd(tread, ddof=0)) if tread.size else np.nan,
            "c0_pupil_dil_mean":   float(np.nanmean(dil))   if dil.size   else np.nan,
            "c0_pupil_dil_std":    float(np.nanstd(dil, ddof=0)) if dil.size else np.nan,
            "c0_pupil_dil_change": float(dil[-1] - dil[0])  if dil.size >= 2 else np.nan,
            "c0_pupil_x_std":      float(np.nanstd(px, ddof=0)) if px.size else np.nan,
            "c0_pupil_y_std":      float(np.nanstd(py, ddof=0)) if py.size else np.nan,
        })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Behavioural state assignment — per-session percentile thresholds
# ---------------------------------------------------------------------------
def assign_behavioural_state(
    c0_df: pd.DataFrame,
    pct_high: float = 75.0,
    pct_low: float = 25.0,
) -> pd.DataFrame:
    """Add categorical state columns to a C0 DataFrame.

    For each session_key, compute the `pct_high` and `pct_low` percentiles
    of `c0_run_mean` and `c0_pupil_dil_mean`. Tag each trial as:

      run_state   = 'running' if c0_run_mean >= session 75th
                    'still'   if c0_run_mean <= session 25th
                    'mid'     otherwise
      pupil_state = 'high'    if c0_pupil_dil_mean >= session 75th
                    'low'     if c0_pupil_dil_mean <= session 25th
                    'mid'     otherwise

    Returns a new DataFrame with extra columns: run_state, pupil_state.
    """
    out = c0_df.copy()
    out["run_state"] = "mid"
    out["pupil_state"] = "mid"
    for sk, sub in out.groupby("session_key"):
        rm_hi = np.nanpercentile(sub["c0_run_mean"], pct_high)
        rm_lo = np.nanpercentile(sub["c0_run_mean"], pct_low)
        pd_hi = np.nanpercentile(sub["c0_pupil_dil_mean"], pct_high)
        pd_lo = np.nanpercentile(sub["c0_pupil_dil_mean"], pct_low)
        idx = sub.index
        out.loc[idx[sub["c0_run_mean"]      >= rm_hi], "run_state"]   = "running"
        out.loc[idx[sub["c0_run_mean"]      <= rm_lo], "run_state"]   = "still"
        out.loc[idx[sub["c0_pupil_dil_mean"] >= pd_hi], "pupil_state"] = "high"
        out.loc[idx[sub["c0_pupil_dil_mean"] <= pd_lo], "pupil_state"] = "low"
    return out


# ---------------------------------------------------------------------------
# Tier C1 — per-(neuron, hash) behaviour-conditioned features
# ---------------------------------------------------------------------------
C1_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "c1_rmi",
    "c1_pupil_resp_slope",
    "c1_state_rel_diff",
    "c1_arousal_gain_diff",
)


def _modulation_index(mean_a: float, mean_b: float, eps: float = EPS_DENOM_C1) -> float:
    """(a − b) / (a + b), NaN if denominator below eps in absolute value."""
    den = mean_a + mean_b
    if not np.isfinite(den) or abs(den) <= eps:
        return float("nan")
    out = (mean_a - mean_b) / den
    return float(out) if np.isfinite(out) else float("nan")


def _ols_slope(x: np.ndarray, y: np.ndarray, eps: float = EPS_PUPIL_VAR) -> float:
    """Closed-form OLS slope of y on x. NaN when var(x) ≤ eps."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    mx, my = float(x.mean()), float(y.mean())
    vx = float(((x - mx) ** 2).mean())
    if not np.isfinite(vx) or vx <= eps:
        return float("nan")
    cov = float(((x - mx) * (y - my)).mean())
    out = cov / vx
    return out if np.isfinite(out) else float("nan")


def compute_tier_c1(
    a0_amp_df: pd.DataFrame,
    c0_df_with_state: pd.DataFrame,
    min_per_state: int = 1,
) -> pd.DataFrame:
    """Per-`(nucleus_id, condition_hash)` behaviour-conditioned features.

    Workflow §3.4 binding rule: Tier C1 is computed from **per-trial**
    response amplitudes (not from the within-hash trial-averaged trace),
    so we read `amp_mean` from the A0_amp parquet rather than recomputing.

    Parameters
    ----------
    a0_amp_df : the A0_amp.parquet, with at least nucleus_id, session_key,
                trial_idx, condition_hash, amp_mean.
    c0_df_with_state : output of `assign_behavioural_state`, with run_state
                       and pupil_state columns per (session_key, trial_idx).
    min_per_state : minimum number of trials per state required to compute
                    a state-conditional statistic. Default 1.

    Returns
    -------
    DataFrame with columns nucleus_id, condition_hash, session_key, n_trials,
    n_running, n_still, n_high_pupil, n_low_pupil, and the 4 C1 feature
    columns above. One row per `(nucleus_id, condition_hash)` for which at
    least one of the four C1 features is computable (others are NaN).
    """
    state_cols = ["session_key", "trial_idx", "run_state", "pupil_state",
                  "c0_pupil_dil_mean"]
    df = a0_amp_df.merge(
        c0_df_with_state[state_cols],
        on=["session_key", "trial_idx"],
        how="inner",
        validate="many_to_one",
    )
    out_rows: list[dict] = []
    for (nid, ch), grp in df.groupby(["nucleus_id", "condition_hash"], sort=False):
        amp = grp["amp_mean"].to_numpy(dtype=np.float64)
        run_state = grp["run_state"].to_numpy()
        pupil_state = grp["pupil_state"].to_numpy()
        pupil_dil = grp["c0_pupil_dil_mean"].to_numpy(dtype=np.float64)

        is_run   = run_state == "running"
        is_still = run_state == "still"
        is_hi    = pupil_state == "high"
        is_lo    = pupil_state == "low"

        # RMI — running modulation index.
        rmi = float("nan")
        if is_run.sum() >= min_per_state and is_still.sum() >= min_per_state:
            rmi = _modulation_index(float(amp[is_run].mean()),
                                    float(amp[is_still].mean()))

        # Arousal gain difference — pupil-state RMI analogue.
        agd = float("nan")
        if is_hi.sum() >= min_per_state and is_lo.sum() >= min_per_state:
            agd = _modulation_index(float(amp[is_hi].mean()),
                                    float(amp[is_lo].mean()))

        # Pupil-response slope.
        slope = _ols_slope(pupil_dil, amp)

        # State-conditional reliability difference: std-of-amp on running −
        # std-of-amp on still. Each state needs ≥ 2 trials for std to make
        # sense.
        rel_diff = float("nan")
        if is_run.sum() >= 2 and is_still.sum() >= 2:
            std_run   = float(amp[is_run].std(ddof=0))
            std_still = float(amp[is_still].std(ddof=0))
            rel_diff = std_run - std_still

        out_rows.append({
            "nucleus_id":        int(nid),
            "condition_hash":    ch,
            "session_key":       grp["session_key"].iloc[0],
            "n_trials":          int(len(amp)),
            "n_running":         int(is_run.sum()),
            "n_still":           int(is_still.sum()),
            "n_high_pupil":      int(is_hi.sum()),
            "n_low_pupil":       int(is_lo.sum()),
            "c1_rmi":               rmi,
            "c1_pupil_resp_slope":  slope,
            "c1_state_rel_diff":    rel_diff,
            "c1_arousal_gain_diff": agd,
        })
    return pd.DataFrame(out_rows)
