"""Tier A — neural-response scalar features per trace.

WORKFLOW.md §4 Tier A names two sub-blocks:

  - **A-amp** (amplitude / statistical) — scalars that summarise *how strong*
    and *how dispersed* the response is.
  - **A-shape** (temporal shape) — scalars that summarise *how the response
    unfolds in time*.

The exact same functions are applied at two granularities:
  - per-trial traces (Tier A0 — `(neuron, trial)` granularity), and
  - within-hash trial-averaged traces (Tier A1 — `(neuron, hash)` granularity).
The distinction A0 vs A1 is *which trace you call the function on*, not which
function you call. WORKFLOW §4 A1 is explicit: do not average A0 features
across trials of a hash; average the *trace* first, then compute the
features on the averaged trace.

Features (10 amp + 10 shape = 20 scalars per row)
=================================================

A-amp (10):
    amp_baseline_mean : mean of the first `baseline_window_s` (default 1 s)
    amp_baseline_std  : std  of the first `baseline_window_s`
    amp_mean          : mean of the full trace
    amp_peak          : max of the full trace
    amp_min           : min of the full trace
                        (deconvolved data is ≥ 0, so usually 0 — kept as a
                         sanity / dynamic-range column)
    amp_range         : peak − baseline_mean (response amplitude above baseline)
    amp_integral      : sum of the full trace (≡ AUC, ≡ mean × n_frames)
    amp_tail_mean     : mean of the last `tail_window_s`  (default 1 s)
    amp_variance      : variance of the full trace
    amp_std           : std of the full trace (sqrt of variance, kept for
                        readability — trees treat the two as equivalent)

A-shape (10):
    shape_latency_s        : argmax position in seconds (= peak time)
    shape_peak_frame_idx   : argmax position in frames (integer; same info
                             as latency_s scaled by sampling rate, kept for
                             explicit readability)
    shape_rise_slope       : (peak − baseline_mean) / latency_s
                             [trace-units / s]; NaN if latency_s = 0
    shape_decay_slope      : (peak − tail_mean) / (duration − latency_s)
                             [trace-units / s]; NaN if peak is at last frame
    shape_early_auc        : sum of the first half of the trace
    shape_late_auc         : sum of the second half
    shape_early_late_ratio : early_auc / late_auc; NaN if late_auc ≤ EPS_DENOM
    shape_adaptation_ratio : last-quartile mean / first-quartile mean;
                             NaN if first-quartile mean ≤ EPS_DENOM
    shape_center_of_mass_s : weighted mean of frame times by trace amplitude,
                             in seconds; NaN if integral ≤ EPS_DENOM
    shape_width_above_half_max_s
                           : duration in seconds where trace ≥ peak/2;
                             NaN if peak ≤ EPS_DENOM (no signal)

Notes on the "pre-stim baseline" and "post-stim tail" features
==============================================================
Inspection of the H5 shows the calcium trace covers exactly the stimulus
window (no pre-stim / post-stim padding). Values are deconvolved activity
(non-negative, zero-floored, sparse). We therefore reinterpret the workflow's
"pre-stim baseline" and "post-stim tail" as **early-window** and **late-window**
means *inside the trace*, both 1 s long by default (≈ 8 frames at 7.5 Hz).
The notebook documents this transparently. Cross-stimulus-family comparability
is achieved by defining all temporal quantities in **seconds** (or fractional
positions for the AUCs and quartiles), not frames.

Numerical safety
================
Edge cases — peak at frame 0 (rise undefined), peak at last frame (decay
undefined), flat / all-zero trace (every ratio undefined) — return NaN, not
inf. Each safe-divide uses an **absolute, biologically-sized threshold** on
the denominator, distinct per quantity (raw trace value vs. mean vs.
half-AUC vs. integral vs. peak) — see `EPS_TRACE`, `EPS_MEAN`, `EPS_AUC`,
`EPS_INTEGRAL`, `EPS_PEAK` in this module. A single global epsilon is wrong
because integrated quantities (AUC ≈ 25–250) and averaged quantities
(mean ≈ 0–5) live on different scales.
"""

from __future__ import annotations

from typing import Final

import numpy as np


CALCIUM_SAMPLING_RATE_HZ: Final[float] = 7.5  # WORKFLOW §2.2; verified in EDA

# Per-quantity "essentially zero" thresholds, biologically sized.
#
# Why per-quantity rather than one global eps: an AUC sum lives on a scale of
# 25–250 (typical activity per frame × ~40 frames), a mean lives on a scale of
# 0–5, a peak on a scale of 1–100. A single 1e-6 threshold is correct for raw
# trace values but lets through "essentially zero" denominators of any of the
# integrated / averaged quantities, which then produce 1e+7-ish ratios. The
# values below are calibrated against the empirical feature distributions:
# they are well below any genuine neural response and orders of magnitude
# above float-arithmetic noise.
EPS_TRACE: Final[float]    = 1e-6  # raw trace values
EPS_MEAN: Final[float]     = 0.1   # means (e.g. first-quartile mean)
EPS_AUC: Final[float]      = 1.0   # half-trace sums (early_auc, late_auc)
EPS_INTEGRAL: Final[float] = 1.0   # full-trace sum (used by center-of-mass)
EPS_PEAK: Final[float]     = 0.5   # peak (used by width-above-half-max)

# Backward-compatible alias used by anything that imports `EPS_DENOM`.
EPS_DENOM: Final[float] = EPS_TRACE


# ---------------------------------------------------------------------------
# Feature column conventions
# ---------------------------------------------------------------------------
AMP_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "amp_baseline_mean",
    "amp_baseline_std",
    "amp_mean",
    "amp_peak",
    "amp_min",
    "amp_range",
    "amp_integral",
    "amp_tail_mean",
    "amp_variance",
    "amp_std",
)

SHAPE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "shape_latency_s",
    "shape_peak_frame_idx",
    "shape_rise_slope",
    "shape_decay_slope",
    "shape_early_auc",
    "shape_late_auc",
    "shape_early_late_ratio",
    "shape_adaptation_ratio",
    "shape_center_of_mass_s",
    "shape_width_above_half_max_s",
)

ALL_FEATURE_NAMES: Final[tuple[str, ...]] = AMP_FEATURE_NAMES + SHAPE_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def _safe_div_scalar(num: float, den: float, eps: float = EPS_TRACE) -> float:
    """Scalar safe-divide: NaN when |den| ≤ eps, else num / den.

    Pass `eps` explicitly with a biologically sized threshold (e.g. `EPS_AUC`,
    `EPS_MEAN`) when dividing by integrated / averaged quantities — the
    default `EPS_TRACE` only catches denominators in raw-trace units.
    """
    if not np.isfinite(den) or abs(den) <= eps:
        return float("nan")
    out = num / den
    return float(out) if np.isfinite(out) else float("nan")


def _safe_div_vec(num: np.ndarray, den: np.ndarray, eps: float = EPS_TRACE) -> np.ndarray:
    """Vectorized safe-divide. NaN where |den| ≤ eps. See `_safe_div_scalar`."""
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = np.isfinite(den) & (np.abs(den) > eps)
    out[mask] = num[mask] / den[mask]
    out[~np.isfinite(out)] = np.nan
    return out


def _window_n_frames(window_s: float, sampling_rate: float, n_frames: int) -> int:
    """Number of frames for a given seconds-window. Always at least 1, never
    larger than the full trace.
    """
    n = int(round(window_s * sampling_rate))
    return max(1, min(n, n_frames))


# ---------------------------------------------------------------------------
# Per-trace functions — clear, readable, slow
# ---------------------------------------------------------------------------
def amplitude_features(
    trace: np.ndarray,
    sampling_rate: float = CALCIUM_SAMPLING_RATE_HZ,
    baseline_window_s: float = 1.0,
    tail_window_s: float = 1.0,
) -> dict[str, float]:
    """Compute the 10 A-amp scalars for one trace."""
    x = np.asarray(trace, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return {k: float("nan") for k in AMP_FEATURE_NAMES}

    n_base = _window_n_frames(baseline_window_s, sampling_rate, n)
    n_tail = _window_n_frames(tail_window_s, sampling_rate, n)

    base = x[:n_base]
    tail = x[-n_tail:]
    baseline_mean = float(base.mean())
    peak = float(x.max())

    return {
        "amp_baseline_mean": baseline_mean,
        "amp_baseline_std":  float(base.std(ddof=0)),
        "amp_mean":          float(x.mean()),
        "amp_peak":          peak,
        "amp_min":           float(x.min()),
        "amp_range":         peak - baseline_mean,
        "amp_integral":      float(x.sum()),
        "amp_tail_mean":     float(tail.mean()),
        "amp_variance":      float(x.var(ddof=0)),
        "amp_std":           float(x.std(ddof=0)),
    }


def shape_features(
    trace: np.ndarray,
    sampling_rate: float = CALCIUM_SAMPLING_RATE_HZ,
    baseline_window_s: float = 1.0,
    tail_window_s: float = 1.0,
) -> dict[str, float]:
    """Compute the 10 A-shape scalars for one trace."""
    x = np.asarray(trace, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return {k: float("nan") for k in SHAPE_FEATURE_NAMES}

    n_base = _window_n_frames(baseline_window_s, sampling_rate, n)
    n_tail = _window_n_frames(tail_window_s, sampling_rate, n)
    duration_s = (n - 1) / sampling_rate if sampling_rate > 0 else float("nan")

    baseline_mean = float(x[:n_base].mean())
    tail_mean = float(x[-n_tail:].mean())
    peak_idx = int(np.argmax(x))
    peak = float(x[peak_idx])
    latency_s = peak_idx / sampling_rate if sampling_rate > 0 else float("nan")

    # Rise: from baseline to peak. Latency in seconds, eps in seconds: 1e-6 s
    # is below one-frame granularity, so any non-zero peak frame index is fine.
    rise_slope = _safe_div_scalar(peak - baseline_mean, latency_s, eps=EPS_TRACE)

    # Decay: from peak to tail. Same time-units denominator.
    remaining_s = duration_s - latency_s
    decay_slope = _safe_div_scalar(peak - tail_mean, remaining_s, eps=EPS_TRACE)

    # Halves and quartiles (fractional positions, family-comparable).
    half = n // 2
    early_auc = float(x[:half].sum())
    late_auc  = float(x[half:].sum())
    # AUC denominator: half-trace sum on a scale of 0–250 typical.
    early_late_ratio = _safe_div_scalar(early_auc, late_auc, eps=EPS_AUC)

    q = max(1, n // 4)
    first_q_mean = float(x[:q].mean())
    last_q_mean  = float(x[-q:].mean())
    # Mean denominator: first-quartile mean on a scale of 0–5 typical.
    adaptation_ratio = _safe_div_scalar(last_q_mean, first_q_mean, eps=EPS_MEAN)

    # Temporal center of mass: ∑(t·x) / ∑x in seconds.
    integral = float(x.sum())
    if integral > EPS_INTEGRAL:
        t = np.arange(n, dtype=np.float64) / sampling_rate
        com_s = float((t * x).sum() / integral)
    else:
        com_s = float("nan")

    # Width above half-max: count of frames where x ≥ peak/2, in seconds.
    if peak > EPS_PEAK:
        width_s = float(np.sum(x >= peak / 2.0)) / sampling_rate
    else:
        width_s = float("nan")

    return {
        "shape_latency_s":              float(latency_s),
        "shape_peak_frame_idx":         float(peak_idx),
        "shape_rise_slope":             float(rise_slope),
        "shape_decay_slope":            float(decay_slope),
        "shape_early_auc":              early_auc,
        "shape_late_auc":               late_auc,
        "shape_early_late_ratio":       float(early_late_ratio),
        "shape_adaptation_ratio":       float(adaptation_ratio),
        "shape_center_of_mass_s":       com_s,
        "shape_width_above_half_max_s": width_s,
    }


def all_features(
    trace: np.ndarray,
    sampling_rate: float = CALCIUM_SAMPLING_RATE_HZ,
    baseline_window_s: float = 1.0,
    tail_window_s: float = 1.0,
) -> dict[str, float]:
    """Concatenate amplitude + shape blocks into one dict."""
    out = amplitude_features(
        trace,
        sampling_rate=sampling_rate,
        baseline_window_s=baseline_window_s,
        tail_window_s=tail_window_s,
    )
    out.update(
        shape_features(
            trace,
            sampling_rate=sampling_rate,
            baseline_window_s=baseline_window_s,
            tail_window_s=tail_window_s,
        )
    )
    return out


# ---------------------------------------------------------------------------
# Vectorized batch helpers — same math, numpy-vectorized
# ---------------------------------------------------------------------------
def batch_amplitude_features(
    traces: np.ndarray,
    sampling_rate: float = CALCIUM_SAMPLING_RATE_HZ,
    baseline_window_s: float = 1.0,
    tail_window_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Vectorized A-amp on a 2-D `(N, n_frames)` array."""
    x = np.asarray(traces, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D, got shape {x.shape}")
    N, n_frames = x.shape
    n_base = _window_n_frames(baseline_window_s, sampling_rate, n_frames)
    n_tail = _window_n_frames(tail_window_s, sampling_rate, n_frames)

    base_mean = x[:, :n_base].mean(axis=1)
    base_std  = x[:, :n_base].std(axis=1, ddof=0)
    peak = x.max(axis=1)

    return {
        "amp_baseline_mean": base_mean,
        "amp_baseline_std":  base_std,
        "amp_mean":          x.mean(axis=1),
        "amp_peak":          peak,
        "amp_min":           x.min(axis=1),
        "amp_range":         peak - base_mean,
        "amp_integral":      x.sum(axis=1),
        "amp_tail_mean":     x[:, -n_tail:].mean(axis=1),
        "amp_variance":      x.var(axis=1, ddof=0),
        "amp_std":           x.std(axis=1, ddof=0),
    }


def batch_shape_features(
    traces: np.ndarray,
    sampling_rate: float = CALCIUM_SAMPLING_RATE_HZ,
    baseline_window_s: float = 1.0,
    tail_window_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Vectorized A-shape on a 2-D `(N, n_frames)` array."""
    x = np.asarray(traces, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D, got shape {x.shape}")
    N, n_frames = x.shape
    n_base = _window_n_frames(baseline_window_s, sampling_rate, n_frames)
    n_tail = _window_n_frames(tail_window_s, sampling_rate, n_frames)
    duration_s = (n_frames - 1) / sampling_rate

    baseline_mean = x[:, :n_base].mean(axis=1)
    tail_mean     = x[:, -n_tail:].mean(axis=1)
    peak_idx = x.argmax(axis=1)
    peak = x.max(axis=1)
    latency_s = peak_idx.astype(np.float64) / sampling_rate

    rise_slope  = _safe_div_vec(peak - baseline_mean, latency_s, eps=EPS_TRACE)
    remaining_s = duration_s - latency_s
    decay_slope = _safe_div_vec(peak - tail_mean, remaining_s, eps=EPS_TRACE)

    half = n_frames // 2
    early_auc = x[:, :half].sum(axis=1)
    late_auc  = x[:, half:].sum(axis=1)
    # AUC denominator: half-trace sum, biologically O(25–250).
    early_late_ratio = _safe_div_vec(early_auc, late_auc, eps=EPS_AUC)

    q = max(1, n_frames // 4)
    first_q_mean = x[:, :q].mean(axis=1)
    last_q_mean  = x[:, -q:].mean(axis=1)
    # Mean denominator: first-quartile mean, biologically O(0–5).
    adaptation_ratio = _safe_div_vec(last_q_mean, first_q_mean, eps=EPS_MEAN)

    # Temporal center of mass.
    integral = x.sum(axis=1)
    t = np.arange(n_frames, dtype=np.float64) / sampling_rate
    com_s = _safe_div_vec((x * t[None, :]).sum(axis=1), integral, eps=EPS_INTEGRAL)

    # Width above half-max.
    width_s = np.full(N, np.nan, dtype=np.float64)
    has_signal = peak > EPS_PEAK
    if has_signal.any():
        sub = x[has_signal]
        sub_peak = peak[has_signal][:, None]
        width_s[has_signal] = (sub >= sub_peak / 2.0).sum(axis=1) / sampling_rate

    return {
        "shape_latency_s":              latency_s,
        "shape_peak_frame_idx":         peak_idx.astype(np.float64),
        "shape_rise_slope":             rise_slope,
        "shape_decay_slope":            decay_slope,
        "shape_early_auc":              early_auc,
        "shape_late_auc":               late_auc,
        "shape_early_late_ratio":       early_late_ratio,
        "shape_adaptation_ratio":       adaptation_ratio,
        "shape_center_of_mass_s":       com_s,
        "shape_width_above_half_max_s": width_s,
    }


def batch_all_features(
    traces: np.ndarray,
    sampling_rate: float = CALCIUM_SAMPLING_RATE_HZ,
    baseline_window_s: float = 1.0,
    tail_window_s: float = 1.0,
) -> dict[str, np.ndarray]:
    out = batch_amplitude_features(
        traces,
        sampling_rate=sampling_rate,
        baseline_window_s=baseline_window_s,
        tail_window_s=tail_window_s,
    )
    out.update(
        batch_shape_features(
            traces,
            sampling_rate=sampling_rate,
            baseline_window_s=baseline_window_s,
            tail_window_s=tail_window_s,
        )
    )
    return out
