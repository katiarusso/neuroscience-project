"""Tier D — stimulus-descriptor features per `condition_hash`.

WORKFLOW.md §4 Tier D
=====================
"Stimulus descriptors are **hash-level** because all trials of the same hash
show the same stimulus. They can therefore be safely attached at either
granularity without conditional-averaging machinery."

  - **Tier D0**: hash-level descriptors broadcast onto A0 rows (one per trial).
  - **Tier D1**: hash-level descriptors broadcast onto A1 rows (one per
    `(neuron, hash)` after within-hash trial averaging).

This module computes the **per-hash** descriptors once. Whether they are
then attached at the A0 or A1 granularity is a downstream decision.

Features computed (10 numeric + stimulus-family categorical)
============================================================
  d_luminance_mean      : mean pixel value across all frames (in [0,1] after /255).
  d_luminance_std       : std of pixel values across all frames.
  d_contrast            : std/mean of pixel values (Michelson-like).
  d_motion_energy       : mean absolute frame-to-frame difference across pixels.
  d_sf_low / mid / high : fraction of 2-D spatial-FFT power in three radial
                          bands (relative to Nyquist 0.5 cycles/pixel):
                          low ∈ [0, 0.1), mid ∈ [0.1, 0.3), high ∈ [0.3, 0.5].
                          Normalized so the three sum to 1 per video.
  d_tf_low / mid / high : fraction of 1-D temporal-FFT power (per pixel,
                          averaged across pixels) in three bands relative
                          to Nyquist of frame rate (7.5 Hz / 2 = 3.75 Hz).
                          Same low/mid/high cutoffs in normalized frequency.

stim_type is returned as a column with values in {Clip, Monet2, Trippy};
downstream code one-hot-encodes it.

Numerical safety / normalization
================================
- Pixels normalized to [0, 1] by dividing the uint8 video by 255.
- The radial-frequency bins are defined in *normalized frequency* (cycles/
  pixel for SF, cycles/frame for TF), so descriptors are comparable across
  the three stimulus families even though they have different (H, W, n_frames):
    Clip   : (75, 144, 256)
    Monet2 : (113, 126, 216)
    Trippy : (113,  90, 160)

What we explicitly do NOT compute (WORKFLOW §3.5 binding)
=========================================================
- No orientation / direction features.
- No `pref_ori`, `pref_dir`, `gOSI`, `gDSI`, or any
  orientation-/direction-selectivity scalar.
- Even though Monet2 is a parametric grating with known direction structure
  and we *could* extract orientation cleanly, we don't — per the workflow's
  Tier-F note, orientation is out of scope for the entire project.
"""

from __future__ import annotations

from typing import Final, Iterable

import numpy as np
import pandas as pd


D_FEATURE_NAMES: Final[tuple[str, ...]] = (
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


# Normalized-frequency thresholds (in [0, 0.5] for both SF and TF, where 0.5 is
# the Nyquist limit). Chosen as "low / mid / high" intuitive bands.
_FREQ_LOW_MID: Final[float] = 0.10
_FREQ_MID_HIGH: Final[float] = 0.30


def _compute_video_features(video: np.ndarray) -> dict[str, float]:
    """Compute the 10 numeric Tier-D descriptors on a single video.

    Parameters
    ----------
    video : (n_frames, H, W) uint8 array.

    Returns dict keyed by `D_FEATURE_NAMES`.
    """
    if video.ndim != 3:
        raise ValueError(f"expected 3-D video, got shape {video.shape}")
    n_frames, H, W = video.shape
    v = video.astype(np.float32) / 255.0  # normalize to [0, 1]

    # ---- Amplitude / luminance descriptors ----
    lum_mean = float(v.mean())
    lum_std = float(v.std(ddof=0))
    contrast = float(lum_std / lum_mean) if lum_mean > 1e-6 else float("nan")

    # ---- Motion energy: mean |Δ frame| ----
    if n_frames > 1:
        motion_energy = float(np.abs(np.diff(v, axis=0)).mean())
    else:
        motion_energy = 0.0

    # ---- Spatial-frequency content ----
    # rfft2 over each frame, in one batched call along axes (1,2).
    F2 = np.fft.rfft2(v, axes=(1, 2))           # (n_frames, H, W//2+1)
    spatial_power = (np.abs(F2) ** 2).mean(axis=0)  # (H, W//2+1)
    # Radial frequencies in cycles/pixel.
    ky = np.fft.fftfreq(H)[:, None]              # (H, 1)
    kx = np.fft.rfftfreq(W)[None, :]             # (1, W//2+1)
    radial = np.sqrt(ky**2 + kx**2)              # (H, W//2+1)
    sf_low_p  = float(spatial_power[radial < _FREQ_LOW_MID].sum())
    sf_mid_p  = float(spatial_power[(radial >= _FREQ_LOW_MID) & (radial < _FREQ_MID_HIGH)].sum())
    sf_high_p = float(spatial_power[radial >= _FREQ_MID_HIGH].sum())
    sf_total = sf_low_p + sf_mid_p + sf_high_p
    if sf_total > 0:
        sf_low, sf_mid, sf_high = sf_low_p/sf_total, sf_mid_p/sf_total, sf_high_p/sf_total
    else:
        sf_low = sf_mid = sf_high = float("nan")

    # ---- Temporal-frequency content ----
    if n_frames > 1:
        T_F = np.fft.rfft(v, axis=0)             # (n_frames//2+1, H, W)
        # Mean power across spatial pixels: how much each TF band contributes overall.
        T_power = (np.abs(T_F) ** 2).mean(axis=(1, 2))   # (n_frames//2+1,)
        t_freq = np.fft.rfftfreq(n_frames)                # in cycles/frame, range [0, 0.5]
        tf_low_p  = float(T_power[t_freq < _FREQ_LOW_MID].sum())
        tf_mid_p  = float(T_power[(t_freq >= _FREQ_LOW_MID) & (t_freq < _FREQ_MID_HIGH)].sum())
        tf_high_p = float(T_power[t_freq >= _FREQ_MID_HIGH].sum())
        tf_total = tf_low_p + tf_mid_p + tf_high_p
        if tf_total > 0:
            tf_low, tf_mid, tf_high = tf_low_p/tf_total, tf_mid_p/tf_total, tf_high_p/tf_total
        else:
            tf_low = tf_mid = tf_high = float("nan")
    else:
        tf_low = tf_mid = tf_high = float("nan")

    return {
        "d_luminance_mean": lum_mean,
        "d_luminance_std":  lum_std,
        "d_contrast":       contrast,
        "d_motion_energy":  motion_energy,
        "d_sf_low":  sf_low,
        "d_sf_mid":  sf_mid,
        "d_sf_high": sf_high,
        "d_tf_low":  tf_low,
        "d_tf_mid":  tf_mid,
        "d_tf_high": tf_high,
    }


#: Per-frame stimulus channel names exposed by `compute_per_frame_video_features`.
#: These are the *time-varying* analogues of (a subset of) the scalar Tier-D
#: descriptors. They are computed at the native video frame index so each
#: oracle hash produces a feature matrix of shape `(n_frames, n_channels)` with
#: the same `n_frames` as the calcium `response_avg` for that hash family
#: (Clip = 75, Monet2/Trippy = 113). Padding/truncation to a common length
#: is the loader's responsibility, not the feature computer's.
PER_FRAME_STIM_CHANNELS: Final[tuple[str, ...]] = (
    "stim_lum_mean",        # per-frame mean pixel intensity (in [0, 1])
    "stim_lum_std",         # per-frame pixel-intensity std
    "stim_norm_contrast",   # per-frame std / mean (Michelson-like)
    "stim_motion_energy",   # per-frame mean |Δ frame|; first frame = 0
    "stim_grad_energy",     # per-frame mean magnitude of spatial gradient
    "stim_edge_frac",       # per-frame fraction of pixels above the 85th-pct gradient
    "stim_center_surround", # per-frame (centre mean − surround mean) of pixel intensity
)


def compute_per_frame_video_features(
    video: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-frame stimulus descriptors for one oracle video.

    The seven channels are the time-varying analogues of (a subset of) the
    scalar Tier-D descriptors used elsewhere in this project. They follow the
    feature definitions in Tommy's reference multichannel CNN, restated for
    direct frame-level access without resampling:

    * `stim_lum_mean(t)`        = mean pixel intensity at frame `t` (∈ [0, 1])
    * `stim_lum_std(t)`         = pixel-intensity std at frame `t`
    * `stim_norm_contrast(t)`   = std/mean (clamped) at frame `t`
    * `stim_motion_energy(t)`   = mean |frame[t] − frame[t-1]|; 0 at `t = 0`
    * `stim_grad_energy(t)`     = mean magnitude of the spatial gradient at frame `t`
    * `stim_edge_frac(t)`       = fraction of pixels at frame `t` above the 85th
                                   percentile of the per-frame gradient magnitude
    * `stim_center_surround(t)` = centre-region mean − surround-region mean at
                                   frame `t` (centre = inner half in each axis)

    Parameters
    ----------
    video : (n_frames, H, W) array, ideally `uint8` (cast to float32/255 internally).

    Returns
    -------
    dict mapping each name in `PER_FRAME_STIM_CHANNELS` to a 1-D `np.float32`
    array of length `n_frames`. The arrays are *not* z-scored — z-scoring is
    a downstream choice that must be made on the training-fold subset only
    (LOSO contract).
    """
    if video.ndim != 3:
        raise ValueError(f"expected 3-D video, got shape {video.shape}")
    n_frames, H, W = video.shape
    v = video.astype(np.float32) / 255.0  # → [0, 1]

    # ---- Per-frame luminance, std, normalised contrast --------------------
    pix = v.reshape(n_frames, -1)
    lum_mean = pix.mean(axis=1).astype(np.float32)
    lum_std  = pix.std(axis=1, ddof=0).astype(np.float32)
    norm_contrast = (lum_std / np.where(lum_mean > 1e-6, lum_mean, 1e-6)).astype(np.float32)

    # ---- Per-frame motion energy (first frame = 0) ------------------------
    motion = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        d = np.abs(np.diff(v, axis=0))                  # (n_frames-1, H, W)
        motion[1:] = d.reshape(n_frames - 1, -1).mean(axis=1).astype(np.float32)

    # ---- Per-frame spatial-gradient magnitude and edge fraction ----------
    grad_energy   = np.zeros(n_frames, dtype=np.float32)
    edge_fraction = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        gy, gx = np.gradient(v[t])
        gm = np.sqrt(gx * gx + gy * gy)
        grad_energy[t]  = float(gm.mean())
        thr = float(np.percentile(gm, 85.0))
        edge_fraction[t] = float((gm > thr).mean()) if np.isfinite(thr) else 0.0

    # ---- Per-frame centre-surround --------------------------------------
    h0, h1 = H // 4, (3 * H) // 4
    w0, w1 = W // 4, (3 * W) // 4
    center = v[:, h0:h1, w0:w1]
    center_n = (h1 - h0) * (w1 - w0)
    surround_n = max((H * W) - center_n, 1)
    center_mean   = center.reshape(n_frames, -1).mean(axis=1)
    surround_mean = (pix.sum(axis=1) - center.reshape(n_frames, -1).sum(axis=1)) / surround_n
    center_surround = (center_mean - surround_mean).astype(np.float32)

    return {
        "stim_lum_mean":        lum_mean,
        "stim_lum_std":         lum_std,
        "stim_norm_contrast":   norm_contrast,
        "stim_motion_energy":   motion,
        "stim_grad_energy":     grad_energy,
        "stim_edge_frac":       edge_fraction,
        "stim_center_surround": center_surround,
    }


def compute_per_frame_stim_map(
    reader,
    hashes: Iterable[str],
    pad_to: int,
    progress_every: int = 50,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-frame stimulus channels for every hash and pad to `pad_to`.

    Returns a dict keyed by `condition_hash` whose value is a dict of channel
    arrays of length `pad_to` (each channel zero-padded on the right to match
    the calcium `response_avg` padding convention used in
    `src/features/raw_traces.py:get_padded_arrays`).
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    hashes = list(hashes)
    for i, h in enumerate(hashes):
        if progress_every and i and i % progress_every == 0:
            print(f"  [per-frame stim] {i}/{len(hashes)} hashes processed …", flush=True)
        try:
            vid = reader.get_video_data(h)
            arr = np.asarray(vid["clip"])
            feats = compute_per_frame_video_features(arr)
        except Exception as e:
            print(
                f"  [per-frame stim] WARN: could not process hash={h!r}: "
                f"{type(e).__name__}: {e}"
            )
            continue
        padded: dict[str, np.ndarray] = {}
        for name, x in feats.items():
            T = len(x)
            if T >= pad_to:
                padded[name] = x[:pad_to].astype(np.float32)
            else:
                buf = np.zeros(pad_to, dtype=np.float32)
                buf[:T] = x
                padded[name] = buf
        out[h] = padded
    return out


def compute_tier_d(
    reader,
    hashes: Iterable[str],
    progress_every: int = 200,
) -> pd.DataFrame:
    """Compute Tier D descriptors for every hash in `hashes`.

    Parameters
    ----------
    reader : a `microns_datacleaner.MicronsFunctionalReader` instance.
    hashes : iterable of condition-hash strings.
    progress_every : how often to print a progress message.

    Returns
    -------
    DataFrame with columns:
      condition_hash, stim_type, n_frames, frame_height, frame_width,
      d_luminance_mean, d_luminance_std, d_contrast, d_motion_energy,
      d_sf_low, d_sf_mid, d_sf_high, d_tf_low, d_tf_mid, d_tf_high
    """
    rows = []
    hashes = list(hashes)
    for i, h in enumerate(hashes):
        if progress_every and i and i % progress_every == 0:
            print(f"  [tier_d] {i}/{len(hashes)} hashes processed …", flush=True)
        try:
            vid = reader.get_video_data(h)
            arr = np.asarray(vid["clip"])
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            stim_type = vid.get("stim_type", "?")
            if isinstance(stim_type, (bytes, bytearray)):
                stim_type = stim_type.decode()
            feats = _compute_video_features(arr)
            rows.append({
                "condition_hash":  h,
                "stim_type":       stim_type,
                "n_frames":        int(arr.shape[0]),
                "frame_height":    int(arr.shape[1]),
                "frame_width":     int(arr.shape[2]),
                **feats,
            })
        except Exception as e:
            print(f"  [tier_d] WARN: could not process hash={h!r}: {type(e).__name__}: {e}")
    return pd.DataFrame(rows)
