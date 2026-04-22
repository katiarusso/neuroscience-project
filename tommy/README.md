# Project Diary — Tommy

Working on the **MICrONS dataset**: predicting cortical layer identity of V1 excitatory neurons from functional neural activity (calcium imaging / fluorescence time series).

---

## Table of Contents
1. [Project Goal](#project-goal)
2. [Data](#data)
3. [Folder Structure](#folder-structure)
4. [EDA Observations](#eda-observations)
5. [Modelling](#modelling)
6. [Open Questions & Ideas](#open-questions--ideas)
7. [Log](#log)

---

## Project Goal

Build a classifier that predicts the **cortical layer** (L2/3, L4, L5, L6) of a V1 excitatory neuron from its functional activity time series, using the MICrONS multimodal connectomics dataset.

---

## Data

| File | Description |
|---|---|
| `data/functional/microns_functional.h5` | Functional recordings (fluorescence/dF/F traces, stimulus metadata, behavioural signals) |
| `microns_env/` | Python virtual environment with all dependencies |

**Dataset version used:** `1718`

**Filtering pipeline (structural):**
- Keep only `tuning_type == 'matched'` neurons (coregistered structural + functional)
- Keep only `cell_type == 'excitatory_neuron'`
- Keep only `brain_area == 'V1'`
- Drop rows with missing `layer`
- Drop non-neuronal objects and inhibitory neurons (underrepresented)

**Class imbalance (matched V1 excitatory neurons):**
- L2/3 and L5 dominate the population
- L1 is severely underrepresented (~14 neurons) — any L1 result is statistically unreliable
- L6/L6b also limited

---

## EDA Observations

### Structural data
- Matched V1 excitatory neurons are the working set; all other subpopulations discarded.
- `cc_abs` (Digital Twin model prediction quality): higher → neuron response more reliably captured. Useful as a data-quality filter.
- `strategy_axon` / `strategy_dendrite` and `status_axon` / `status_dendrite` encode reconstruction and proofreading quality — relevant for weighting or filtering.
- L2/3 is by far the largest class; L1 has too few samples to model reliably.

### Functional time series
- Individual neuron traces are **highly heterogeneous**: sparse large-amplitude transients, many neurons near-silent. No obvious structure at single-trial level.
- **Within-layer variability >> between-layer differences** in mean activity. Layer mean traces almost completely overlap; ±1 SD bands cover the full dynamic range.
- High-activity events are scattered randomly — no layer-specific temporal motif visible in single-trial heatmaps.
- Trial-averaging reveals a **weak stimulus-onset transient in the first ~0.5 s**, slightly more pronounced in L1 and L5, but error bands remain heavily overlapping across all layers.
- **Peak activity distributions are nearly identical across layers** — peak amplitude is not a discriminative feature.
- Pairwise correlation matrix is uniformly low-positive (~0.1–0.2) with **no block structure along layer boundaries** → neurons in the same layer are not more correlated with each other than with neurons from other layers.
- PCA / t-SNE / UMAP projections show **no separable layer clusters** from trial-averaged time series alone.

### Experiments / sessions
- Multiple sessions and scans; stimulus types include **Clip, Monet2, Trippy**.
- Treadmill, pupil, and eye-position signals are available and modulate V1 gain — important covariates.
- Not all sessions contain all layers — use the layer-availability table (Section 11 of `functional_timeseries_EDA.ipynb`) to select sessions before building classifiers.

---

## Modelling

### Naive baseline — raw time series features (`naive classification approach.ipynb`)

**Setup:**
- Input: raw (z-scored) time series for matched V1 excitatory neurons across all usable session/scan pairs.
- A neuron can contribute multiple samples (one per trial/session) — **same neuron may appear in both train and test splits**, making the task slightly easier than realistic deployment.
- Labels: cortical layer.

**Models tried:** logistic regression, random forest, MLP (PyTorch).

**MLP architecture:** `input_dim → 512 → 256 → 128 → num_classes` with BatchNorm + Dropout, class-weighted cross-entropy.

**Metrics reported:** accuracy, balanced accuracy, macro-F1, confusion matrix.

**Result / status:** baseline established. Numbers TBD — fill in after runs.

---

## Open Questions & Ideas

- [ ] **Neuron-level train/test split**: prevent the same neuron from appearing in both splits — more realistic evaluation.
- [ ] **Feature engineering**: instead of raw traces, try summary statistics (mean, std, peak, AUC, rise time) or frequency-domain features.
- [ ] **Stimulus-conditioned analysis**: separate sessions by stimulus type (Clip vs. Monet2 vs. Trippy) and check whether discriminability changes.
- [ ] **Behavioural covariates**: regress out treadmill / pupil signals before classification to isolate stimulus-driven variance.
- [ ] **cc_abs filtering**: restrict to high-quality neurons (e.g. cc_abs > threshold) and check if layer signal improves.
- [ ] **Connectivity features**: incorporate synaptic connectivity (structural graph) as additional input — L-specific connectivity patterns may be more discriminative than activity alone.
- [ ] **Self-supervised pre-training**: pre-train a time-series encoder on all neurons before fine-tuning on the layer classification task.
- [ ] **Multi-task learning**: jointly predict layer + cell_type + brain_area.
- [ ] **Address class imbalance more aggressively**: oversample L1/L6, or exclude them and reframe as L2/3 vs L4 vs L5 3-class problem.

---

## Log

| Date | Entry |
|---|---|
| 2026-04-22 | Diary created. Structural + functional EDA complete. Naive baseline classifier built (MLP + logistic regression + RF on raw z-scored traces). Key finding: raw time series carry very little layer-discriminative signal — within-layer variability dominates. |

---

*Add new entries to the Log table and update the observations/ideas sections as the project evolves.*
