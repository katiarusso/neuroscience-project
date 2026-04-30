# Guess the Neuron — MICrONS V1 Functional-Only Cell-Type Classification

**Working title.** *Guess the neuron from function:* predicting cortical layer, and then excitatory subtype within deep layers, from MICrONS functional data only.

---

## 1. Project framing

### 1.1 Official task
> Using functional data only, build a classifier for excitatory neuron type. Can one see different layers? Is it possible to differentiate different excitatory types in layers 5 and 6?

### 1.2 Scope (group decision)
We restrict the analysis to:

- **V1 only** — V1 is the densest area in the dataset and the most constrained biologically, which makes it the cleanest substrate for asking a function-vs-identity question.
- **Excitatory neurons only** — inhibitory and non-neuronal classes are out of scope by the prompt.
- **Matched neurons only** — we keep only neurons with reliable functional-to-structural coregistration. Without coregistration the label cannot be trusted.

### 1.3 Two-target strategy
The project proceeds in two phases that must not be conflated:

- **Phase 1 — Layer decoding.** Layer is the broadest and most biologically robust label: it is tied to circuit position, input/output role, and dendritic morphology. If function carries identity information at all, it should appear here first.
- **Phase 2 — Within-layer subtype decoding (L5/L6).** Only attempted after Phase 1 has stabilized. The rationale is empirical: dendritic-morphology work (Weis et al., 2025) shows excitatory neurons form a continuum with sharper clusters in **deep layers**, especially L5 and to a lesser extent L6. So the most realistic "subtype" target is L5 IT vs ET vs NP, and L6 IT vs CT, *within* layer.

### 1.4 Scientific question (re-statement)
> **How much of excitatory neuronal identity is recoverable from function alone, and at which level of granularity does that recovery remain reliable?**

This framing is stronger than "let us train a classifier" because it makes a partial or negative result still informative: if layer decodes but subtype does not, we are saying that within-layer excitatory variation is more morphological than functional at this measurement resolution.

### 1.5 Why this question is non-trivial
There is no a-priori guarantee that the calcium response to videos contains layer-discriminative information. V1 excitatory neurons across layers share a lot of basic visual selectivity (orientation, direction, spatial/temporal frequency), and most layer-discriminative properties (RF size, feedback influence, behavioral modulation, reliability) are subtle and confounded by recording quality and stimulus exposure. The project is therefore as much a **representation problem** (what to compute from the trace) as a **modeling problem** (which classifier to fit).

---

## 2. Dataset at a glance

### 2.1 Structural side (via `microns-datacleaner`, version 1718)
- ~90k anatomical nuclei in the canonical units table.
- ~42k V1 excitatory neurons before the matching filter.
- **~8.9k V1 excitatory neurons remain after `functional_data='best_only'` matching** — this is our working population.
- Class breakdown (AIBS v661 vocabulary, V1 excitatory, matched, best-only):

  | Layer | Type      | Approx. count |
  |-------|-----------|---------------|
  | L2/3  | 23P       | ~4.2k |
  | L4    | 4P        | ~2.7k |
  | L5    | 5P-IT     | ~1.2k |
  | L5    | 5P-ET     | ~320 |
  | L5    | 5P-NP     | ~26 |
  | L6    | 6P-IT     | ~216 |
  | L6    | 6P-CT     | ~121 |

  Class imbalance is severe: L4 and L2/3 dominate, 5P-NP and L6 subtypes are small. This drives every choice on metrics, weighting, and Phase-2 scoping.

- **Quality flag:** `cc_abs` (correlation between recorded response and a digital-twin model). Median ≈ 0.4 but **systematically lower for 6P-CT/IT (~0.27)**. This means recording quality is *correlated with the label*, which is a confound we must control for explicitly (Section 6 and 9).

### 2.2 Functional side (`microns_functional.h5`, ~20 GB)
- **14 session–scan blocks.** Each block contains its own pool of recorded units.
- **464 trials per block.** A trial = one stimulus presentation, with neural, pupil, treadmill, and stimulus video time series stored together.
- **Sampling rate ≈ 7.5 Hz** (GCaMP6s calcium imaging). Trial length depends on stimulus type:
  - **Clip** (natural movies): 75 frames ≈ 10 s.
  - **Monet2** (drifting orientation noise): 113 frames ≈ 15 s.
  - **Trippy** (phase-shuffled noise): 113 frames ≈ 15 s.
- **Stimulus library:** 2,411 unique condition hashes (≈2,112 Clip, 150 Monet2, 149 Trippy). Each hash identifies one specific video clip.
- **Oracle hashes:** ~116 hashes (96 Clip + 10 Monet2 + 10 Trippy) are repeated across **all 14 sessions**. These are the *only* stimuli that allow direct cross-session comparisons of the *same neuron-vs-stimulus pair* — they are the backbone of any reliability-based feature.
- **Per-trial channels:**
  - Calcium response: shape `(n_units_in_session, n_frames)`, one trial.
  - Pupil: 4D (position x, position y, dilation, plus a confidence/secondary).
  - Treadmill speed: scalar over time.
  - Stimulus video: pre-aligned to the calcium frame rate, shape `(n_frames, H, W)`, uint8 — `144×256` for Clip, `126×216` for Monet2, `90×160` for Trippy.

### 2.3 Identity convention across scans
A given neuron may appear in multiple session–scan blocks. The cross-scan identifier is `nucleus_id` (from the structural side); `unit_id` is local to the block and **must not** be used as a cross-session key.

### 2.4 What "best-only" means
For each `nucleus_id` with at least one valid functional match, `best-only` keeps a single (session, scan, unit_id) — the highest-quality match. This avoids repeated-measure dependence in Phase 1 at the cost of discarding extra recordings. We treat best-only as the **primary** regime and pooled-scan as a **sensitivity analysis** (Section 6.4).

---

## 3. Methodological cornerstones

These four decisions must be settled before any modeling. They shape the entire pipeline and each carries non-trivial trade-offs.

### 3.1 Best-only vs pooled scans
- **ML reason for best-only.** Removes within-neuron repeated measures, simplifies grouping, eliminates the highest-quality bias driving any neuron's representation.
- **ML reason for pooled.** More data per neuron means more reliable feature estimates, especially for noisy summaries (latency, reliability).
- **Neuroscience reason.** A neuron's response distribution is genuinely scan-dependent (anesthesia state, FOV, drift), so pooling is not "free more data" but introduces session-as-confound.
- **Decision.** Phase 1 reports use **best-only**. Pooled-scan is a robustness check: features re-computed by averaging across scans of the same `nucleus_id`, with grouped CV by `nucleus_id`. We compare deltas, not absolutes.

### 3.2 Unit of analysis (what is a row?)
The single biggest design choice. Four options, ordered from most aggressive averaging to most preserved structure.

| Option | Row | Pros | Cons |
|--------|-----|------|------|
| A. One row per neuron, all stimuli pooled | 1 vector / neuron | Simplest classical ML setup | Destroys stimulus-specific tuning, which is exactly where layer differences are expected |
| B. One row per (neuron × stimulus family) | ≤3 rows / neuron | Preserves Clip vs Monet2 vs Trippy contrast | Within-family variation is collapsed |
| C. One row per (neuron × hash), repeated trials averaged | ~tens of rows / neuron | Preserves per-stimulus tuning, supports reliability features | Repeated rows per neuron require grouped CV; uneven sampling per neuron |
| D. One row per (neuron × hash × trial) | hundreds of rows / neuron | No information lost; supports trial-by-trial behavioral analysis | Heavy repeated-measure dependency; inflates dataset size |

**Decision.** The natural unit is **C (neuron × hash, trial-averaged)**. From C we build:
- A **wide** matrix (Option A) by concatenating per-hash features for a fixed set of hashes (e.g. the 116 oracle hashes) — one row per neuron, suitable for classical models.
- A **family-specific** view (Option B) by grouping C rows on stimulus family — useful for asking which family is most informative.
- An **aggregated** view (Option A, learned) by training a model that consumes set-of-hash features and pools them inside the model (e.g. attention over hashes).

### 3.3 To trial-average or not? — the signal/noise dichotomy
This is often discussed as binary ("average or don't") but the better framing is that **averaging produces signal features and the residuals produce noise features, and both are biologically informative**.

- **Signal features** (computed on trial-averaged trace per (neuron, hash)): mean amplitude, peak, latency, time-to-peak, response shape. These approximate the deterministic stimulus-driven response.
- **Noise / variability features** (computed across the trial set per (neuron, hash) before averaging): trial-to-trial variance, Fano factor, trial-trial Pearson correlation (a.k.a. *oracle correlation* on oracle hashes), coefficient of variation, signal-to-noise ratio.

The neuroscience rationale is direct: cell types differ in **reliability**, not just amplitude. L4 cells receive the most direct LGN input and are typically the most trial-reliable; L5 ET cells are noisier and more state-modulated; L6 cells are reliability-low and tuning-narrow. Throwing away the variance is throwing away one of the few features known a-priori to differ between layers.

**Decision.** For every (neuron, hash) cell, compute both a signal vector and a variability scalar set. They are two feature families used jointly downstream.

### 3.4 The role of behavior when behavior is shared across neurons
**Question (raised explicitly in the brief):** if pupil dilation and running speed are the same for every neuron at the same instant, how can they discriminate between neurons?

The shared variable is not the feature. Per-neuron features are the **interaction** between behavior and the neuron's response. Concretely, for a behavioral signal `b(t)` (running speed, pupil) and neural trace `r_i(t)` of neuron *i*, we compute per-neuron statistics such as:

- **Behavioral modulation index:** `(mean response when running) − (mean response when still)` divided by their sum, computed **within hash** so that stimulus is held fixed.
- **Pupil-response coupling:** regression slope of trial-averaged response onto trial-averaged pupil size, again within hash to remove stimulus confound.
- **State-conditional reliability:** trial-trial correlation computed separately on running vs still trials.

These are scalars per neuron. The "shared" behavioral channel becomes informative the moment it is used as a *covariate over which the neural response is conditioned*. This is also the reason behavior should not be added as a raw input dimension to a single-feature-vector model expecting per-neuron features — it would carry no signal there. Where behavior *is* used as a raw channel is in **multi-channel time-series models** (Section 5, Stage 4–5) where convolution can learn per-neuron interactions implicitly.

Biological grounding (Niell & Stryker 2010; Dadarlat & Stryker 2017): locomotion gates V1 responses, and the gain of that gating differs across layers — L2/3 and L5 ET show strong running-up modulation, L4 and L6 less. This is one of the few well-documented per-cell-type functional differences in mouse V1 and we expect it to be one of the more discriminative features.

---

## 4. Feature representations (the tiers)

### 4.0 The framing in plain terms
Before any classifier can see a neuron, that neuron has to be turned into a **fixed-size vector of numbers**. There are many ways to do that compression, and each way is implicitly a guess about what makes neurons of different layers look different. The seven tiers below are seven different guesses, ordered roughly from easiest-to-build to most engineering-heavy. They are *not alternatives* — we add them up and compare which ones actually move the score.

A one-line plain-language description of each tier, before the detailed cards:

- **Tier A — How big does this neuron respond?** Average response, peak, integral, baseline. One scalar set per neuron.
- **Tier B — How consistent is it?** When the same video is shown twice, does the neuron respond the same way both times? A reliability score per neuron.
- **Tier C — What does it like?** A per-neuron tuning curve across stimuli — one response value per video.
- **Tier D — How does behavior change its response?** When the mouse runs vs sits still, how much does the neuron's response change? A *delta* per neuron, even though running itself is shared.
- **Tier E — How is the response shaped over time?** Not "how big" but "how does it rise, peak, and decay" — the full temporal trace.
- **Tier F — Where in space, what pattern?** A receptive-field map per neuron, optionally corrected by where the mouse is looking.
- **Tier G — Cross-stimulus fingerprint.** The vector of responses to the 116 oracle stimuli, used as the neuron's barcode.

The plan is: build A, B, C, run a baseline. Add D. See if anything decodes layer above chance and above the confound baselines (Section 6). Then keep adding tiers one at a time and measure which one *adds the most*. The output of the project is partly an accuracy number, but mostly the **ablation table** that says which tiers carry the biological signal.

For each tier below we record: the biological prior, the ML rationale, what exactly to compute, and the failure mode that would make the tier uninformative.

### Tier A — Stimulus-agnostic response statistics
- **Prior.** Different layers have different overall response amplitudes and baseline regimes (L2/3 is sparser, L4 is more reliable, L6 is dimmer).
- **What.** Per neuron over all valid trials: mean response, peak, baseline, integral, post-stimulus decay, cross-trial mean amplitude.
- **ML role.** Cheapest baseline; collapsing all stimulus information.
- **Failure mode.** Indistinguishable amplitude profile — likely if recording quality dominates over biology.

### Tier B — Reliability and trial-to-trial structure
- **Prior.** Cell-type-specific noise structure (L4 reliable, L5 ET noisy, L6 low-SNR).
- **What.** Within-hash trial-trial correlation, Fano factor, oracle correlation (using the 116 oracle hashes), CV across trials, signal-to-noise.
- **ML role.** Per-neuron scalars complementary to amplitude; key feature predicted by Schneider et al. (2023).
- **Failure mode.** Reliability dominated by `cc_abs` quality — controlled by including `cc_abs` as a covariate baseline (Section 6.5).

### Tier C — Stimulus-specific tuning
- **Prior.** Layer differences are most evident under specific stimulus regimes — gratings (Monet2) for orientation tuning, natural movies (Clip) for high-level selectivity, phase noise (Trippy) for low-level features.
- **What.**
  - Per (neuron × hash) signal features (Section 3.3).
  - Per (neuron × family) summaries: family preference index, dispersion of responses across hashes within family, entropy/selectivity across hashes (sparseness measure à la Treves–Rolls).
  - Family-vs-family contrasts (e.g. natural minus synthetic preference).
- **ML role.** Wide-format inputs; the per-hash version is the main candidate to beat the Tier A baseline; the per-family version is the lighter compressed alternative.
- **Failure mode.** Per-hash features dominated by "did this neuron respond at all" — handled by including reliability features and stimulus-presence baselines.

### Tier D — Behavioral modulation (per-neuron interactions)
- **Prior.** Cell-type-specific modulation by locomotion and pupil (Section 3.4).
- **What.** Running modulation index (within hash, then averaged), pupil–response coupling slope (within hash), state-conditional reliability, response gain difference between high and low arousal.
- **ML role.** Adds a second axis of identity beyond pure visual tuning.
- **Failure mode.** Behavioral distribution differs across sessions, so a *neuron* may look highly modulated only because its session had more variance; controlled by computing modulation indices on *matched* high/low-state subsets per session.

### Tier E — Temporal response shape
- **Prior.** Cell types differ in onset latency, rise time, adaptation (Schneider et al., 2023).
- **What.** Full trial-averaged calcium trace per (neuron, hash). Optional basis projection (PCA, B-splines) to reduce dimensionality before linear models. Per-trace scalars: latency-to-half-peak, full-width-half-maximum, monotonic-decay coefficient, early-vs-late ratio.
- **ML role.** First time we use the *shape* of the response, not its scalar summaries. If linear models on flattened or projected traces beat scalar baselines, the temporal axis is informative.
- **Failure mode.** Calcium dynamics are dominated by GCaMP6s decay (~500 ms), so very fast cell-type differences can be invisible at this temporal resolution. We acknowledge this as a hard ceiling.

### Tier F — Receptive-field-derived features (the gaze-aware aggregate)
This tier directly addresses the brief's question about aggregate variables that depend on the spatial structure of the stimulus.

- **Prior.** V1 is retinotopic; layer differs in RF size and complexity (L4 small simple, L2/3 larger more complex, L5 broad and high-gain, L6 narrow and CT-specific). RF properties are among the cleanest layer-discriminative variables in the V1 literature.
- **What.**
  1. **Stimulus-driven RF estimate.** Per neuron, regress the calcium response onto the stimulus pixel activity (spike-triggered-like estimate / linear–nonlinear fit), giving a `(τ, H, W)` spatiotemporal kernel.
  2. **Gaze-aware RF estimate.** Mice move their eyes even when head-fixed. The pupil position channel gives an estimate of gaze, so the *retinal* image is the screen image shifted by gaze. We re-estimate the RF on the **gaze-shifted stimulus**: this disentangles stimulus-on-screen from stimulus-on-retina and gives a cleaner per-neuron RF.
  3. **RF descriptors.** From the kernel: 2D Gaussian fit (σx, σy → RF size, aspect ratio), preferred orientation/direction, spatial-frequency peak (via 2D FFT), simple/complex index (parity of phase), temporal lag of the peak.
  4. **Stimulus-locked response (the "look only at the area the mouse is looking at" idea).** Once we have a per-neuron RF, we can restrict the response analysis to the time windows where high-contrast stimulus content falls *inside* that neuron's RF (after gaze correction). This gives a fair, RF-aligned response feature: each neuron is judged on its own preferred input rather than on a stimulus-blind average.
- **ML role.** A small set of physically meaningful, layer-discriminative scalars that should be more interpretable than any black-box embedding. Strong candidate to be the most informative tier in Phase 2 (within-layer subtype).
- **Failure mode.** Calcium temporal blur limits the spike-time-precision RF; gaze tracking is imperfect; some neurons have RFs outside the stimulus monitor extent (filtered out in EDA).
- **Cost.** This tier is the most engineering-heavy. It is reasonable to defer it to after Tiers A–E and use it as the "advanced feature" of the project.

### Tier G — Cross-stimulus / oracle-hash consistency
- **Prior.** A neuron's *signature* is more reliably identified by *which stimuli it prefers relative to others* than by absolute response amplitude.
- **What.** Per neuron, the vector of trial-averaged responses to the 116 oracle hashes — a "stimulus fingerprint". Then derive: cosine similarity matrices, response-rank profiles, low-rank embedding of the fingerprint matrix (PCA across neurons → loadings), and, if used as input, the fingerprint vector itself.
- **ML role.** This is the strongest *cross-session* representation we have because oracle hashes are the same in every session; comparing fingerprints across sessions is fair.
- **Failure mode.** 116 hashes is small relative to neuron count; overfitting risk on the wide 116-dim feature without regularization.

### What is unlikely to be sufficient on its own
- A single global average per neuron across all stimuli (Tier A only).
- Pooling all trials without preserving stimulus identity (kills Tier C and Tier B).
- Behavior added as a raw input dimension to a per-neuron feature vector (kills Tier D, Section 3.4).
- A CNN trained directly on raw traces before scalar baselines have established what is decodable (no interpretation foothold).

---

## 5. Modeling roadmap

The roadmap is built so that each step adds at most one new modeling assumption on top of a known baseline, and each step answers one specific question.

### Stage 1 — Linear baselines on engineered features
- **Inputs.** Tier A + Tier B + Tier C (signal features per hash), in the wide format described in Section 3.2.
- **Models.** Multinomial logistic regression with L1/L2 regularization, ridge classifier, linear SVM, LDA.
- **Question answered.** Is layer information already linearly accessible from response statistics? If yes, the rest of the project is about pushing that signal further; if no, the rest is about diagnosing whether the bottleneck is representation or model.
- **Decision rule.** If above-chance after grouped CV with calibrated baselines, escalate. If at chance, stop and run Tier D + Tier B variability features before going deeper.

### Stage 2 — Add behavioral interactions (Tier D)
- **Inputs.** Stage 1 features + Tier D modulation indices.
- **Question answered.** Does behavioral modulation add information, after stimulus tuning is already in the features?
- **Ablation.** Run with-Tier-D vs without-Tier-D, both with grouped CV. Report delta in balanced accuracy and in per-class recall (especially L5 ET, where we predict the largest gain).

### Stage 3 — Nonlinear classifiers on the same features
- **Inputs.** Same as Stage 2.
- **Models.** Random forest, gradient boosted trees (XGBoost/LightGBM), shallow MLP, RBF-SVM.
- **Question answered.** Are there interactions between the engineered features that linear models miss?
- **Important interpretation rule.** A null gain over linear *does not* prove the problem is "linear" — it could mean the features are too compressed for nonlinear structure to matter. The diagnostic for that is Stage 4.

### Stage 4 — Temporal models on Tier E
- **Inputs.** Trial-averaged trace per (neuron, hash) — first as flattened vector, then as projected basis (PCA / B-splines), then as 1D-CNN input.
- **Models.** Linear + ridge on flattened traces (sanity check); 1D CNN on per-hash traces; per-neuron set-attention over hash-specific traces.
- **Question answered.** Does the *shape* of the response carry information that scalar summaries lost?
- **Caution.** With ~9k neurons and severe class imbalance, deep models are overfit-prone. Use grouped CV by `nucleus_id`, early stopping, and aggressive regularization.

### Stage 5 — Multi-modal CNN
- **Inputs.** Per-trial calcium trace + pupil + treadmill + (optionally) downsampled stimulus video.
- **Models.** Multi-branch 1D-CNN: one branch per channel, then a pooling head per neuron. The multi-channel formulation is what allows behavior to be useful at the trial level (Section 3.4) — convolutions can learn per-neuron interactions implicitly.
- **Question answered.** Does the fully multimodal trial-level representation beat the engineered-feature pipelines?
- **Decision rule.** Only run if Stage 4 shows a meaningful temporal gain. Otherwise this is engineering for engineering's sake.

### Stage 6 — Receptive-field-aware features (Tier F)
- **Inputs.** Tier F descriptors as scalar features added to the Stage 1–3 pipeline; or RF-aligned response features replacing Tier A.
- **Question answered.** Do physically interpretable RF descriptors carry more layer/subtype information than statistical summaries? Particularly important for Phase 2 within-layer subtype decoding, where we expect RF size and complexity to matter.

### Stage 7 — Phase 2: within-layer subtype
Run Stages 1–4 of the same pipeline restricted to L5 (5P-IT, 5P-ET, 5P-NP) and to L6 (6P-IT, 6P-CT). Class counts force this to be a separate analysis with its own metrics and resampling regime (Section 7).

---

## 6. Evaluation, splitting, and confound control

### 6.1 Golden splitting rule
**Split by `nucleus_id`, never by row.** Any feature row generated from the same neuron must end up in the same fold. This is enforced via `GroupKFold` with `nucleus_id` as the group, even in best-only (where it is also a sensible defensive default).

**Note on splitting by scan.** We do *not* split by scan in the primary protocol. Two different neurons recorded in the same scan are allowed to fall on opposite sides of the train/test split. The reason is that every scan contains multiple cell types: blocking by scan would mean the model never learns the *within-scan* distinctions we actually care about. Splitting by neuron (and only by neuron) is the right granularity — what we never allow is the *same* neuron to appear on both sides. Whether the scan itself is a confound is a separate question, addressed below.

### 6.1bis The scan composition confound — what it is and how we handle it
The 14 session–scan blocks were not recorded with a uniform mix of cell types. Each scan placed its imaging field of view at a particular location and depth, under particular conditions (laser power, drift, the day's behavioral state distribution, etc.), and ended up with whichever neurons fell in that FOV. As a result, **"which scan a neuron is from" is itself correlated with the layer label**, even before any biology comes in.

Why this matters for us: a model that uses neural features can end up predicting layer through session-specific recording artefacts (a particular noise floor, a particular photobleaching profile, a particular pupil-size distribution that day) rather than through genuine biology. The features look neural but they are partly session-fingerprints.

The handling is layered:

- **Session-only baseline (Section 6.5).** A classifier whose only input is the session/scan ID one-hot. If that already beats chance on layer, then session is informative about layer and any "real" model has to beat that bar by a meaningful margin to claim it is using biology.
- **Per-session standardization.** Z-score every feature within its session before pooling across sessions, so that session-level offsets and scales (one session being globally brighter, or having a wider behavioral distribution) cannot be read by the classifier as a layer signal.
- **Session as covariate.** In linear models, include session-fixed-effect dummies. The model can absorb session-level offsets explicitly rather than confusing them with biology.
- **Leave-one-session-out as a sanity check.** Beyond the standard `GroupKFold` by `nucleus_id`, we additionally run a leave-one-session-out CV: each fold holds out an entire session, trains on the others, predicts on the held-out session. If standard grouped-CV scores 47% balanced accuracy but leave-one-session-out drops to 30%, the model was leaning on session-specific signal. If the two are close, the model has learned something that generalizes across recording conditions. We report both.

### 6.2 Stratification
Stratify folds by layer label as much as group constraints allow (e.g. `StratifiedGroupKFold`). In Phase 2 (subtype), stratify by subtype within layer.

### 6.3 Metrics
- **Primary:** balanced accuracy and macro F1. With this class imbalance, plain accuracy is misleading.
- **Secondary:** confusion matrix, per-class recall (especially the minority classes), one-vs-rest AUC for subtype experiments.
- **Calibration check:** majority-class baseline, stratified-random baseline.

### 6.4 Robustness checks (run on the best Stage)
- Best-only vs pooled-scan.
- Per-stimulus-family vs all-families.
- With and without `cc_abs` quality covariate.
- With and without per-session normalization.
- Removing the smallest classes (5P-NP) to test whether reported gains come from rare-class fluctuation.

### 6.5 Confound baselines (this is what makes the project defensible)
Always report:

- **Quality-only baseline.** Train a classifier that uses only `cc_abs` (and any other recording-quality scalar). If our model does not beat this baseline, we are decoding quality, not biology.
- **Scan-membership baseline.** Train a classifier that uses only the session/scan ID one-hot. If beats chance, layer composition differs across sessions and we must control for it.
- **Cortical-depth baseline.** If `nucleus_y` (cortical depth) is functional-data-adjacent (it is structural, but trivially predictable from any neural feature that scales with depth), report a baseline that uses *only* depth. The functional model must beat depth-only by a meaningful margin.

### 6.6 Reporting format
For every experiment: balanced accuracy ± std over folds, confusion matrix, comparison to baselines, and an ablation table showing the marginal contribution of each tier added.

---

## 7. Phase 2 — within-layer subtype decoding

Phase 2 starts only after Phase 1 has produced a stable layer decoder.

### 7.1 Sample sizes (best-only)
- L5: 5P-IT ≈ 1.2k, 5P-ET ≈ 320, 5P-NP ≈ 26 → drop 5P-NP from main analysis, treat 5P-IT vs 5P-ET as the headline binary task.
- L6: 6P-IT ≈ 216, 6P-CT ≈ 121 → small-sample regime; report with bootstrap CIs and prefer simple models.

### 7.2 Predicted markers per subtype
- **5P-ET vs 5P-IT.** ET cells project subcortically and are expected to have stronger running modulation, broader RFs, and higher gain. Tier D (running modulation) and Tier F (RF size) are the predicted dominant features.
- **6P-CT vs 6P-IT.** CT cells have narrow tuning and quieter baselines; they are also lower in `cc_abs`. Tier B (reliability) and Tier C (selectivity / sparseness) are the predicted dominant features. Confound control on `cc_abs` is critical here precisely because the quality bias is largest.

### 7.3 Models
Linear baselines first (logistic regression with L2). Tree models (gradient boosting) second. Deep models are not justified for these sample sizes.

### 7.4 Metrics
Balanced accuracy and per-class recall, with bootstrap CIs over neurons. Report calibration carefully — a +5% balanced accuracy on N=337 is within fluctuation range and must be flagged.

---

## 8. Interpretability and ablations

Interpretability serves two purposes: validating that the model is using biology rather than confound, and turning a numerical result into a scientific claim.

### 8.1 Tier ablation table
The single most informative output of the project. For each tier (A–G), report the layer balanced accuracy when that tier is *removed* from the best model. The tier whose removal hurts most is the one that carries the layer information.

### 8.2 Per-model interpretability
- **Linear models.** Standardized coefficients, per-class coefficient inspection, sparsity (L1) for feature selection.
- **Tree models.** Permutation importance and SHAP. SHAP is used selectively (it is expensive and can dominate the project budget).
- **Temporal/CNN models.** First-layer filter visualization, temporal occlusion ablations (zero out windows, measure performance drop), channel-ablation (drop pupil channel, drop stimulus channel).

### 8.3 Scientific output to aim for
We want the discussion to read as:
> "Layer information appears to be carried mainly by [stimulus-specific reliability and behavioral modulation], with [temporal shape] adding a small but consistent margin. RF-derived features dominate the L5-IT vs L5-ET separation."

Not:
> "The CNN got 47% balanced accuracy."

---

## 9. Risks and confounds (consolidated)

| Risk | Mechanism | Control |
|------|-----------|---------|
| Repeated-measure leakage | Same neuron across train/test | `GroupKFold` by `nucleus_id` always |
| Quality–label confound | `cc_abs` lower for 6P → quality decodes label | `cc_abs`-only baseline; per-class quality reporting |
| Scan composition confound | Some layers concentrate in some sessions | Scan-only baseline; per-session normalization |
| Class imbalance | Minority classes inflate variance | Balanced metrics, class weights, drop smallest class for Phase 2 |
| Over-compression of features | Single-vector-per-neuron loses tuning | Tier C as default; Tiers E and F if needed |
| Behavior treated as raw-shared input | Per-neuron-vector framing kills behavior signal | Use Tier D (modulation indices), or multi-channel models |
| Temporal blur from GCaMP6s | <500 ms cell-type differences are invisible | Acknowledge as ceiling; lean on amplitude/reliability/RF rather than fast latency |
| Overfitting deep models | ~9k neurons, severe imbalance | Group-CV, early stopping, only use deep models when justified by simpler ones |
| Biological overclaiming | "Decodes layer" ≠ "function determines layer" | Report decoding as statistical association, not causal coding |

---

## 10. Possible extensions

These are out-of-scope for the main project but useful in discussion or as stretch goals.

- **Foundation-model embeddings (Tier H).** The Wang et al. (2025) MICrONS digital twin is a network trained to *predict* a neuron's calcium response from the stimulus and behavior — superficially a different task from cell-type classification. It is still useful here, and the reason is structural rather than coincidental. The model is built as a **shared visual backbone + a per-neuron readout head**. The backbone is one big network, the same for every neuron, that turns stimulus and behavior into a high-dimensional feature vector at every time point. The readout head is the part that is *specific to one neuron* — typically a small set of weights describing where in the visual field the neuron looks (spatial readout) and which directions in the backbone's feature space it cares about (feature readout). That per-neuron readout vector is the "embedding" stored in `readout_info/foundation_model.pkl`.

  By construction, two neurons with similar readout vectors respond similarly to the same stimuli. The vector is therefore a compact description of the neuron's functional preferences — receptive field location, preferred features, gain, behavioral modulation — packed into a single ~1024-D vector. This is *exactly* the thing we are trying to engineer by hand in Tiers C, D, E and F, except that the backbone has integrated information across thousands of stimuli using a learned representation optimized end-to-end on the entire MICrONS corpus. The Wang paper itself shows that these vectors recover cell-type structure, so it is not a speculative use.

  How we would use Tier H: (i) as an **upper-bound benchmark** — the gap between our hand-crafted tiers and the foundation embedding tells us how much functional information our pipeline is leaving on the table; (ii) as a **complementary feature block** concatenated with our engineered tiers, to test whether they add independent information. The trade-off is interpretability: a coefficient on "running modulation index" tells a biological story; a coefficient on dimension 743 of the embedding does not. Tier H is therefore a benchmark and complement, not a replacement for the engineered core.
- **Self-supervised pretraining on raw traces.** Train a Transformer/CNN encoder on the unlabeled response set with masked-trace reconstruction; fine-tune for layer.
- **Cross-area generalization.** Train on V1, evaluate on RL/AL/LM (out of scope by group decision, but a strong robustness check).
- **Within-neuron stability tests.** For neurons that appear in multiple scans (the pooled-scan regime), test how stable our predictions are across scans of the same `nucleus_id`. A model whose predictions flip across scans is a model decoding scan more than identity.
- **Connectivity-based sanity check.** MICrONS has synaptic connectivity. We could check whether neurons with similar predicted-class probability are also more connected (Ding et al., 2025), which would be a structure-side validation we are not allowed to *use* but can *compare against*.

---

## 11. Report narrative (2-page format)

The report mirrors the methodology, not the chronology of the work.

- **Introduction.** MICrONS lets us link cell identity to function. Open question: how much identity is recoverable from function alone? Restrict to V1 matched excitatory.
- **Question.** Phase 1 — layer; Phase 2 — within-layer subtype.
- **Method.** Best-only regime, neuron×hash unit of analysis, tiered feature engineering, grouped CV, confound baselines.
- **Results.** Layer decoding result, ablation table by tier, behavioral-modulation contribution, temporal-model gain, Phase 2 result with bootstrap CIs.
- **Discussion.** What carries the signal, what does not, where the calcium temporal blur and class imbalance limit us, what a negative subtype result would mean (function continuous, identity discrete in morphology only).

---

## 12. Literature anchors

We use a small, focused set of references rather than an exhaustive list.

- **MICrONS dataset and function–connectivity.** Fahey et al., *Functional connectomics spanning multiple areas of mouse visual cortex*, Nature 2025. Ding et al., *Functional connectomics reveals general wiring rule in mouse visual cortex*, Nature 2025. — Frame the dataset and motivate function-vs-identity.
- **Cell identity organization.** Gouwens et al., *Classification of electrophysiological and morphological neuron types in the mouse visual cortex*, Nat. Neurosci. 2019. Weis et al., *An unsupervised map of excitatory neuron dendritic morphology in the mouse visual cortex*, Nat. Comm. 2025. — Motivate Phase 2 in deep layers specifically.
- **Function carries identity information.** Schneider et al., *Transcriptomic cell type structures in vivo neuronal activity across multiple time scales*, Cell Reports 2023. — Motivates Tier B and Tier E.
- **Behavioral state in V1.** Niell & Stryker, *Modulation of visual responses by behavioral state in mouse visual cortex*, Neuron 2010. Dadarlat & Stryker, *Locomotion enhances neural encoding of visual stimuli in mouse V1*, J. Neurosci. 2017. — Motivate Tier D.

---

## 13. Final formulation

> We ask whether the anatomical identity of excitatory V1 neurons in the MICrONS dataset is recoverable from function alone. We frame the problem in two phases — coarse layer first, fine subtype within deep layers second — and we treat representation choice as the primary scientific variable, comparing a tiered set of features (response statistics, reliability, stimulus-specific tuning, per-neuron behavioral modulation, temporal response shape, receptive-field descriptors, and oracle-stimulus fingerprints) under matched grouped cross-validation and explicit confound baselines. The contribution of the work is not a single accuracy number, but the ablation map: an account of which functional features carry layer and subtype identity, and which do not.
