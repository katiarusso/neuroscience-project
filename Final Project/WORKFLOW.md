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

### 1.3 Phased target strategy
The project proceeds in three phases that must not be conflated:

- **Phase 1 — Layer decoding from scalar/tabular features.** The scientific backbone of the project. Layer is the broadest and most biologically robust label: it is tied to circuit position, input/output role, and dendritic morphology. If function carries identity information at all, it should appear here first, with interpretable engineered features.
- **Phase 2 — Layer decoding from raw temporal / multimodal CNN.** Only after the scalar/tabular pipeline of Phase 1 has produced an interpretable baseline. CNNs are powerful but uninterpretable on their own; they earn their place only as evidence that something *beyond* engineered scalars carries layer information.
- **Phase 3 — Within-layer excitatory subtype decoding (L5 / L6).** Only attempted after Phase 1/2 have stabilized. The rationale is empirical: dendritic-morphology work (Weis et al., 2025) shows excitatory neurons form a continuum with sharper clusters in **deep layers**, especially L5 and to a lesser extent L6. So the most realistic "subtype" target is L5 IT vs ET vs NP, and L6 IT vs CT, *within* layer.

### 1.4 Scientific question (re-statement)
> **How much of excitatory neuronal identity is recoverable from function alone, and at which level of granularity does that recovery remain reliable?**

This framing is stronger than "let us train a classifier" because it makes a partial or negative result still informative: if layer decodes but subtype does not, we are saying that within-layer excitatory variation is more morphological than functional at this measurement resolution.

### 1.5 Why this question is non-trivial
There is no a-priori guarantee that the calcium response to videos contains layer-discriminative information. V1 excitatory neurons across layers share a lot of basic visual selectivity (orientation, direction, spatial/temporal frequency), and most layer-discriminative properties (RF size, feedback influence, behavioral modulation, reliability) are subtle and confounded by recording quality and stimulus exposure. The project is therefore as much a **representation problem** (what to compute from the trace) as a **modeling problem** (which classifier to fit).

### 1.6 The neuron-level / observation-level mismatch — the central methodological principle
The label we want to predict is **neuron-level**: every nucleus has one cortical layer and one cell type. The functional observations we have are **trial-level or hash-level**: the same neuron is measured across hundreds of trials, on many stimuli, sometimes in multiple sessions. Every methodological choice in this project is downstream of this mismatch.

**Core rule.** *Because the label is neuron-level but the observations are trial/hash-level, every model must either use grouped splits and neuron-level aggregation of row predictions, or explicitly model each neuron as a permutation-invariant set of observations.* No exception.

This rules out three things:
- A row-level metric as the primary report (rows are not the unit of biology).
- A train/test split that allows the same `nucleus_id` to appear on both sides (would leak neuron-specific fingerprints rather than test layer-generalizable structure).
- A wide table with thousands of hash-specific columns as the primary representation (it pretends the mismatch isn't there and hides per-hash imbalance and missingness).

It positively requires:
- The canonical processed representation is **long** — one row per `(nucleus_id, hash)` or `(nucleus_id, hash, trial)`.
- Models output **neuron-level predictions** — either by aggregating row-level probabilities, or by consuming each neuron as a set of rows directly (set models).
- Train/test splits are always grouped by `nucleus_id`.

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

  Class imbalance is severe: L4 and L2/3 dominate, 5P-NP and L6 subtypes are small. This drives every choice on metrics, weighting, and Phase-3 scoping.

- **Quality flag:** `cc_abs` (correlation between recorded response and a digital-twin model). Median ≈ 0.4 but **systematically lower for 6P-CT/IT (~0.27)**. This means recording quality is *correlated with the label*, which is a confound we must control for explicitly (Section 6 and 10).

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
For each `nucleus_id` with at least one valid functional match, `best-only` keeps a single (session, scan, unit_id) — the highest-quality match. This avoids one source of repeated-measure dependence at the cost of discarding extra recordings. We treat best-only as the **primary** regime and pooled-scan as a **sensitivity analysis** (Section 6.4). Note that even in best-only, repeated measures across **trials and hashes** remain — the row count per neuron is large.

### 2.5 Variable levels (which axis each quantity lives on)

| Variable | Level |
|---|---|
| Layer / cell type | neuron |
| Calcium response trace | neuron × trial |
| Hash / stimulus identity | trial (and equivalently hash) |
| Stimulus visual descriptors (luminance, motion energy, etc.) | hash (because all trials of the same hash show the same stimulus) |
| Pupil, treadmill | trial (shared across all neurons in that trial; varies across repeats of the same hash) |
| Reliability / oracle correlation | computed across repeated trials within `(neuron × hash)` |
| Session / scan | recording-condition variable; potential confound |

These levels determine how each variable is allowed to enter the model. Stimulus descriptors can be safely attached at the hash level. **Behavior cannot be naively averaged after collapsing trials** — see §3.4.

---

## 3. Methodological cornerstones

These six decisions must be settled before any modeling. They shape the entire pipeline.

### 3.1 Best-only vs pooled scans
- **ML reason for best-only.** Removes one source of repeated measures, simplifies grouping, and eliminates the highest-quality bias driving any neuron's representation.
- **ML reason for pooled.** More data per neuron means more reliable feature estimates.
- **Neuroscience reason.** A neuron's response distribution is genuinely scan-dependent (anesthesia state, FOV, drift), so pooling is not "free more data" but introduces session-as-confound.
- **Decision.** Phase 1 reports use **best-only**. Pooled-scan is a robustness check: features re-computed by including all scans of the same `nucleus_id`, with grouped CV by `nucleus_id`. We compare deltas, not absolutes.

### 3.2 Unit of analysis — long is canonical, wide is derived
The canonical processed representation of the data is **long format**, not wide.

- **A0 — `(nucleus_id, trial)` long table.** One row per neuron per trial. Carries the trial-level neural summary, the trial-level behavior summary, the trial's hash and stimulus descriptors, and the session/scan ID.
- **A1 — `(nucleus_id, hash)` long table.** One row per neuron per hash, obtained by aggregating the repeated trials of that hash for that neuron. Carries the trial-averaged neural summary, hash-level stimulus descriptors, behavior-conditioned features (Tier C1, see §3.4), and reliability features (Tier B, see §3.3).

Long is the master representation because:

- It is faithful to the experiment: rows correspond to actual observations.
- Missingness and per-hash imbalance are explicit and inspectable.
- Adding a new feature family is just appending a column.
- Both A0 and A1 are derivable from the same underlying H5 reads, with the trial-averaging step the only thing that distinguishes them.

Wide tables are **derived views**, built on demand from the long master only when a particular Stage-1 model needs them. The motivating example was a wide matrix with 116 oracle hashes × K features — that view exists only as a model input for one specific experiment; it is not the cached canonical form.

### 3.3 To trial-average or not? — within-hash yes, across-hash never
Trial averaging is justified *within a hash* and is not justified *across hashes*.

- **Within-hash averaging is signal estimation.** Repeated presentations of the same stimulus are the textbook PSTH/firing-rate setting: averaging suppresses trial noise and approximates the stimulus-locked response. The trial-averaged trace per `(neuron, hash)` is the canonical "signal" estimator.
- **Within-hash variability is reliability.** What the average throws away — the trial-to-trial residuals — is exactly what reliability measures (Fano-like factor, coefficient of variation, oracle correlation, SNR) capture. Throwing it away is throwing away one of the few features known a-priori to differ between cell types. **Reliability is therefore a separate feature family (Tier B) computed from the trials *before* the within-hash average is taken.**
- **Across-hash averaging is destructive.** Different hashes are different stimuli with different content, contrast, and timing. Averaging across them collapses the per-stimulus tuning structure that may carry layer information. It is forbidden as a feature-engineering step in this project. Where the across-hash dimension must be summarized for a model, it is preserved either as multiple rows (long), as multiple columns (derived wide), or as separate models (per-family).

### 3.4 Behavior — conditional aggregation, not naive averaging
Behavior (running, pupil) is **trial-level**: shared across neurons within a trial, but **varying across repeated presentations of the same hash**. This second part is the critical one: even after we trial-average within a hash, the behavioral context of those trials was different. Naive trial-averaging of the neural trace destroys the interaction between behavior and response that is most likely to be cell-type-specific.

The correct handling depends on which level we are working at:

- **At A0 (trial-level rows).** Behavior is just a trial-level scalar (Tier C0): mean running speed, mean / change in pupil, etc. The interaction between neural response and behavior is captured by the model directly, because both live on the same row.
- **At A1 (hash-level rows).** Behavior must be turned into **behavior-conditioned neural features (Tier C1) computed before the within-hash trial-averaging**, not after. Concretely, for each `(neuron, hash)` with multiple trials:
  - Partition the trials by behavioral state (running vs still, high-pupil vs low-pupil).
  - Compute the response mean (and other neural statistics) within each state.
  - The C1 feature is the difference, ratio, or regression slope across those state-conditional means.
  - Examples: running modulation index, pupil-response coupling slope, state-conditional reliability.

The A1 row therefore carries one neural summary (the trial-average) plus a few behavior-conditioned scalars (the modulation indices). It does not carry "average pupil at this hash" — that quantity has no per-neuron meaning.

Biological grounding (Niell & Stryker 2010; Dadarlat & Stryker 2017): locomotion gates V1 responses, and the gain of that gating differs across layers — L2/3 and L5 ET show strong running-up modulation, L4 and L6 less. This is one of the few well-documented per-cell-type functional differences in mouse V1 and we expect Tier C1 to be one of the more discriminative feature blocks.

### 3.5 What counts as "functional data" — no precomputed tuning parameters, no orientation features
The brief constrains us to functional data only. We interpret this strictly: our inputs are the **raw functional traces, behavioral channels, and stimulus video**, not precomputed properties derived from them by other models or by other people's analysis pipelines.

- **Allowed inputs.** The calcium response time series in `microns_functional.h5`, pupil position/size, treadmill speed, the stimulus video, and session/scan/trial metadata.
- **Disallowed as features.** Precomputed tuning parameters from `digital_twin_properties_bcm_coreg_v4.csv` — `pref_ori`, `pref_dir`, `gOSI`, `gDSI`, the published oracle correlation, and any related index. These are tuning quantities the BCM digital twin model already computed for each neuron. Using them would (i) outsource the feature engineering to someone else's model, (ii) carry that model's specific assumptions, and (iii) defeat the scientific purpose of recovering identity from raw functional signal.
- **Orientation/direction tuning is excluded entirely.** `pref_ori`, `pref_dir`, `gOSI`, `gDSI`, and any orientation- or direction-selectivity scalar, **whether read off the CSVs or computed by us**, are out of scope per the project brief.
- **Disallowed as labels.** Predicted cell-type tables (`baylor_*`, `cg_cell_type_calls`, `cell_type_multifeature_combo`) are not used as targets. Our ground truth is `aibs_metamodel_celltypes_v661.csv` with corrections.
- **Allowed only as confound covariates.** `cc_abs` and other quality scalars derived from the digital twin appear in confound baselines (§6.5), never as model inputs.

### 3.6 Three modeling strategies for the neuron-level / observation-level mismatch
Given that observations are at row level and the label is at neuron level, three modeling strategies are admissible. The Phase-1 plan uses all three.

1. **One-row-per-neuron weak baseline.** Sample one row per neuron (e.g. one trial picked at random, or the trial-average of all trials regardless of hash — the latter only allowed *as a baseline*, see §3.3 caveat). Train a standard tabular model. This is a **diagnostic** baseline, intentionally weak, that tells us how badly we lose by ignoring the mismatch. It should never be the headline result.
2. **Long-row model + neuron-level probability aggregation.** Train a tabular model (logistic regression, random forest, gradient-boosted trees) on long rows, with `GroupKFold` by `nucleus_id`. At test time, predict probabilities for every row of every test neuron, then **average those probabilities across the rows of the same neuron** to get one probability vector per neuron. The final predicted label is the argmax of the averaged neuron-level probabilities. This is the **mandatory classical baseline** of Phase 1.
3. **Set models.** Each neuron is represented as a set of rows (its trials, or its hashes). Use Deep Sets (Zaheer et al., 2017) or a Set Transformer (Lee et al., 2019) to learn the aggregation internally. This is the most conceptually faithful framing because the order of hashes and trials is arbitrary; the model is required to be permutation-invariant. Set models are tried whenever computational budget permits.

The choice of which strategy is the headline depends on the empirical results — strategy 2 is the floor every model must beat, and strategy 3 is the natural ceiling for the engineered-feature pipeline.

---

## 4. Feature representations (the tiers)

### 4.0 The framing in plain terms

Each tier is a hypothesis about what makes neurons of different layers look different. They are added together to form the long-row feature matrix that goes into Phase-1 modeling. The tiers are organized so that the **neural-response family is built first, then behavior is added, then stimulus context** — because that is the order in which biology is most likely to carry signal, and because each tier should be ablatable from the next.

The tier suffixes `0` and `1` denote the granularity:

- `*0` features live on the trial-level long table (A0 long: one row per `(neuron, trial)`).
- `*1` features live on the hash-level long table (A1 long: one row per `(neuron, hash)`, after within-hash trial-averaging).

A tier without a suffix is granularity-agnostic (e.g. Tier B is naturally `(neuron, hash)` because reliability is defined across repeated trials of the same hash; Tier G is naturally a per-neuron summary built from A1).

### Tier A0 — neuron × trial neural summary
- **Row.** One per `(nucleus_id, trial)`.
- **What.** Scalar features summarizing the trial's calcium trace, in two sub-blocks for clean ablation:
  - **Amplitude / statistical (sub-block A0-amp).** Pre-stimulus baseline, mean response during stimulus, peak, integral / AUC, post-stimulus tail mean, response variance during stimulus.
  - **Temporal shape (sub-block A0-shape).** Latency / time-to-peak, rise slope, decay slope, early AUC (first half of stimulus), late AUC (second half), early/late ratio, adaptation ratio across the trace, and optionally a small number of PCA components of the trial trace fit on the training set.
- **Why temporal shape is here, not in a later tier.** The temporal shape of a single trial is *still a property of the neural response*, just compressed. Putting it next to amplitude in A0 lets us ablate `(amp only)` vs `(amp + shape)` cleanly.
- **Failure mode.** Calcium temporal blur (~500 ms GCaMP6s) is a hard ceiling on what shape features can distinguish.

### Tier A1 — neuron × hash neural summary
- **Row.** One per `(nucleus_id, hash)`, obtained by within-hash trial-averaging of the calcium trace.
- **What.** The same scalar feature set as A0 (amplitude + temporal shape sub-blocks), computed on the trial-averaged trace.
- **Why.** The within-hash average is the cleanest estimator of the stimulus-locked response. Many cell-type properties (latency, adaptation) are clearer on the averaged trace than on noisy single trials.
- **Note.** A1 is *not* obtained by averaging the A0 features across trials of the same hash. The trace is averaged first, then the features are computed on the averaged trace.

### Tier B — reliability / trial-to-trial structure  *(attached to A1)*
- **Row.** One per `(nucleus_id, hash)`, attached to the A1 row.
- **What.** Statistics computed across the trials of that `(neuron, hash)` *before* the within-hash average:
  - trial-to-trial variance and standard deviation,
  - coefficient of variation,
  - Fano-like factor (variance/mean of per-trial response amplitudes),
  - signal-to-noise ratio (mean / std),
  - within-hash trial-trial Pearson correlation of the full traces,
  - oracle correlation when the hash has enough trials (the canonical reliability scalar; Sinz/Tolias-style),
  - state-conditional reliability when behavior is partitioned (also feeds C1).
- **Per-neuron summary.** Tier B values can be averaged across hashes per neuron to give a per-neuron reliability scalar set, useful as a low-dimensional input or for interpretability.
- **Why.** Cell types differ in trial-to-trial reliability independently of tuning. L4 cells receive direct LGN input and are typically the most reliable; L5 ET cells are noisier and more state-modulated; L6 cells are reliability-low and tuning-narrow.
- **Failure mode.** Reliability dominated by `cc_abs` quality — controlled by the `cc_abs` confound baseline (§6.5).

### Tier C — behavior
Behavior comes in two layers, matching A0 and A1 granularity.

**Tier C0 (trial-level)** — attached to A0.
- **Row.** One per `(nucleus_id, trial)`.
- **What.** Trial-level behavior summaries: mean / max running speed, mean and change of pupil dilation, pupil-position statistics, trial-level arousal proxies.
- **Note on per-neuron meaning.** Because behavior is shared across all neurons in a trial, raw C0 alone is not neuron-specific; its information emerges only through the model's interaction with the neural channel on the same row.

**Tier C1 (hash-level, behavior-conditioned)** — attached to A1.
- **Row.** One per `(nucleus_id, hash)`.
- **What.** Per-neuron scalars derived from partitioning the trials of that hash by behavioral state and computing state-conditional response statistics (§3.4):
  - running modulation index = (mean response when running − mean response when still) / sum,
  - pupil-response coupling slope (regression of trial response on trial pupil within hash),
  - state-conditional reliability (Tier B computed separately on running vs still subsets),
  - response gain difference between high and low arousal.
- **Why.** This is the interaction between behavior and neural response, distilled into per-`(neuron, hash)` scalars. It is the right way to use shared behavioral channels as per-neuron features.

### Tier D — stimulus descriptors
Stimulus descriptors are **hash-level** because all trials of the same hash show the same stimulus. They can therefore be safely attached at either granularity without conditional-averaging machinery.

**Tier D0 (trial-level)** — attached to A0 by joining on hash.
- **Row.** One per `(nucleus_id, trial)`.
- **What.** Hash-level stimulus descriptors duplicated across the trial rows of that hash.

**Tier D1 (hash-level)** — attached to A1.
- **Row.** One per `(nucleus_id, hash)`.
- **What.** Hash-level stimulus descriptors:
  - stimulus family (Clip / Monet2 / Trippy) as a categorical,
  - global luminance mean / std,
  - contrast,
  - frame-difference / motion energy summary,
  - spatial-frequency content proxies (e.g. spectral energy in low/mid/high SF bands),
  - temporal-frequency content proxies,
  - other hash-level summary statistics.
- **Excluded.** No orientation- or direction-tuning features (§3.5), even when the stimulus is a parametric Monet2 grating.

### Tier E — temporal traces  *(Phase 2 territory)*
Tier E is the un-summarized version of A0/A1: the raw trace itself, fed into temporal models.

**Tier E0 (trial-level).**
- **Row.** One per `(nucleus_id, trial)`.
- **What.** The full calcium trace for that trial, optionally with companion behavior traces (treadmill, pupil) and stimulus-derived temporal channels (luminance, contrast, motion energy as time series).

**Tier E1 (hash-level).**
- **Row.** One per `(nucleus_id, hash)`.
- **What.** The within-hash trial-averaged calcium trace, optionally with within-hash trial-averaged behavior and stimulus channels. Cleaner than E0 for asking "what is the stimulus-locked temporal shape per cell type?" because the noise has been suppressed; loses the behavioral-state interaction that E0 retains.

E0 and E1 are the inputs to the **Phase-2** CNN pipeline (§7), not Phase 1.

### Tier F — Receptive-field-derived features  *(OPTIONAL · deferred to last)*

**Status.** Tier F is **optional** and is the **last** thing we attempt — only after Phases 1, 2, and 3 have produced defended results. Two constraints bind any future Tier F implementation:

1. **No orientation tuning, in any form.** `pref_ori`, `pref_dir`, `gOSI`, `gDSI`, and every orientation- or direction-selectivity feature — whether read off the structural CSVs *or computed by us from the raw data* — are out of scope per the project brief. Tier F therefore restricts itself to **spatial RF descriptors** (location, size, shape, temporal lag) and intentionally does not include any orientation-tuning channels or features.
2. **Preliminary work in this direction has not been encouraging**, so a Tier F revisit would need a redesigned implementation: gaze correction via the pupil channel before STA, per-neuron STA-quality filtering, and rigorous validation against the existing pipeline. Not building Tier F is the safe default; we ship without it.

If revisited: per-neuron spatiotemporal kernel via gaze-corrected STA → 2D Gaussian fit (σx, σy → RF size, aspect ratio), spatial-frequency peak via 2D FFT (magnitude only, no orientation tuning), temporal lag of the peak. The kernel can also be used to define an RF-aligned response feature for stimulus-locked analysis.

### Tier G — Cross-stimulus / oracle-hash consistency
- **Built from A1.** Per neuron, take the vector of trial-averaged response amplitudes across the 116 oracle hashes (only oracle hashes, because they are shared across all sessions). This is a per-neuron 116-D "stimulus fingerprint".
- **What can be derived.** Cosine similarity matrices across neurons, response-rank profiles, low-rank embedding of the fingerprint matrix (PCA across neurons → loadings), and the fingerprint vector itself when used as a model input.
- **Why.** This is the strongest *cross-session* representation we have because the oracle hashes are the same in every session. Comparing fingerprints across sessions is fair in a way that comparing per-hash features generally is not.
- **Failure mode.** 116 hashes is small relative to neuron count; overfitting risk on the wide 116-D fingerprint without regularization.

### Tier H — Foundation-model embeddings  *(OPTIONAL · extension only)*
The Wang et al. (2025) MICrONS digital-twin per-neuron readout vectors (~1024-D). **Optional**, described in §11. Not part of the main pipeline; listed here only for ablation-table completeness if loaded.

### What is unlikely to be sufficient on its own
- A single-row-per-neuron baseline that throws away the long structure (§3.6 strategy 1) — diagnostic only.
- Pooling all trials with no hash structure (kills C0/C1, B, D1).
- Behavior added at A1 without conditional aggregation (kills the interaction; §3.4).
- A CNN trained on raw traces before scalar baselines have established what is decodable.

---

## 5. Phase 1 — modeling roadmap (controlled ablation study)

Phase 1 is framed as a **controlled ablation study** of the question:

> *Given a neuron as a set of stimulus-conditioned functional observations, which aspects of function recover layer identity?*

Each stage adds at most one new feature family on top of the previous one and is run with **identical train/test neuron splits**, so the deltas are fair. Every Phase-1 model is one of the three strategies in §3.6 (single-row baseline, long-row + neuron-level probability aggregation, set model), with the long-row + aggregation as the mandatory baseline.

### Stage 1 — Neural response only: A0 vs A1
- **Inputs.** Tier A0 (trial-level) vs Tier A1 (hash-level), each tested separately.
- **Question.** Does within-hash trial-averaging cost or save us information at the prediction level? Is layer information already accessible from stimulus-locked neural response statistics?
- **Models.** Logistic regression (L1/L2), random forest, gradient-boosted trees, plus a set model where computationally feasible. All with `GroupKFold` by `nucleus_id` and neuron-level probability aggregation at test time.
- **Sub-ablation inside A0/A1.** Amplitude-only vs amplitude + temporal-shape sub-blocks, to isolate whether response shape adds anything beyond magnitude.
- **Decision rule.** If both A0 and A1 are at chance with calibrated baselines, stop here and run Tier B before going further. If above chance, escalate to Stage 2.

### Stage 2 — Neural + reliability: A1 vs A1 + B
- **Inputs.** A1 only, then A1 + B.
- **Question.** Does the trial-to-trial structure carry layer information beyond the trial-averaged signal?
- **Ablation.** With/without Tier B, same models, identical splits. Report Δ in balanced accuracy and per-class recall (especially L5 ET and L6 CT, where reliability is predicted to differ).
- **Note.** We can also run Tier B as a per-neuron reliability scalar set on the one-row-per-neuron baseline — this is one of the rare cases where a per-neuron scalar set is biologically meaningful, since reliability is an intrinsic neuron property.

### Stage 3 — Add behavior: A0 + C0 vs A1 + B + C1
- **Inputs.** Two parallel paths.
  - **Trial-level path:** A0 (amp + shape) + C0.
  - **Hash-level path:** A1 + B + C1.
- **Question.** Does trial-level behavior contain information that is *lost* when we summarize at A1 with engineered C1 features? Or are the C1 modulation indices sufficient?
- **Models.** Same set as Stage 1; both paths benefit from neuron-level probability aggregation at test time.
- **Ablation.** With/without behavior on each path; with/without C1 on the hash-level path.

### Stage 4 — Add stimulus context: A0 + C0 + D0 vs A1 + B + C1 + D1
- **Inputs.** Both paths from Stage 3, with stimulus descriptors attached.
- **Question.** Does explicit stimulus context improve decoding? On the hash-level path, does D1 (hash-level stimulus descriptors) play naturally with A1's already-stimulus-locked signal?
- **Decision rule.** Compare the best model from each path. The headline Phase-1 model is the better of the two after this ablation chain.

### Stage 5 — Family analysis (three distinct uses)
This is run on the best Phase-1 model from Stage 4, and is reported as three separate experiments because each asks a different question.

1. **Family-level aggregation as a sanity ablation.** Rebuild the long table with rows at `(neuron, family)` granularity (three rows per neuron, summarizing across hashes within each family). If the family-level model performs nearly as well as the hash-level model, fine-grained hash identity adds little. If hash-level outperforms, per-stimulus tuning is doing real work. Either result is informative.
2. **Per-family pipelines as a real biological question.** Train Stage-4 separately on Clip-only, Monet2-only, Trippy-only hashes (filtered long table). Compare layer-decoding performance across families. The neuroscience question is whether layer is more recoverable from naturalistic clips or from synthetic stimuli — a real claim the report can make.
3. **Cross-family transfer as a domain-shift experiment.** Train on one family, evaluate on another. **Reported explicitly as a transfer / generalization experiment, not a benchmark robustness check.** A score collapse here is a *positive* scientific finding — it would say layer's functional signature is family-specific.

The three uses are kept distinct in the report so a transfer-failure is not misread as a model failure.

### Decision logic between stages
- If a stage is at chance, do not silently escalate to a more complex model — first add the next feature family that the workflow predicts should help. If still at chance after all engineered tiers, the conclusion is real (function does not carry layer information at this resolution), and Phase 2 might still find something.
- If a stage is well above chance, run the §6.5 confound baselines on it before claiming biology. A multi-channel model that outperforms a session-only baseline by 1% is probably not biology.
- Identical neuron splits across stages are mandatory; otherwise the deltas are meaningless.

---

## 6. Evaluation, splitting, and confound control

### 6.1 Golden splitting rule
**Split by `nucleus_id`, never by row.** Every row generated from the same neuron must end up in the same fold. Enforced with `GroupKFold` (or `StratifiedGroupKFold`) on `nucleus_id`. The same neuron never appears in both train and test.

**Note on splitting by scan.** We do *not* split by scan in the primary protocol. Two different neurons recorded in the same scan are allowed to fall on opposite sides of the train/test split — every scan contains multiple cell types, so blocking by scan would prevent the model from learning within-scan distinctions. What we never allow is the *same* `nucleus_id` on both sides. Whether the scan itself is a confound is a separate question, addressed in §6.1bis.

### 6.1bis The scan composition confound
The 14 session–scan blocks were not recorded with a uniform mix of cell types. Each scan placed its FOV at a particular location and depth, and ended up with whichever neurons fell in that FOV. As a result, **"which scan a neuron is from" is itself correlated with the layer label**, even before any biology. A model that uses neural features can end up predicting layer through session-specific recording artefacts (noise floor, photobleaching, behavioral state distribution) rather than biology.

The handling is layered:

- **Session-only baseline (§6.5).** A classifier whose only input is the session/scan ID one-hot. If that already beats chance on layer, then session is informative about layer and any "real" model has to beat that bar by a meaningful margin.
- **Per-session standardization.** Z-score every feature within its session before pooling, so session-level offsets and scales cannot be read by the classifier as a layer signal.
- **Session as covariate.** In linear models, include session-fixed-effect dummies.
- **Leave-one-session-out as a sanity check.** Beyond `GroupKFold` by `nucleus_id`, additionally run a leave-one-session-out CV. Sharp drop versus standard grouped-CV indicates session-leaning behavior. Both are reported.

### 6.2 Stratification
Stratify folds by layer label as much as group constraints allow (`StratifiedGroupKFold`). In Phase 3 (subtype), stratify by subtype within layer.

### 6.3 Metrics — neuron-level only
Final metrics are computed at the **neuron level**, never at the row level, because the label and the scientific claim live at the neuron level.

- For long-row models: aggregate per-row predicted probabilities to per-neuron probability vectors (mean across the neuron's rows), then take argmax for the predicted label and compute metrics on those neuron-level predictions.
- For set models: the model outputs neuron-level predictions directly.
- **Never aggregate hard labels across rows** (mode-of-predictions is information-poor and unstable when probabilities are close).
- **Equal-row-count handling.** Neurons can have very different row counts (different numbers of valid hashes/trials). Mean-of-probabilities weights every row equally, so high-row-count neurons do not dominate per-neuron predictions; they are simply estimated more confidently.

Primary metrics: balanced accuracy and macro F1. Secondary: confusion matrix, per-class recall (especially the minority classes), one-vs-rest AUC for subtype experiments. Calibration baselines: majority-class, stratified-random.

### 6.4 Robustness checks (run on the best Stage)
- Best-only vs pooled-scan.
- Per-stimulus-family vs all-families (also informs §5 Stage 5 family analyses).
- With and without `cc_abs` quality covariate.
- With and without per-session standardization.
- Stability across random seeds.

### 6.5 Confound baselines (mandatory)
Always reported alongside any headline number:

- **Quality-only baseline.** Only `cc_abs` and other quality scalars as input. If the headline model does not beat this, we are decoding quality, not biology.
- **Scan-membership baseline.** Only the session/scan ID one-hot as input. If above chance, scan composition is informative; the headline must beat this margin.
- **Cortical-depth baseline.** Only cortical depth (`nucleus_y`) as input. The structural depth-only baseline is the most important comparator because depth and layer are tightly linked.

### 6.6 Reporting format
For every experiment: balanced accuracy ± std over folds at the **neuron level**, confusion matrix, per-class recall, comparison to baselines, and an ablation table showing the marginal contribution of each tier added.

---

## 7. Phase 2 — raw temporal / multimodal CNN

Phase 2 starts only after Phase 1 has produced a Phase-1-best result and the §6.5 confound baselines have been run on it.

**Why this order.** First we need to know what is decodable from interpretable engineered features. CNNs are more powerful but less interpretable. If a CNN improves performance over the Phase-1 best, we can interpret the gain as evidence that temporal / multichannel structure carries information beyond scalar summaries. Without that scaffold, a CNN result is a number without a story.

### 7.1 Inputs (Tier E)
- **E1 (hash-level trial-averaged trace) for stimulus-locked temporal shape.** Recommended starting point. Cleaner signal because trial noise is suppressed; loses behavioral-state interaction.
- **E0 (trial-level trace) for trial-level multimodal dynamics.** Required if behavioral modulation is a key axis to test inside the CNN.
- **Multichannel inputs** (calcium + treadmill + pupil + stimulus-derived temporal channels) are valid from the start; tested with channel ablations to identify what the CNN actually uses.

### 7.2 Architectures
- 1D-CNN on temporal traces (per-row), with neuron-level aggregation in the head.
- Set model over per-row trace embeddings (Deep Sets / Set Transformer over CNN-encoded traces) when budget allows.
- Architectural priors from preliminary group exploration: shallow CNNs with ~80–160 total filters, kernel sizes around 5–7, dropout ≈ 0.3, no residual connections at this scale.

### 7.3 Mandatory rules for Phase 2
- **Neuron-level outputs only.** Either row-level predictions aggregated per neuron, or set models that aggregate inside.
- **`GroupKFold` by `nucleus_id`.** No per-trial split that lets the same neuron's trials appear on both sides — that is not a CNN result, it is a memory result.
- **§6.5 confound baselines run alongside.** A CNN that beats Phase-1 best by 1% is not a Phase-2 result if the session-only baseline is also high.
- **Compared to Phase-1 best, not to chance.** A "Phase-2 reaches X% accuracy" number is meaningful only relative to the engineered-feature ceiling.

### 7.4 Channel ablation
Ablating each channel of the multimodal input one at a time isolates which modality carries the signal. The same channel ablation also serves as the §8 interpretability output for the CNN.

---

## 8. Phase 3 — cell-type and within-layer subtype decoding

Phase 3 starts only after Phase 1 / Phase 2 have produced a stable layer decoder. It is a **hybrid** — three parallel deliverables that complement each other — rather than a single task.

### 8.1 Inclusion and sample sizes (best-only)
- L5: 5P-IT ≈ 1.2k, 5P-ET ≈ 320, 5P-NP ≈ 26.
- L6: 6P-IT ≈ 216, 6P-CT ≈ 121.

5P-NP is small but is a real L5 cell type and **stays in the dataset**; we do not silently merge it into 5P-IT or 5P-ET. The way we handle the small-sample asymmetry is through the *reporting structure* (the headline metric is binary, not multi-class), not through dropping the class. The same logic applies to L6.

### 8.2 Three parallel Phase-3 deliverables

**(a) Within-layer binary tasks — the headline.** Cleanest tests of within-layer functional structure because they hold the layer signal constant.
- **L5 binary: 5P-IT vs 5P-ET.** Restricted to L5 (5P-NP excluded *only* from this binary). Dominant L5 axis in the literature.
- **L6 binary: 6P-IT vs 6P-CT.** Restricted to L6.

**(b) Within-L5 3-way as secondary.** A 3-way 5P-IT / 5P-ET / 5P-NP confusion matrix on the full L5 population. Reported with explicit small-sample caveats on 5P-NP. The diagnostic question is whether 5P-NP confuses with IT, with ET, or distributes broadly.

**(c) Full multi-class cell-type prediction — parallel headline.** A 6- or 7-class cell-type prediction over all V1 excitatory subtypes simultaneously. This is the harder, layer-confounded version of the question. Aggregating its predictions to layer also gives an internal consistency check against the Phase-1 layer score.

**Reading the three together.** Layer decodable but neither within-layer binary works → function carries layer but not subtype. Layer decodable and at least one binary works → genuine sub-layer structure for that axis. Full multi-class beats layer baseline (when aggregated) → model is using something beyond layer alone.

### 8.3 Predicted markers per subtype
- **5P-ET vs 5P-IT.** ET cells project subcortically and are expected to have stronger running modulation, broader RFs, and higher gain. **Tier C1 (running modulation index)** is the predicted dominant feature; Tier F spatial RF size if Tier F is ever attempted.
- **6P-CT vs 6P-IT.** CT cells have narrow tuning and quieter baselines; they are also lower in `cc_abs`. **Tier B (reliability) and Tier D1 selectivity-related descriptors** are the predicted dominant features. `cc_abs` confound control is critical here precisely because the quality bias is largest.

### 8.4 Models
- **Within-layer binaries (a)** and **3-way (b).** Linear baselines first (logistic regression with L2). Tree models (gradient boosting) second. Deep models are not justified at these sample sizes.
- **Full multi-class (c).** Same simple-models-first progression. A Phase-2 deep model can be tested only after the simpler models have established a baseline and only with §6.5 confound baselines reported alongside.

### 8.5 Metrics
Balanced accuracy, macro F1, and per-class recall, with bootstrap CIs over neurons. Confusion matrices for all three deliverables. Classes with too-wide CIs (5P-NP, 6P-CT) flagged in reporting.

---

## 9. Interpretability and ablations

Interpretability serves two purposes: validating that the model is using biology rather than confound, and turning a number into a scientific claim.

### 9.1 Tier ablation table
The single most informative output of the project. For the best Phase-1 model (and separately for the Phase-2 CNN), report **balanced accuracy when each tier is *removed*** from the full feature set:

- A0-amp (amplitude only),
- A0-shape,
- A1 (each sub-block),
- B,
- C0,
- C1,
- D0,
- D1,
- G.

The tier whose removal hurts most carries the layer information. The tier whose removal does not change the score is not part of the story.

### 9.2 Per-model interpretability
- **Linear models.** Standardized coefficients per class; per-tier and per-feature inspection; sparse (L1) variants for selection.
- **Tree models.** Permutation importance and SHAP (selectively, since SHAP is expensive).
- **Set models.** Attention weights per row; ablate rows by hash, family, or behavioral state.
- **Phase-2 CNN.** First-layer filter visualization, temporal occlusion ablations, channel ablation.

### 9.3 Scientific output to aim for
We want the report to read as:
> "Layer information appears to be carried mainly by [stimulus-specific reliability and behavioral modulation], with [temporal shape] adding a small but consistent margin. Per-family pipelines show layer is more recoverable from [naturalistic / synthetic] stimuli."

Not:
> "The CNN got 47% balanced accuracy."

---

## 10. Risks and confounds (consolidated)

| Risk | Mechanism | Control |
|------|-----------|---------|
| Repeated-measure leakage | Same `nucleus_id` across train/test | `GroupKFold` by `nucleus_id` always |
| Row-level metric misreport | Reporting row-level scores instead of neuron-level | Aggregate predicted probabilities per neuron before metrics |
| Quality–label confound | `cc_abs` lower for 6P → quality decodes label | `cc_abs`-only baseline; per-class quality reporting |
| Scan composition confound | Some layers concentrate in some sessions | Scan-only baseline; per-session standardization; LOSO sanity check |
| Class imbalance | Minority classes inflate variance | Balanced metrics, class weights, headline binary tasks for Phase 3 (small classes kept in data, flagged in reporting) |
| Wide-table backbone | Hides per-hash imbalance and missingness; doesn't scale | Long is canonical; wide is derived only when a model needs it |
| Behavior averaged after trial collapse | Destroys neural × behavior interaction | C1 features computed *before* within-hash averaging |
| Across-hash averaging | Destroys per-stimulus tuning | Forbidden as a feature-engineering step |
| Temporal blur from GCaMP6s | <500 ms cell-type differences invisible | Acknowledge as ceiling; lean on amplitude / reliability / shape rather than fast latency |
| Overfitting deep models | ~9k neurons, severe imbalance | Group-CV, early stopping; deep models only when justified by simpler ones |
| Family-transfer misread as model failure | Cross-family score collapse can be biology, not bug | Report cross-family experiments explicitly as transfer / generalization, not benchmark |
| Biological overclaiming | "Decodes layer" ≠ "function determines layer" | Report decoding as statistical association, not causal coding |

---

## 11. Possible extensions

Out-of-scope for the main project but useful as discussion or stretch goals.

- **Tier H — Foundation-model embeddings.** The Wang et al. (2025) MICrONS digital twin's per-neuron readout vectors (~1024-D). The model is built as a shared visual backbone + a per-neuron readout head; the readout head is a compact description of the neuron's functional preferences. Two uses: (i) upper-bound benchmark against our hand-crafted tiers; (ii) complementary feature block to test whether engineered tiers add independent information. Trade-off: interpretability (a coefficient on dim 743 of the embedding does not tell a biological story).
- **Self-supervised pretraining on raw traces.** Train a Transformer/CNN encoder on the unlabeled response set with masked-trace reconstruction; fine-tune for layer.
- **Cross-area generalization.** Train on V1, evaluate on RL/AL/LM (out of scope by group decision, strong robustness check).
- **Within-neuron stability tests (pooled-scan regime).** For neurons appearing in multiple scans, test prediction stability across scans of the same `nucleus_id`. Predictions that flip across scans imply session decoding rather than identity decoding.
- **Connectivity-based sanity check.** MICrONS has synaptic connectivity (Ding et al., 2025). Neurons with similar predicted-class probability could be checked against synaptic connectivity patterns as structure-side validation we *compare against* rather than *use*.
- **Tier F (RF descriptors).** As described in §4. Optional, deferred, last; orientation-tuning features excluded.

---

## 12. Report narrative (2-page format)

The report mirrors the methodology, not the chronology of the work.

- **Introduction.** MICrONS lets us link cell identity to function. Open question: how much identity is recoverable from function alone? Restrict to V1 matched excitatory.
- **Question.** Phase 1 layer; Phase 2 deep-model layer ceiling; Phase 3 within-layer subtype.
- **Method.** Best-only regime, **long-table backbone with neuron-level prediction aggregation**, tiered feature engineering ordered neural → reliability → behavior → stimulus, grouped CV by `nucleus_id`, mandatory confound baselines.
- **Results.** Layer decoding result (Phase-1 best vs Phase-2 CNN), tier ablation table, per-family pipeline comparison, cross-family transfer flagged as domain-shift, Phase-3 hybrid result with bootstrap CIs.
- **Discussion.** What carries the signal, what does not, where the calcium temporal blur and class imbalance limit us, what a negative subtype result would mean (function continuous, identity discrete in morphology only), and what the cross-family transfer says about stimulus-family-specific functional signatures.

---

## 13. Literature anchors

A small, focused set rather than an exhaustive list.

- **MICrONS dataset and function–connectivity.** Fahey et al., *Functional connectomics spanning multiple areas of mouse visual cortex*, Nature 2025. Ding et al., *Functional connectomics reveals general wiring rule in mouse visual cortex*, Nature 2025.
- **Cell identity organization.** Gouwens et al., *Classification of electrophysiological and morphological neuron types in the mouse visual cortex*, Nat. Neurosci. 2019. Weis et al., *An unsupervised map of excitatory neuron dendritic morphology in the mouse visual cortex*, Nat. Comm. 2025.
- **Function carries identity information.** Schneider et al., *Transcriptomic cell type structures in vivo neuronal activity across multiple time scales*, Cell Reports 2023.
- **Behavioral state in V1.** Niell & Stryker, *Modulation of visual responses by behavioral state in mouse visual cortex*, Neuron 2010. Dadarlat & Stryker, *Locomotion enhances neural encoding of visual stimuli in mouse V1*, J. Neurosci. 2017.
- **Set-level neuron representation.** Zaheer et al., *Deep Sets*, NeurIPS 2017. Lee et al., *Set Transformer*, ICML 2019.
- **Repeated-stimulus / reliability framework.** Classical neural-encoding / PSTH logic for within-stimulus averaging; Fano-factor and oracle-correlation-style measures of trial-to-trial reliability are standard tools in the field.

---

## 14. Final formulation

> We ask whether the anatomical identity of excitatory V1 neurons in the MICrONS dataset is recoverable from function alone. The biological label is neuron-level, but the functional observations are trial-level and hash-level; this mismatch is the central methodological constraint, and every model in this project either uses grouped train/test splits with neuron-level aggregation of row predictions, or models each neuron as a permutation-invariant set of observations. The pipeline is a controlled ablation study built on a long-format master table (one row per `(neuron, trial)` or `(neuron, hash)`), adding feature families in a fixed order — neural response → reliability → behavior → stimulus context — and running interpretable engineered models first, then a multimodal CNN, and finally within-layer subtype decoding. The contribution of the work is not a single accuracy number but the ablation map: an account of which functional features carry layer and subtype identity, and which do not.
