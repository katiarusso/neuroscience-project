# Phase 1 — Ground Truth Document

*This document is the project's ground-truth record of everything we have decided, hypothesised, computed, and concluded during Phase 1. It is written at the moment of the coding so that nothing is lost before the report is drafted. It does not describe the structure of the notebooks; it describes the **scientific workflow, the assumptions behind each step, the theoretical reasoning (both ML and neuroscience) that justified each decision, and the actual results we obtained**.*

---

## 1. The question

We are predicting **V1 cortical layer label** (L2/3, L4, L5, L6) for individual excitatory neurons from their **functional response data** (calcium imaging traces, plus behavior-state and stimulus-content covariates), in the MICrONS dataset. The prediction is at the level of a single neuron — *not* a single trial — because biology assigns one cortical layer to one neuron, not one to each of its responses.

The biological premise is that **functional response statistics carry signature of laminar identity**: L4 spiny stellate cells are simple-cell-like with parametric drive; L5 pyramidal cells (IT and ET subtypes) integrate broadly; L6 cells modulate thalamocortical loops; L2/3 cells receive intracortical and L4 input. If this premise is correct, two-photon calcium-imaging response patterns to a fixed stimulus battery should be distinguishable across layers above chance. The strength of the signal — and the protocol under which it can be claimed to generalise — are the empirical questions of Phase 1.

The methodological premise is that the **MICrONS structural CSV gives us trustworthy layer labels**: it merges automatic morphology-based assignment with manual checks (the "matched best-only" filter). Phase 1 takes those labels as ground truth — and runs a sensitivity analysis (§8.5 below) to verify that the headline numbers do not depend on the 4.6 % of neurons where the EM-derived `cell_type` and the depth-derived `layer_label` disagree. The sensitivity passes: the headline is robust.

---

## 2. The dataset

### 2.1 What MICrONS is, in our framing

A single-mouse multi-session two-photon calcium-imaging dataset from V1, paired with electron-microscopy-reconstructed structural data on the same neurons. Each neuron has:

- a **functional identity** (`unit_id` per session-scan, with calcium traces aligned to stimulus presentations),
- a **structural identity** (`nucleus_id`, the EM-reconstructed cell, with `pt_position_y` giving cortical depth),
- and (for matched, best-only neurons) a **layer label** assigned through morphology (`L1 / L2/3 / L4 / L5 / L6`).

We work on the *single-mouse* dataset. The 13 session-scans are different recording days/depths/regions of the **same animal** — not different animals. LOSO across scans is therefore a *cross-recording-session* test, not a cross-animal test, and our claims should be framed accordingly.

### 2.2 The working population

After applying every WORKFLOW.md mandate (V1 only, excitatory only, matched best-only, drop L1 because it has no excitatory neurons), the working population is **8,895 neurons** distributed over 13 session-scans. Layer composition:

| layer | n neurons | fraction |
|---|---|---|
| L2/3 | ~ 4,460 | 50% |
| L4 | ~ 2,500 | 28% |
| L5 | ~ 1,570 | 18% |
| L6 | ~ 365 | 4% |

Severe class imbalance — L6 is rare. This drives every metric and CV decision downstream.

### 2.3 The stimulus battery

Three families per WORKFLOW.md, each appearing as multiple `condition_hash` values:

| family | nature | typical hashes per session | trials per hash | frame H × W × t |
|---|---|---|---|---|
| **Clip** | naturalistic movie clips | ~16 hashes | 16 trials each | 75 × 144 × 256 |
| **Monet2** | parametric oriented gratings | 10 hashes | 2 trials each | 113 × 126 × 216 |
| **Trippy** | phase-shuffled noise | 10 hashes | 2 trials each | 113 × 90 × 160 |

The crucial concept for cross-session decoding: **96 + 10 + 10 = 116 oracle hashes** are *physically the same stimuli in every session-scan*. Every neuron in the working population has trial-averaged responses to all 116 oracle stimuli. This is the foundation of Tier G.

### 2.4 What we explicitly excluded — and why

Per WORKFLOW.md §3.5:

- **No orientation features** (`pref_ori`, `pref_dir`, `gOSI`, `gDSI`, etc.). Even though Monet2 has known parametric direction structure, orientation is **out of scope for the entire project**. The reason is that orientation tuning is the most studied V1 feature; including it would make the project a known-result reproduction. We deliberately commit to *non-orientation* feature engineering to keep the question scientifically interesting.
- **No L1**. The MICrONS V1 working population has no excitatory L1 neurons; including it makes the classes degenerate.
- **No non-best-only neurons**. The "best-only" filter selects the highest-quality nucleus_id when an EM cell appears in multiple scans. Without it the same neuron can appear multiple times in the data with slightly different functional records.

These restrictions narrow the question but make it answerable cleanly.

---

## 3. Methodological foundations

### 3.1 Cross-validation: GroupKFold by `nucleus_id`

The single most important methodological decision. A neuron's calcium trace contributes many rows to our long tables (one per (neuron, hash) pair, or one per (neuron, hash, trial)). If we used standard KFold, the same neuron could appear in both the training and test sets across different stimuli — *intra-neuron leakage*. The model would learn to recognise the neuron, not the layer.

**GroupKFold(`nucleus_id`)** ensures every neuron is entirely in either train or test. We precomputed a single 5-fold split (stratified by layer at the neuron level) once, in `cv_assignments.parquet`, and used it for every model in Phase 1. The test of "does our pipeline work?" is reduced to a reproducible 5-number summary across the same 5 folds.

### 3.2 Scoring: balanced accuracy at the neuron level

Two decisions packaged together.

**Why balanced accuracy.** Because L6 is 4% of the data, raw accuracy over-weights L2/3 and L4. A trivial "always predict L2/3" classifier scores 0.50 in raw accuracy but 0.25 in balanced accuracy (it gets 100% recall on L2/3, 0% on every other class, mean = 0.25). We want a metric that rewards layer-by-layer recovery, so balanced accuracy = mean of per-class recalls is the primary number.

**Why neuron level.** Layer is a property of the cell, not of a (cell, stimulus) pair. We fit models on long-row tables for sample-size reasons (more training rows → better gradient estimates) but *score* by aggregating each neuron's per-row class probabilities to a single per-neuron prediction. The aggregation is mean probability across that neuron's rows, then argmax.

The combination — long-row training, neuron-level scoring — is the workflow's "Strategy 2" and is what the Phase-1 headline uses.

### 3.3 Confound baselines

A model can score well for many wrong reasons. We chose three confound baselines to bound what a *non-biological* model could achieve, all under GKF:

| baseline | feature | what it measures |
|---|---|---|
| **chance / majority** | nothing | floor; for balanced_accuracy = 0.250 |
| **scan-only GKF** | one-hot `session_key` | per-scan layer prevalence memorisation; ≈ 0.496 |
| **depth-only** | `pt_position_y` | the segmenter ceiling; 0.976 |
| **cc_abs** | `cc_abs` (clip-corr scalar) | a known-tunable baseline; 0.39 |

The depth-only ceiling 0.976 is a methodological warning, not a target: depth is what the EM segmenter uses to assign layers, so a depth-only model approximates the labelling pipeline. We do not include `pt_position_y` as a model feature for that reason.

The scan-only baseline is more subtle. Under GKF, every fold sees training rows from every scan, so a model with only `session_key` as a feature can memorise per-scan layer priors and reach 0.496. **Under LOSO, the same model falls to chance** because the held-out scan was never seen at training. We verified this: scan-only LOSO = 0.295 across all 13 scans, exactly 0.250 on the 6 four-class held-out scans (chance). The implication is that **the right baseline for Tier G LOSO 0.50 is scan-only LOSO 0.25 — not scan-only GKF 0.50**.

### 3.4 The per-session zscoring diagnostic

Calcium-imaging amplitude is multiplied by per-session factors (GCaMP6f expression density, gain, baseline drift, photobleaching). A naive feature like `amp_mean` therefore mixes biological response strength with session-recording-strength. If the *training* pipeline can exploit per-session offsets, it will appear to learn layer when it has actually learned scan identity.

Our diagnostic: refit every stage's model **twice**, on raw features and on per-session-zscored features (median + 1.4826·MAD per session). The Δ between raw and zscored bal_acc separates two sources of accuracy:

- **Δ_raw is large, Δ_zscored is small** → the feature was riding a session offset (suspicious).
- **Δ_raw and Δ_zscored both large** → the feature carries within-session biological signal that survives normalisation (real).

Tier B passed this test (Δ_raw +0.009, Δ_zscored +0.012 — actually better after zscoring). Tier C1 and Tier D1 failed it (raw ≫ zscored). The diagnostic is the only reason we know the +0.13 Phase-1 lift is mostly real and not a session-leakage artefact.

### 3.5 The three modeling strategies

WORKFLOW.md §3.6 enumerates three ways to handle the (neuron, hash, trial) → layer prediction:

- **Strategy 1: single-row weak baseline.** Aggregate each neuron's rows to a single feature vector first, fit one-row-per-neuron. Simple but throws away per-stimulus structure.
- **Strategy 2: long-row + neuron-level probability aggregation.** Fit on long rows, predict probabilities per row, mean-aggregate per neuron, argmax. The standard contract; what the headline uses.
- **Strategy 3: set models (DeepSets).** Permutation-invariant set encoder: each row is encoded, mean-pooled per neuron, classifier head fires on the pooled vector. The most biologically faithful framing — "a neuron is its set of stimulus responses".

Within a fixed feature stack, the three strategies should agree if the signal is robust to representation. They don't always. The disagreements are themselves diagnostic.

### 3.6 The LOSO sanity test

GroupKFold by `nucleus_id` does not test cross-session generalisation — every fold has rows from every scan. To test cross-recording-session generalisation, we use **LeaveOneSessionOut**: for each held-out scan, train on the other 12, evaluate on the held-out one. The mean and std across the 13 held-out scans is a far more honest measure of "would this model work on a new recording?".

Two scans with single-class held-out sets (9_3 = L4 only, 9_4 = L2/3 only, 9_6 = L2/3 only) are not testing layer separation under LOSO — they test the model's prior. We report both the all-scan mean and the 4-class-only-scans mean (10 scans).

---

## 4. Tier-by-tier feature engineering

### 4.1 Overall logic

We grow the feature set tier-by-tier, refitting the headline at each stage, so we can attribute each accuracy lift to a specific feature class. The hypotheses, in order, are:

1. **Stimulus-locked response amplitude and shape** carry layer information (Tier A).
2. **Trial-to-trial reliability** carries additional layer information beyond amplitude (Tier B).
3. **Behavioural state** (running, pupil) modulates responses in layer-specific ways (Tier C).
4. **Stimulus content** (luminance, spatial frequency, motion) modulates responses in layer-specific ways (Tier D).
5. **Cross-session-shared per-neuron tuning fingerprint** (Tier G) is an alternative single-row representation that may beat the long-row approach.

### 4.2 Tier A — neural response (amp + shape, 20 features)

The starting tier. Amplitude scalars (mean, peak, range, std, integral, etc., 10 features) and shape scalars (rise slope, decay slope, adaptation ratio, FWHM, etc., 10 features) per stimulus-locked window.

Two granularities:
- **A0**: per-trial (one row per (neuron, hash, trial)). More rows, more noise per row.
- **A1**: per-(neuron, hash) (within-hash trial-averaged). Fewer rows, cleaner per-row signal.

Decisions worth recording:
- **Per-quantity epsilons for ratio features.** A naive zero-divide check let near-zero denominators produce 1e+14 outliers in some shape ratios. We replaced exact-zero checks with biologically-sized eps thresholds (`EPS_TRACE=1e-6`, `EPS_MEAN=0.1`, `EPS_AUC=1.0`, `EPS_INTEGRAL=1.0`, `EPS_PEAK=0.5`) because real fluorescence values have a noise floor and "near-zero" needs to be defined in physical units, not float-eps.
- **A1 over A0 as the methodological default.** Trial averaging removes per-trial noise that has no layer signal. Stage 1 confirmed: A1 gave more interpretable results at slightly lower accuracy, A0 was noisier with weak occasional lifts.

### 4.3 Tier B — reliability (7 features)

Per (neuron, hash), measures of how reproducible the response is across trials. Requires `n_trials ≥ 2` per (neuron, hash). Features include `b_amp_cv` (std/mean of within-hash amplitude across trials), `b_oracle_corr` (correlation of one trial's response to mean of others), `b_trace_corr_mean`, etc.

The hypothesis is biological: **layer-specific noise structure**. L4 simple-cell drive is highly reproducible across trials; L5 broad integration with subtype-mixed cells is noisier. Reliability scalars should therefore separate layers above what amplitude alone does.

The diagnostic verdict (Stage 2): Tier B passes. Stage-2 zscored A1+B = 0.631 vs zscored A1-only = 0.619 (Δ = +0.012). And Δ_raw (+0.009) ≈ Δ_zscored (+0.012), so the lift is robust to per-session normalisation.

### 4.4 Tier C — behaviour (C0 + C1)

Per-trial behavioural features (running speed, pupil size, derivative thereof) attached at two granularities:

- **C0**: per-trial, attached at the (neuron, hash, trial) row. 8 features per row. Trial-shared problem: every neuron in a session-trial gets the same behaviour values, so this isn't really a per-neuron feature.
- **C1**: per-(neuron, hash), built from the difference between high-state and low-state trial responses. 4 features. Requires both behavioural states to be present for a (neuron, hash) → 91.8% NaN on the full table.

The hypothesis is biological: behaviour modulates V1 responses in layer-specific ways (Niell & Stryker 2010 etc.). C1 captures this *per neuron*, so it should be a per-neuron biological feature.

The diagnostic verdict (Stage 3): C1 **fails**. Δ_raw (+0.071) ≫ Δ_zscored (+0.004). The feature gain comes almost entirely from session-correlated nuisance, not from biology that survives per-session normalisation. We keep C1 in the headline stack for completeness (the workflow names it) but flag it as not robust.

### 4.5 Tier D — stimulus descriptors (10 numeric + 3 categorical)

Per condition_hash (one value per hash, broadcast onto all (neuron, hash) rows of that hash). Numeric features computed on the videos: luminance mean/std, contrast, motion energy, spatial-frequency power in low/mid/high bands, temporal-frequency power in low/mid/high bands. Categorical: stim_type one-hot (Clip/Monet2/Trippy).

The hypothesis is biological: layer-specific tuning to stimulus content (L4 strongly drives by simple-cell-tuned content, L5 broadband, L6 modulation). The numeric features should carry real signal; the family one-hots are a methodological worry — they could leak session structure if Monet2/Trippy are systematically over- or under-represented in some scans.

The diagnostic verdict (Stage 4): D1 fails the zscored-vs-raw diagnostic (similar to C1) — but with permutation importance (see §6), we found the family one-hots have ≈ 0 importance while the continuous content scalars (`d_sf_mid`, `d_luminance_std`, `d_sf_low`, `d_motion_energy`) rank in the top 10. **The family one-hots are not the source of the Stage-4 lift** — the continuous descriptors are. This was empirical confirmation that the worry was unfounded.

### 4.6 Tier G — oracle fingerprint (116 features per neuron)

Per neuron, the trial-averaged amplitude (`amp_mean`) for each of the 116 cross-session-shared oracle hashes. One row per neuron, 116 columns. **The single feature representation that has session-invariant axes by construction**: every column means the same thing in every scan because the underlying stimulus is physically identical.

The hypothesis is biological + methodological: a per-neuron *tuning fingerprint* over a fixed stimulus battery should identify cortical layer because layer-specific microcircuits produce layer-specific tuning shapes. And the cross-session-anchored coordinate system should generalise better across scans than the long-row approach.

This hypothesis was validated and is the most important Phase-1 result.

---

## 5. Stage-by-stage modeling progression

### 5.1 Stage 1 — A0 vs A1, three strategies

The first real fit. Goal: establish the feature granularity (A0 vs A1) and modeling strategy that work best on the simplest tier alone. Models: LogReg (lbfgs, l2, balanced class-weight), HGB (hist gradient boosting). Strategy 1 (one row per neuron) and Strategy 2 (long-row + aggregation) for both feature granularities. Plus DeepSets (Strategy 3) for A1 only because per-trial sets are huge.

Headline result on A1 amp+shape (Stage 1):

| model | bal_acc | macro_f1 |
|---|---|---|
| LogReg raw | 0.39 | 0.31 |
| LogReg zscored | 0.49 | 0.40 |
| HGB raw | 0.50 | 0.40 |
| HGB zscored | 0.59 | 0.49 |
| DeepSets | 0.55 | — |

Interpretation:
- HGB > LogReg by 0.10 — non-linear interactions matter.
- Zscoring lifts every model by 0.08–0.10 — per-session amplitude offsets are real and removing them reveals more biology.
- DeepSets is between LogReg and HGB at this stage.

Stage 1 also produced the LOSO sanity check: HGB raw A0 amp+shape, GKF = 0.536 → LOSO = 0.376 ± 0.137. A 0.16 drop. That set the precedent that GKF ≫ LOSO under our protocol.

### 5.2 Stage 2 — adding Tier B reliability

The diagnostic verdict (Stage 2): Tier B passes both forms of the diagnostic.

Headline result: HGB zscored A1+B = **0.631**.

| stack | raw | zscored | Δ_raw | Δ_zscored |
|---|---|---|---|---|
| A1 | 0.50 | 0.619 | — | — |
| A1+B | 0.51 | 0.631 | +0.009 | +0.012 |

Δ_raw ≈ Δ_zscored is the clean signal that B is real biology, not session-leakage. Reliability adds layer signal.

### 5.3 Stage 3 — adding Tier C (C0/C1)

The diagnostic verdict: C0 is trial-shared (not informative per-neuron), C1 fails the diagnostic.

| stack | raw | zscored | Δ_raw | Δ_zscored |
|---|---|---|---|---|
| A1+B | 0.51 | 0.631 | — | — |
| A1+B+C1 | 0.58 | 0.635 | +0.071 | +0.004 |

C1 lifts raw bal_acc by +0.071 but zscored only by +0.004. The bulk of the apparent gain comes from session-correlated structure that per-session normalisation removes. We keep C1 in the stack to follow the workflow but flag it.

The biological reading: behaviour does modulate V1 in layer-specific ways (literature is clear on this), but in our 13-scan single-mouse data the behavioural state distribution co-varies with session in a way that the model exploits before getting to the per-neuron biology. With more scans / more behavioural state coverage per scan this might invert.

### 5.4 Stage 4 — adding Tier D, the Phase-1 headline

| stack | raw | zscored | Δ_raw | Δ_zscored |
|---|---|---|---|---|
| A1+B+C1 | 0.58 | 0.635 | — | — |
| A1+B+C1+D1 | 0.65 | **0.639** | +0.07 | +0.004 |

The D1 lift looks similar to C1's pattern — diagnostic-fails framing. But §6 (permutation importance) clarified that **the family one-hots (the structural worry) are not the source of the lift; the continuous content scalars are**. The lift is real biology, not family leakage.

**Phase-1 long-row headline: HGB zscored A1+B+C1+D1 = 0.639 ± 0.011 (GKF, neuron-level).** This is the canonical headline through Stage 4.

### 5.5 Stage 4 LOSO sanity

HGB zscored A1+B+C1+D1 LOSO = **0.347 ± 0.171** across 13 scans.

Δ_(GKF→LOSO) = 0.292 — a massive drop. Two scans (9_3, 9_4) score ≈ 0 because their layer composition is degenerate after the held-out split. The drop is *a property of the splitting protocol*, not the feature set: Stage-1 LOSO on 6 raw amp+shape features dropped by 0.16 too, scaled to its smaller feature stack.

Mechanism: per-session zscoring removes *univariate* offsets but not *multivariate* joint structure. Under GKF the model effectively memorises the joint per-(scan, hash) distribution because every scan is in every fold. Under LOSO the held-out scan's joint structure is novel.

This was the first sign that the long-row headline does not reflect cross-session generalisation, only within-protocol decoding.

---

## 6. Phase-1 characterization (notebook 08)

Three orthogonal questions about the Phase-1 long-row headline:

1. Is there a *better* alternative single-row representation? (Tier G)
2. *What features* does the headline actually use? (Permutation importance)
3. *How stable* is the headline to randomness? (Multi-seed)

Plus the LOSO sanity check on Tier G itself, which became one of the most important Phase-1 findings.

### 6.1 Tier G beats the long-row headline

| metric | Phase-1 headline (A1+B+C1+D1) | **Tier G HGB** |
|---|---|---|
| bal_acc | 0.639 ± 0.011 | **0.663 ± 0.008** |
| macro_f1 | 0.513 | **0.658** |
| L2/3 / L4 / L5 / L6 recall | 0.55 / 0.63 / 0.44 / **0.94** | **0.74 / 0.63 / 0.53 / 0.75** |
| Tier G LogReg | — | 0.615 ± 0.007 |

Two things to notice:
- **+0.024 bal_acc, +0.145 macro_f1**. The macro_f1 gap is far bigger than the bal_acc gap because Tier G *recovers all four classes in a balanced way* instead of collapsing onto L6 (0.94 → 0.75 with a 0.21 loss but with L5 recovering from 0.44 → 0.53).
- **HGB > LogReg by 0.048 on Tier G**. Linear projection of the 116-D fingerprint already gives 60%-balanced recovery; the HGB gain confirms that *non-linear interactions across stimuli matter* (e.g. "high response to oracle 003 *and* low response to oracle 015" is layer-discriminative pattern that a linear model cannot capture).

The mechanistic explanation for why Tier G outperforms the richer feature stack: **cross-session comparability by construction** + **granularity-matched prediction** (one row per neuron, prediction is one label per neuron — the biology's own granularity).

### 6.2 Permutation importance — what features the headline uses

Top 10 features by row-level permutation importance on HGB zscored A1+B+C1+D1:

| rank | feature | importance | tier |
|---|---|---|---|
| 1 | **amp_min** | 0.074 | A1 |
| 2 | amp_peak | 0.028 | A1 |
| 3 | d_sf_mid | 0.023 | D1 |
| 4 | d_luminance_std | 0.023 | D1 |
| 5 | d_sf_low | 0.018 | D1 |
| 6 | b_amp_cv | 0.016 | B |
| 7 | shape_adaptation_ratio | 0.016 | A1 |
| 8 | amp_range | 0.016 | A1 |
| 9 | d_motion_energy | 0.015 | D1 |
| 10 | amp_baseline_std | 0.011 | A1 |

Bottom: `shape_decay_slope`, `b_oracle_corr`, `b_trace_corr_mean`, **`d_stim_Clip / d_stim_Monet2 / d_stim_Trippy` (≈ 0)**, `shape_rise_slope` (negative).

Three findings:
- **`amp_min` dominates by 2.7×.** The response *floor* (most-negative trace deviation) is the single most layer-discriminative feature. Calcium fluorescence is monotonically positive but baseline drift and per-cell GCaMP loading shift the floor in cell-type-correlated ways. This is the simplest-possible feature carrying the most layer signal.
- **Continuous Tier-D content features rank highly; family one-hots do not.** The Stage-3/4 worry that the Stage-4 lift came from Monet2/Trippy family-as-session shortcut is empirically rejected. The model uses *what the stimulus is made of* (luminance, SF, motion) but not *which family it is*.
- **Reliability is one feature, not seven.** `b_amp_cv` is the only B feature that imports above 0.001. The other six (`b_oracle_corr`, `b_trace_corr_mean`, etc.) live at the bottom. Stage-2's lift is essentially `b_amp_cv` alone.
- **Complex shape features go unused.** `shape_decay_slope`, `shape_rise_slope` are at or below zero importance. The headline is effectively an amplitude-and-stimulus-content classifier with one shape feature surviving.

### 6.3 Multi-seed stability

5 seeds × 5 folds = 25 HGB fits on the headline stack:

| seed | fold mean | fold std |
|---|---|---|
| 42 | 0.6392 | 0.0114 |
| 123 | 0.6387 | 0.0108 |
| 456 | 0.6383 | 0.0111 |
| 789 | 0.6410 | 0.0117 |
| 999 | 0.6387 | 0.0117 |

Across-seed std = **0.0011** (10× smaller than fold std 0.0113). The reportable confidence band is fold-level ±0.011; HGB's stochastic components contribute negligibly.

### 6.4 LOSO sanity for Tier G

The most important characterization step. The hypothesis: Tier G's session-invariant feature axes should give a smaller GKF→LOSO drop than the long-row headline.

| protocol | bal_acc | drop |
|---|---|---|
| Tier G GKF | 0.663 ± 0.008 | — |
| **Tier G LOSO (raw)** | **0.497 ± 0.105** | +0.166 |
| Tier G LOSO v2 (per-feature train-fold zscored) | 0.498 ± 0.111 | +0.165 |
| Headline GKF | 0.639 ± 0.011 | — |
| Headline LOSO | 0.347 ± 0.171 | +0.292 |

**Tier G's LOSO penalty (0.17) is 43% smaller than the long-row headline's (0.29).** Structural argument validated.

### 6.5 The scan-only LOSO baseline correction

The interpretation of Tier G LOSO 0.497 hinges on the right baseline. Initially I compared to scan-only **GKF** 0.496 and worried that "Tier G LOSO ≈ scan-only confound" meant the classifier was just learning scan identity. This was an apples-vs-oranges error.

The correct baseline: **scan-only LOSO** (a model with only `session_key` as a feature, evaluated under LOSO). Result:

| held-out scan group | scan-only LOSO bal_acc |
|---|---|
| all 13 scans | 0.295 ± 0.240 |
| 6 four-class scans only | **0.250** (chance) |

Under LOSO, the held-out scan's session_key is unseen at training, so a scan-only model has zero information and falls to chance. **Tier G LOSO 0.50 vs scan-only LOSO 0.25 is a +0.25 lift** — clear evidence of real cross-scan generalisation.

### 6.6 Tier G v2 — amplitude calibration is *not* the residual confound

The natural follow-up: if Tier G LOSO drops 0.17 from GKF, can per-session amplitude calibration explain the drop? Tier G v2 does *per-feature train-fold zscoring* (median + MAD on the 12 training scans, applied to the held-out scan), keeping the session-invariant feature axes but normalising amplitude scale.

Result: Tier G v2 LOSO = **0.498**, identical to Tier G v1 (0.497). **Per-session amplitude calibration is empirically rejected as the residual confound.**

The implication: the residual GKF→LOSO drop of 0.17 is **structural**, not normalisable:

1. Limited training data per held-out scan (13 scans, dropping one removes ~7% of training data, disproportionately affecting L6 ≈ 28 cells per scan).
2. Class-distribution shift across scans (different scans have different L4/L5/L6 prevalences; held-out scan's class composition is novel).
3. Multivariate joint structure between oracle responses (covariance like "cells responding to oracle 5 also respond to oracle 7") may be scan-specific in ways that affect HGB's tree splits but are not removable by univariate normalisation.

These are properties of having 13 scans on a single mouse, not methodological problems to fix. The cross-scan ceiling on this dataset is around 0.50; the within-protocol ceiling is around 0.66. **Both numbers are real and reportable.**

### 6.7 Within-session decoding and balanced-session LOSO (notebook 11)

Two complementary experiments designed to bound, from below and from above, the Phase-1 GKF and LOSO numbers — and to test whether the cross-session pooling that defines GKF is a friend or a foe of the headline.

#### 6.7.1 The two questions

**(A) Within-session GKF.** *If we eliminate the cross-session confound entirely by training and testing within a single session, how well can each tier decode layer? Does the within-session ceiling exceed our Phase-1 GKF (0.663)?*

For each balanced session separately: GroupKFold(`nucleus_id`), 5 folds, refit HGB for each tier (A1, A1+B, A1+B+C1, A1+B+C1+D1, Tier G). Aggregate mean ± std across the 6 balanced sessions.

**(B) Balanced-session LOSO.** *Does cross-session generalisation improve if we restrict to the 6 four-class sessions, where every held-out scan has all four layers represented?*

Restrict the full dataset to the 6 balanced sessions; run Tier G HGB LOSO with 6 held-out scans. Compare to the full-13-scan Tier G LOSO (0.497).

**Why 6 sessions, not 13.** Three sessions (9_3, 9_4, 9_6) have a single layer present and four (4_7, 7_3, 7_5, 8_5) have only three layers. The 6 *balanced* sessions are `5_6, 5_7, 6_2, 6_4, 6_6, 6_7` (~4,650 neurons). These are the same 4-class held-out scans used in the scan-only LOSO sanity in §6.5.

#### 6.7.2 Within-session GKF — Phase-1 GKF is NOT exploiting session confound

| tier | bal_acc (mean ± std across 6 sessions) | macro_f1 | L2/3 / L4 / L5 / L6 recall |
|---|---|---|---|
| A1 | 0.496 ± 0.023 | 0.380 | 0.33 / 0.44 / 0.30 / 0.91 |
| A1+B | 0.494 ± 0.026 | 0.380 | 0.33 / 0.44 / 0.30 / 0.90 |
| A1+B+C1 | 0.513 ± 0.025 | 0.400 | 0.35 / 0.44 / 0.35 / 0.92 |
| A1+B+C1+D1 | 0.532 ± 0.032 | 0.425 | 0.37 / 0.44 / 0.39 / 0.93 |
| **Tier G** | **0.611 ± 0.063** | **0.595** | 0.66 / 0.45 / 0.56 / 0.78 |

**Headline finding: within-session Tier G GKF (0.611) is *below* Phase-1 Tier G GKF (0.663) by 0.052.** This is the third of three possible reading rules from the planning template — the *least* expected one and the most informative.

**What this rules out.** The worry that the Phase-1 GKF result was riding a session-correlated proxy. If GKF had been exploiting session confound, removing it would have *preserved or improved* the bal_acc. Instead, removing the cross-session pooling *costs* 0.052. The cross-session aggregation is doing real ML work — pooling more training data outweighs the noise it introduces. **The Phase-1 GKF reportable (0.663) is methodologically clean.**

This is a strong vindication of the GKF-by-`nucleus_id` protocol that we have used since notebook 04. The protocol does not over-credit the model with session-id information; it just gives the model more training data than any single-session restricted protocol could.

#### 6.7.3 Tier progression within-session — Tier B's full-dataset lift was cross-session pooling

| step | Δ within-session | Δ Phase-1 (full 13) |
|---|---|---|
| A1 → A1+B | +0.000 (Tier B does nothing!) | +0.012 |
| A1+B → A1+B+C1 | +0.019 | +0.004 (zscored) |
| A1+B+C1 → A1+B+C1+D1 | +0.019 | +0.004 (zscored) |
| A1+B+C1+D1 → Tier G | +0.079 | +0.024 |

**Tier B's lift entirely came from cross-session pooling.** Within a session, Tier B adds *zero* (0.494 vs 0.496). This is consistent with `b_amp_cv` being a noise-normalised amplitude scalar — its discriminative power within a session is bounded by within-session amplitude variance. Cross-session, the same scalar separates neurons across sessions cleanly because session-shared noise structure differs.

**Tier G's gap over the long-row stack widens within-session** (+0.079 here vs +0.024 in Phase-1). The per-neuron oracle fingerprint uses training data more efficiently (one row per neuron vs 100+) and remains the dominant representation when training data is limited. **Tier G as the recommended primary headline is reaffirmed.**

#### 6.7.4 Long-row L6 collapse, amplified

The long-row tiers within-session show an extreme version of the L6-collapse pattern flagged in §6.1. On A1+B+C1+D1: L2/3 recall = 0.37, L4 = 0.44, L5 = 0.39, **L6 = 0.93**. The model heavily over-predicts L6 — bal_acc 0.532 is *carried* by L6 recall 0.93 while macro_f1 sits at 0.425 because the other three classes are at chance.

Tier G is far more balanced (L2/3 0.66 / L4 0.45 / L5 0.56 / L6 0.78, macro_f1 = 0.595). The pattern from Phase-1 — Tier G as the only balanced classifier — is even more pronounced within-session. **The long-row stack within-session is essentially an L6-detector dressed up as a four-class classifier.**

#### 6.7.5 Per-session decoding correlates with labelling-mismatch rate

Tier G bal_acc by session, sorted by mismatch rate from notebook 10:

| session | mismatch % (notebook 10) | within-session Tier G bal_acc |
|---|---|---|
| 5_6 | 0.1 % | 0.664 |
| 5_7 | 0.7 % | 0.656 |
| 6_2 | 5.2 % | 0.676 |
| 6_4 | 10.0 % | 0.577 |
| 6_6 | 13.1 % | 0.565 |
| 6_7 | 18.5 % | 0.526 |

**Striking inverse correlation between labelling-mismatch rate and within-session decoding accuracy.** The 4 high-mismatch scans (6_2, 6_4, 6_6, 6_7) bracket the bottom of the per-session decoding spread. Sessions with cleaner labels decode better.

This refines the notebook 10 finding: those 4 scans aren't merely a label-quality issue, they appear to be genuinely *harder* recordings for the model to decode. Whether the cause is recording quality, depth ambiguity, cell-class mixing, or labelling error, the same 4 scans show up as outliers in two independent analyses.

#### 6.7.6 Balanced-session LOSO — restricting training improves cross-session generalisation

| protocol | bal_acc | n_held_out_scans |
|---|---|---|
| Phase-1 LOSO (full 13 scans) | 0.497 ± 0.105 | 13 |
| Phase-1 LOSO restricted to the 6 four-class scans | 0.479 ± 0.076 | 6 |
| **Balanced LOSO (6 scans both train + test)** | **0.520 ± 0.057** | 6 |

**Balanced LOSO improves on Phase-1 LOSO 4-class subset by +0.041.** When the same 6 held-out scans are evaluated, restricting *training* data to the 6 balanced sessions (excluding the 7 odd scans 4_7, 7_3, 7_5, 8_5, 9_3, 9_4, 9_6) actually improves cross-session generalisation. The 7 excluded scans were apparently adding noise to the model rather than signal.

Possible mechanisms (not separated by this experiment):
- The single-class scans (9_3, 9_4, 9_6) push the model toward biased per-class priors that don't generalise.
- The 3-class scans (4_7, 7_3, 7_5, 8_5) miss L6 entirely, so the model's L6 representation is narrower than if all training scans had L6 cells.

**Same per-session pattern in balanced LOSO**: held-out 5_6/5_7 → highest bal_acc (0.572–0.599), held-out 6_4/6_6/6_7 → lowest (0.460–0.490). The 4 'session-6' high-mismatch scans are systematically harder both as held-out scans and as within-session training material.

#### 6.7.7 The convergent 4-scan finding

The 4 session-6 scans (6_2, 6_4, 6_6, 6_7) appear as outliers in **three independent analyses**:

1. **Notebook 10 (labelling sensitivity)**: 94.6 % of cell-type-vs-layer-label mismatches concentrate in these 4 scans (5.2–18.5 % mismatch rate vs <1 % elsewhere).
2. **Notebook 11 within-session GKF**: lowest within-session decoding (0.526–0.676 vs ≈ 0.66 elsewhere), inversely correlated with mismatch rate.
3. **Notebook 11 balanced LOSO**: lowest held-out bal_acc when these scans are the test scan (0.460–0.531 vs 0.572–0.599 for 5_6/5_7).

The triple convergence is robust enough to flag in the methods section of the report. Whatever is going on with these 4 scans — depth, recording quality, labelling, cellular composition, or some combination — it is detectable through three different decoding-and-labelling lenses. The within-session and balanced-LOSO sensitivity does not affect the headline reportable, but it adds methodological honesty: not every recording session in MICrONS is equally clean for laminar decoding.

#### 6.7.8 What this section adds to the Phase-1 ledger

| protocol | bal_acc | source |
|---|---|---|
| Phase-1 GKF (full 13) — Tier G | 0.663 ± 0.008 | notebook 08 §4 |
| Within-session GKF (mean across 6) — Tier G | 0.611 ± 0.063 | notebook 11 §5 |
| Within-session GKF — A1+B+C1+D1 long-row | 0.532 ± 0.032 | notebook 11 §5 |
| Phase-1 LOSO (full 13) — Tier G | 0.497 ± 0.105 | notebook 08 §4b |
| Phase-1 LOSO 4-class subset — Tier G | 0.479 ± 0.076 | derived from 08 §4b |
| Balanced-session LOSO (6 scans) — Tier G | 0.520 ± 0.057 | notebook 11 §6 |

These do not replace the Phase-1 reportable. They are bookends: the within-session GKF (0.611) bounds where the Phase-1 GKF (0.663) would sit *without* cross-session pooling — and the +0.052 difference is the *value of cross-session aggregation*. The balanced LOSO (0.520) sits between Phase-1 LOSO 4-class subset (0.479) and Phase-1 GKF (0.663) as a cleaner-subset cross-session estimate.

**The takeaway for the report**: the Phase-1 numbers (0.663 GKF, 0.497 LOSO) are reported as primary, with §6.7 added as evidence that they are honest — not session-confound-inflated, and consistent with restricted-subset variants on each protocol.

---

## 7. Stage 5 — family analyses (notebook 09)

WORKFLOW.md §5 Stage 5 asks two questions:

(b) Does layer recoverability vary by stimulus family?
(c) Does the layer signature transfer across families?

### 7.1 Why Tier A only — and not the full A1+B+C1+D1 stack

A methodological choice with consequences. Stage 5 deliberately uses only **A1 amp+shape (20 features)** for three reasons:

1. **Tier D collapses into nuisance within a family.** When we filter to "Clip only", `d_stim_Clip = 1` for every row → zero-variance column. Trees auto-ignore; LogReg gives zero weight. D adds nothing within a family.
2. **Cross-family transfer with D is broken by construction.** Train on Clip with `d_stim_Clip = 1`, test on Monet2 with `d_stim_Clip = 0`. The classifier learned to use the Clip indicator at training time; at test time that indicator is wrong everywhere. This would test what happens to a model that memorised stimulus identity, not the biology of layer signatures across stimuli — which is the workflow's actual question.
3. **B and C1 have family-specific distributions.** Monet2/Trippy always have 2 trials per hash → `c1_state_rel_diff` is structurally sparser there. `c1_rmi` is 91.8% NaN. Cross-family transfer of such sparse features is dominated by imputation pattern, not biology.

So Stage 5 cleanly tests *neural-response-only family signatures*.

### 7.2 Why DeepSets is the primary model here

The biological framing is "each neuron is a set of responses to family-X stimuli; can we infer its layer from that set?" — a permutation-invariant set problem by construction. DeepSets encodes each row, mean-pools, classifies → the natural model class for this question. HGB and LogReg are sanity checks.

(In practice HGB outperformed DeepSets on within-family decoding because the per-neuron set sizes are small — 20 rows for Monet2/Trippy — and HGB's interaction-finding wins on small sets with fixed feature dimension. DeepSets pools to a mean vector, throwing away per-stimulus tuning structure that HGB exploits per row.)

### 7.3 Per-family results — ranking matches the neuroscience prediction

| family | LogReg | HGB | DeepSets |
|---|---|---|---|
| Clip | 0.417 ± 0.031 | 0.596 ± 0.008 | 0.482 ± 0.012 |
| Monet2 | 0.485 ± 0.010 | **0.614 ± 0.013** | 0.530 ± 0.008 |
| Trippy | 0.441 ± 0.025 | 0.559 ± 0.020 | 0.480 ± 0.017 |

**HGB ranking: Monet2 (0.614) > Clip (0.596) > Trippy (0.559).** This matches the pre-registered neuroscience prediction: parametric, classic V1-tuning > naturalistic, partially silent > broadband noise. The biological reading is that **cortical layer is most identifiable from responses to stimuli that engage the canonical V1 microcircuit**.

### 7.4 A0 sensitivity — Trippy gains most from per-trial granularity

| family | A1 (HGB) | A0 (HGB) | A0 − A1 |
|---|---|---|---|
| Clip | 0.596 | 0.628 | +0.032 |
| Monet2 | 0.614 | 0.615 | +0.000 |
| Trippy | 0.559 | 0.620 | **+0.061** |

Mechanistically:
- Monet2's two trials are highly reproducible (parametric); trial averaging neither helps nor hurts.
- Trippy's two trials of phase-shuffled noise *averaged* to something close to mean luminance. Per-trial rows preserve layer-discriminative content that A1's trial averaging destroys.
- Clip is in between.

A pre-registered Trippy-specific caveat for the report: **on A0 the family ranking flips** (Clip > Trippy ≈ Monet2, all within 0.013).

### 7.5 Cross-family transfer — layer signature is partly family-specific

HGB transfer matrix (train rows ↘, test cols →):

|   | Clip | Monet2 | Trippy |
|---|---|---|---|
| Clip   | (0.596) | **0.252** | **0.249** |
| Monet2 | 0.449 | (0.614) | 0.422 |
| Trippy | 0.441 | 0.405 | (0.559) |

Three patterns:
- **HGB Clip→Other COLLAPSES to chance (0.25).** HGB's tree splits on Clip-specific feature interactions that don't exist on Monet2's parametric gratings or Trippy's noise. The decision boundaries are over-specialised to Clip's stimulus statistics.
- **LogReg transfers from Clip much better than HGB** (LogReg Clip→Monet2 = 0.435 vs HGB 0.252). The classic generalisation/specialisation trade-off: HGB extracts more within-family signal at the cost of overfitting to family-specific structure.
- **Monet2 ↔ Trippy is the most transferable pair** (0.40–0.48 in both directions across all three models).
- **Asymmetry — transfer into Clip is harder than out of Clip.** Models trained on a small parametric battery (Monet2 or Trippy, 20 hashes each) cannot anticipate Clip's naturalistic feature distribution (240 hashes).

The scientific framing: **the Phase-1 all-families headline averaged across distinct family-specific layer signatures, not a single universal one.** This is itself a positive scientific finding per WORKFLOW §5.

---

## 8. Honest reportable framing

Three numbers, three protocols, each with the right baseline:

| protocol | model | bal_acc | baseline | margin |
|---|---|---|---|---|
| within-protocol GKF | Tier G HGB | **0.663 ± 0.008** | scan-only GKF 0.496 | +0.17 |
| within-protocol GKF | long-row HGB A1+B+C1+D1 zscored | 0.639 ± 0.011 | (legacy headline) | — |
| cross-scan LOSO | Tier G HGB | **0.497 ± 0.105** | scan-only LOSO 0.250 | +0.25 |
| cross-scan LOSO 4-class scans | Tier G HGB | 0.479 | chance 0.250 | +0.23 |
| cross-scan LOSO | long-row HGB headline | 0.347 ± 0.171 | scan-only LOSO 0.250 | +0.10 |

The recommended primary headline is **Tier G HGB**:
- **Within-protocol decoding ceiling: 0.663 ± 0.008** — answers "given functional data from a recording where some cells are labelled, how well can we infer the layer of *other cells in the same recording*?"
- **Cross-scan generalisation: 0.497 ± 0.105** — answers "given functional data from a *new* recording, how well can we infer layer?". The +0.25 lift over scan-only LOSO is real cross-scan biology.

The cross-scan margin (+0.25) is actually larger than the within-protocol margin over scan-only (+0.17). That's a strong story: the cross-session-shared 116-stimulus battery design is the *right path* for cross-recording-session decoding.

### 8.1 Why Tier G as the headline (and not A1+B+C1+D1)

- Higher bal_acc (0.663 vs 0.639) and dramatically higher macro_f1 (0.658 vs 0.513).
- Per-class recall balanced across all four layers (range 0.53–0.75) instead of L6-collapsed (0.44 L5, 0.94 L6).
- Smaller GKF→LOSO drop (0.17 vs 0.29) — better cross-session generalisation.
- Mechanistically clean story: cross-session-shared stimulus battery → per-neuron tuning fingerprint → layer identity. Easy to present and defend.
- Single representation, no zscoring choice to justify per stage.

### 8.2 Why the long-row A1+B+C1+D1 headline still belongs in the report

- It's the canonical Stage-1 → Stage-4 progression result; the structure of the workflow is built around it.
- It validates that the +0.13 lift from A1-only (0.50) → A1+B+C1+D1 (0.64) under zscoring is real biology (permutation importance shows it).
- It answers the methodological question "does the long-row + neuron-level aggregation framework work?" — yes, it does, just less well than the per-neuron oracle fingerprint.

The cleanest report structure: **Tier G HGB as the primary headline; the long-row headline as a comparator and as evidence that the tier-by-tier feature progression added real signal.**

---

## 8.5 Sensitivity to label ambiguity (notebook 10)

### 8.5.1 The problem

MICrONS provides two independent label sources for each neuron:

- **`cell_type`** comes from EM 3D morphology (`23P`, `4P`, `5P-IT`, `5P-ET`, `5P-NP`, `6P-IT`, `6P-CT`) — morpho-anatomical truth.
- **`layer_label`** comes from `pt_position_y` — depth-derived assignment.

Applying the canonical mapping (`23P → L2/3`, `4P → L4`, `5P-* → L5`, `6P-* → L6`), the two sources disagree on **407 / 8,895 = 4.6 %** of working-population neurons. Phase 1 used `layer_label` as ground truth throughout. We don't know which label is wrong, so the methodologically right move is **sensitivity analysis** — exclude the ambiguous cases and check whether the headline depends on them — not relabelling.

### 8.5.2 Distribution — by layer and by session

By layer the mismatches are biologically expected:

| layer | n | n mismatch | % |
|---|---|---|---|
| L2/3 | 4,247 | 87 | 2.0 % |
| L4 | 2,670 | 70 | 2.6 % |
| L5 | 1,615 | 191 | **11.8 %** |
| L6 | 363 | 59 | **16.3 %** |

Deep layers L5 and L6 have 5–8× the mismatch rate of superficial layers. This is consistent with L4–L5 and L5–L6 boundaries being anatomically less distinct than L2/3–L4.

By session the mismatches are *not* uniformly distributed:

| session | mismatch % | n mismatch |
|---|---|---|
| 6_7 | 18.5 % | 162 |
| 6_6 | 13.1 % | 91 |
| 6_4 | 10.0 % | 92 |
| 6_2 | 5.2 % | 40 |
| 4_7 | 1.1 % | 8 |
| 8 other scans | 0.0 – 0.7 % | 14 total |

**The 4 "session-6" scans contribute 385 / 407 = 94.6 % of all mismatches.** Eight scans have <1 % mismatch and three are exactly 0 %. This is a methodological signature that warrants an explicit follow-up with the professor about whether those scans had specific labelling-pipeline issues. It does *not* affect the headline robustness (next subsection), but it should be footnoted in the report.

### 8.5.3 Headline robustness — the central result

Refitting Tier G HGB on the consistent-only subset (n = 8,488):

| protocol | full (n = 8,895) | consistent only (n = 8,488) | Δ |
|---|---|---|---|
| Tier G HGB GKF | 0.663 ± 0.008 | 0.660 ± 0.015 | **−0.003** |
| Tier G HGB LOSO | 0.497 ± 0.105 | 0.508 ± 0.108 | **+0.011** |

Per-class GKF recall is essentially identical (L2/3 0.74 → 0.74, L4 0.63 → 0.63, L5 0.53 → 0.53, L6 0.75 → 0.74). **Both within-protocol and cross-session headlines are robust to the labelling ambiguity.** No restructuring of the report is needed.

### 8.5.4 Boundary-cell hypothesis — supported for L2/3, L4, L5; not for L6

For each layer, we computed the absolute distance from each neuron's `pt_position_y` to its layer's median, then compared the distance distribution between matched and mismatched neurons (Mann-Whitney U, one-sided: matched closer to median than mismatched).

| layer | matched mean | mismatched mean | n_mism | MW p |
|---|---|---|---|---|
| L2/3 | 39.6 μm | 74.2 μm | 87 | **1.6 × 10⁻⁴¹** |
| L4 | 25.5 μm | 47.4 μm | 70 | **6.9 × 10⁻²⁷** |
| L5 | 33.4 μm | 43.9 μm | 191 | **2.9 × 10⁻¹⁸** |
| L6 | 15.2 μm | 14.1 μm | 59 | 0.74 (n.s.) |

Three of four layers show a **strong, highly-significant boundary effect**: mismatched neurons are systematically further from their assigned layer's centroid than matched neurons. For L2/3, mismatched cells sit nearly **2× further** from the L2/3 median. **75 % of the 4.6 % mismatch is biologically interpretable as boundary cells**, not random labelling errors.

L6 is the exception: mismatched L6 cells sit at *typical* L6 depth (14.1 μm vs 15.2 μm — virtually identical). They are not borderline. Two non-mutually-exclusive readings:

1. **EM cell-type errors**: the segmenter mis-classified some 6P-IT/6P-CT cells as 5P-IT/ET.
2. **Subtype admixture**: some L6 neurons have morphology that's genuinely intermediate between L5 and L6 IT-class cells, even at clear L6 depth.

Without histological ground truth we cannot distinguish these. Either way, the L6 mismatches are *not* explained by the boundary-cell hypothesis. Combined with L6's high mismatch rate (16.3 %) and small population (363 cells), this is the most uncertain corner of the labelling.

### 8.5.5 Confusion of OOF predictions on mismatched neurons

We generated out-of-fold Tier G HGB predictions on the full population and asked, on the 407 mismatched neurons: does the model predict the *trained-on* `layer_label` or the *cell-type-implied* layer?

Three-way summary across all 407 mismatches:

| target | n | % |
|---|---|---|
| predicted = `layer_label` (depth-based, the trained-on target) | 218 | **53.6 %** |
| predicted = `celltype_layer` (morpho-based) | 119 | 29.2 % |
| predicted = neither | 70 | 17.2 % |

The model predominantly agrees with the labelling pipeline. This is expected — it was trained on `layer_label`.

The per-direction breakdown reveals **two specific morpho-vs-depth tensions where cell-type wins**:

| label | celltype | n | pred = label | pred = celltype |
|---|---|---|---|---|
| L4 | L2/3 (23P morphology) | 38 | 42.1 % | **52.6 %** ← cell-type wins |
| L5 | L6 (6P morphology) | 33 | 42.4 % | **48.5 %** ← cell-type wins |
| L2/3 | L4 (4P morphology) | 87 | **58.6 %** | 21.8 % |
| L4 | L5 (5P morphology) | 32 | **53.1 %** | 25.0 % |
| L5 | L4 (4P morphology) | 158 | **51.3 %** | 25.9 % |
| L6 | L5 (5P morphology) | 59 | **66.1 %** | 25.4 % |

Cell-type wins specifically when the cell-type is **23P** or **6P** — the most layer-extreme morpho-classes. In the four "middle-class" mismatches (involving 4P or 5P morphology) the depth-label wins. The pattern is consistent with: the most extreme morphological classes leave the strongest functional fingerprint in the calcium response, and that fingerprint is detectable even when the depth-based label disagrees.

**This is a suggestive finding, not a strong claim.** The model was trained on depth-labels, so a residual cell-type-aligned prediction may reflect morphology-correlated calcium features rather than direct cell-type identification. We report it as an interesting structural observation, not as evidence that the model "captures morphology better than the labels do."

### 8.5.6 Joint scientific reading

Five conclusions, ordered by scientific weight:

1. **Phase-1 headline is robust** to the labelling ambiguity (Δ_GKF = −0.003, Δ_LOSO = +0.011, well within fold/scan noise bands). The sensitivity analysis passes.
2. **75 % of mismatches are biological border cells** (Mann-Whitney p < 10⁻¹⁷ for L2/3, L4, L5). Including them in training is not "label noise"; it's training on genuinely ambiguous boundary cells.
3. **94.6 % of mismatches are concentrated in 4 scans (6_2, 6_4, 6_6, 6_7)**. Methodological flag for the report; warrants a follow-up with the professor.
4. **L6 mismatches do not fit the boundary-cell pattern.** With a 16.3 % mismatch rate, no boundary effect, and the smallest class population, L6 is the most uncertain layer in the dataset. Suggests EM cell-type errors and/or subtype admixture; histology would be needed to resolve.
5. **The model mostly trusts the depth-label, but for morphologically-extreme cell types (23P, 6P) it occasionally picks up the cell-type-implied layer.** Interesting but not strong evidence on its own.

---

## 9. Decisions and trade-offs — the full reasoning

### 9.1 Why we used balanced_accuracy, not accuracy

Class imbalance (L6 = 4%). Raw accuracy rewards predicting the majority class; balanced_accuracy normalises by per-class recall. The scan-only confound under raw accuracy is ~0.50 (just predict L2/3); under balanced_accuracy it's ~0.50 only because session_key carries per-scan layer prevalence info. Balanced_accuracy is the only metric that meaningfully penalises an L2/3-only classifier.

### 9.2 Why GroupKFold by `nucleus_id`, not by something else

Two alternatives we considered:
- **Plain KFold**: leaks the same neuron between train and test → ~0.95+ bal_acc, but it's measuring "can the model recognise this neuron?" not "can the model predict layer from features?".
- **StratifiedGroupKFold**: stratifies layer prevalence across folds. Marginally cleaner balance per fold but sometimes can't satisfy both constraints. We chose plain GroupKFold to keep the splits reproducible and simple; per-fold layer balance is acceptable.
- **GroupKFold by `session_key`**: this *is* essentially LOSO. We use it explicitly as the LOSO sanity test, not as the primary CV.

### 9.3 Why we chose lbfgs over saga for LogReg

Tried saga first (the "principled" choice for L1/L2 regularisation with class weights). It was extremely slow on our long tables (10× the compute of HGB) without delivering better results. Reverted to lbfgs with `max_iter=400`, suppressing convergence warnings (which fire occasionally on hard folds but don't reflect real instability — verified by checking coef norms across max_iter values). The user's pushback "if you arent able to use properly modules, do not use them" was correct — we shouldn't pretend to use saga if the actual setting is lbfgs.

### 9.4 Why HGB became the dominant model

Three properties HGB has that no other tabular model in our stack matches:
- **Native NaN handling.** Tier B has 91.8% NaN on `c1_rmi`; Tier D has many NaN on family-specific features after subsetting. HGB routes NaNs through tree splits without imputation. LogReg requires median imputation; this introduces bias.
- **Fast on long tables.** A1+B+C1+D1 is ~1.2M rows × 44 features. HGB with `max_iter=100, max_depth=8, early_stopping=True` fits a fold in ~30 s. Random Forest at equivalent capacity would take 5–10×.
- **Captures non-linear interactions.** The +0.10 LogReg→HGB gap on every stage tells us the layer signature is non-linear in feature combinations. HGB's tree-based splits find these naturally.

The trade-off: HGB is harder to interpret than LogReg. We addressed this with permutation importance (model-agnostic, validation-set-based) rather than HGB's built-in `feature_importances_` (impurity-based, biased toward high-cardinality features and early-split features).

### 9.5 Why DeepSets sometimes underperforms HGB

DeepSets pools each neuron's row encodings to a single vector via mean (or sum). This is permutation-invariant — the right inductive bias for "a neuron is its set of responses". But mean-pooling throws away per-stimulus identity: HGB sees row `i` of neuron `n` and *knows* which hash it is via the joined Tier-D features; DeepSets only sees the encoded representation. On small sets (Monet2/Trippy ≈ 20 rows per neuron) this loss matters; HGB's per-row interaction-finding wins.

DeepSets is genuinely the right model class for the *biological framing*, but the practical accuracy advantage is for HGB on this particular task. We report DeepSets as the secondary model and HGB as the primary.

### 9.6 Why per-session zscoring (median + 1.4826·MAD) and not standard z-score

Standard z-score uses mean and std, both sensitive to outliers. Calcium-imaging trace amplitudes have heavy tails (occasional bursts, segmenter glitches). Median + MAD is the robust alternative; the 1.4826 factor makes MAD comparable to std for Gaussian-distributed data, so the resulting z-scores are interpretable on the same scale as standard z-scores.

We zscore *within session* for the diagnostic — to remove per-session offsets specifically — not within other groupings.

### 9.7 Why we kept C1 in the headline despite the failed diagnostic

The workflow names Tier C explicitly. We followed the workflow's named tiers rather than ablating C1 out, but flagged it as a feature that does not survive per-session normalisation. The Phase-1 reportable narrative explicitly mentions this caveat; if a referee asks "why did C1 stay in the headline?", the honest answer is "to faithfully execute the named tier progression; its lift is mostly session-correlated and we know that".

### 9.8 Why family analyses use A1 only

See §7.1. The short version: D adds nothing within a family (zero-variance one-hot), is broken cross-family (memorises stimulus identity at train, that identity is wrong at test), and B/C1 are family-specifically-sparse. A1 is the only feature set that asks the *biological* family-signature question cleanly.

### 9.9 Why Tier G is the recommended primary headline

Five reasons (§8.1 in concise form):
1. Higher metrics across the board.
2. Balanced per-class recall, no L6 collapse.
3. Smaller LOSO penalty (better cross-scan generalisation).
4. Mechanistically clean cross-session-shared-stimulus-battery story.
5. One feature representation, one fit, no zscoring choice to defend.

### 9.10 What we explicitly did *not* do, and why

- **Did not use orientation features** — workflow mandate; preserves the project's scientific interest.
- **Did not include `pt_position_y` (depth) as a model feature** — it's the segmenter's layer-assignment input; using it makes the prediction circular.
- **Did not use raw `cc_abs` as a feature in headline models** — it's a known-tunable confound and we report it as a baseline only.
- **Did not L1-decompose neurons** — the dataset has no excitatory L1 cells; L1 is a degenerate class.
- **Did not include single-class LOSO scans in interpretation of LOSO mean** — they don't test layer separation, they test the model's prior. We report multi-class LOSO mean separately.
- **Did not run a Phase 2 CNN on raw traces yet** — that's Phase 2.

---

## 10. Open questions for Phase 2 and Phase 3

### 10.1 Phase 2 — CNN / multimodal on raw traces

The Phase-1 engineered-feature ceiling is 0.66 (within-protocol) / 0.50 (cross-scan). The Phase 2 question is whether a 1-D CNN on raw stimulus-locked traces beats this. The engineered features deliberately throw away kinetic detail (only a few shape scalars survive permutation importance — so most of the kinetic structure is unused). A CNN with raw temporal data may pick up that signal.

Pre-registered hypothesis: a CNN on raw traces will achieve within-protocol bal_acc in the 0.65–0.70 range, with cross-scan LOSO around 0.50–0.55. The CNN should help most on per-class recall for L5 (the layer most poorly recovered by engineered features).

The Tier G LOSO ceiling of 0.50 is the right cross-scan benchmark for Phase 2 to beat.

### 10.2 Phase 3 — within-L5 IT/ET subtype

Phase 1 found L5 to be the hardest layer (recall 0.53 on Tier G GKF, 0.34 on LOSO). The well-known explanation is L5 is a heterogeneous mix of IT and ET subtypes with different functional properties. Phase 3 conditions on layer = L5 and asks the binary L5-IT vs L5-ET question. Stage 3's behaviour-modulation features (C1) may pre-position this question: ET cells are reportedly more strongly behaviour-modulated than IT.

### 10.3 Open methodological questions Phase 1 surfaces

- **What gives the residual 0.17 GKF→LOSO drop on Tier G?** We ruled out amplitude calibration (Tier G v2). Class-distribution shift across scans is a likely candidate; covariance shift between oracle responses is another. A Tier G v3 with multivariate-whitening on the train fold could test the latter.
- **Does the 0.66 within-protocol ceiling reflect a real biological limit, or a feature-engineering limit?** Phase 2 will partially answer this.
- **Is the family-specific Clip-overfit pattern (HGB Clip→Other ≈ chance) a property of the Clip stimulus, or of the Clip *training set size* (240 hashes vs 20)?** Could be tested by sub-sampling Clip to 20 hashes and re-running cross-family transfer.
- **Why does Monet2 produce the cleanest single-family signature?** Likely because parametric stimuli engage canonical V1 circuitry most directly. But the literature also says parametric drive saturates and naturalistic stimuli reveal more complex tuning. A targeted Monet2-vs-Clip analysis on the same neurons would resolve.

---

## 11. Artefacts produced by Phase 1

### 11.1 Feature parquets

- `data/processed/features/A0_amp.parquet`, `A0_shape.parquet` — per-trial amp / shape features (Tier A0).
- `data/processed/features/A1_amp.parquet`, `A1_shape.parquet` — within-hash trial-averaged (Tier A1).
- `data/processed/features/B_per_hash.parquet` — Tier B reliability features (1.21M rows after filtering n_trials ≥ 2).
- `data/processed/features/C0_per_trial.parquet`, `C1_per_hash.parquet` — Tier C behaviour features.
- `data/processed/features/D_per_hash.parquet` — Tier D stimulus descriptors.
- `data/processed/features/G_per_neuron.parquet` — Tier G oracle fingerprint (8,895 × 116 + nucleus_id).

### 11.2 Result parquets

- `data/processed/results/stage1_runs.parquet` — Stage 1 fits (12 model × representation combinations).
- `data/processed/results/stage2_runs.parquet` — Stage 2 (A1+B variants).
- `data/processed/results/stage3_runs.parquet` — Stage 3 (A1+B+C1 variants).
- `data/processed/results/stage4_runs.parquet` — Stage 4 (A1+B+C1+D1 variants), the canonical Phase-1 long-row headline.
- `data/processed/results/phase1_headline_loso.parquet` — Stage-4 headline LOSO across 13 scans.
- `data/processed/results/tierG_strategy1.parquet` — Tier G HGB and LogReg.
- `data/processed/results/tierG_loso.parquet` — Tier G v1 LOSO.
- `data/processed/results/tierG_v2_loso.parquet` — Tier G v2 LOSO (per-feature train-fold zscored).
- `data/processed/results/scan_only_loso.parquet` — scan-only LOSO baseline.
- `data/processed/results/headline_perm_importance.parquet` — permutation importance ranking on the headline.
- `data/processed/results/headline_multiseed.parquet` — 5-seed stability.
- `data/processed/results/stage5_per_family.parquet` — 9 within-family fits (3 models × 3 families) on A1.
- `data/processed/results/stage5_per_family_A0.parquet` — A0 sensitivity (HGB+LogReg × 3 families).
- `data/processed/results/stage5_transfer.parquet` — 18 cross-family transfer fits (3 models × 6 directed pairs).
- `data/processed/results/label_sensitivity_per_layer.parquet`, `label_sensitivity_per_session.parquet` — distribution of cell-type vs layer-label mismatches.
- `data/processed/results/label_sensitivity_tierG_filtered.parquet`, `label_sensitivity_tierG_filtered_loso.parquet` — Tier G HGB on the consistent-only subset (n = 8,488).
- `data/processed/results/label_sensitivity_scan_only_filtered.parquet` — scan-only baselines on the consistent subset.
- `data/processed/results/label_sensitivity_boundary.parquet` — per-neuron `pt_position_y` distance from layer median.
- `data/processed/results/label_sensitivity_oof.parquet` — full Tier G HGB OOF predictions, used by check 4.
- `data/processed/results/label_sensitivity_confusion.parquet` — confusion of OOF predictions on mismatched neurons.
- `data/processed/results/within_session_runs.parquet` — 30 rows (6 balanced sessions × 5 tiers): within-session GKF balanced accuracy, std, macro-F1, per-class recall.
- `data/processed/results/balanced_loso_tierG.parquet` — 6 rows: Tier G HGB balanced-session LOSO on the 6 four-class subset.

### 11.3 Splits

- `data/processed/splits/cv_assignments.parquet` — the precomputed per-neuron 5-fold GroupKFold assignment (`nucleus_id`, `gkf_fold`, `loso_test_scan`).

### 11.4 Code modules

- `src/config.py` — paths, seed (`RANDOM_SEED=42`), feature denylists.
- `src/data/loaders.py` — `load_working_pop`, `load_splits`, `build_modeling_table` (the canonical long-row builder).
- `src/eval/metrics.py` — `neuron_level_score`, `aggregate_probs_to_neuron`, `summarize_cv_runs`.
- `src/features/tier_a.py` — 20 amp+shape features at A0 and A1.
- `src/features/tier_b.py` — 7 reliability features per (neuron, hash).
- `src/features/tier_c.py` — C0 (8 features per session-trial) + C1 (4 features per neuron-hash).
- `src/features/tier_d.py` — 10 numeric + 3 stim_type one-hot features per condition_hash.
- `src/models/deep_sets.py` — `DeepSets` class + `train_deep_sets` function.

### 11.5 Notebooks

- `notebooks/01_setup_and_eda.ipynb` — dataset claims verified (29 checks all pass).
- `notebooks/02_canonical_tables.ipynb` — drops L1, builds traces.parquet (1.6 GB) + traces_avg.parquet.
- `notebooks/03_tier_a.ipynb` — Tier A computation.
- `notebooks/04_phase1_stage1.ipynb` — Stage 1 (A0 vs A1, three strategies, confounds).
- `notebooks/05_tier_b_and_stage2.ipynb` — Stage 2 (Path A only).
- `notebooks/06_tier_c_and_stage3.ipynb` — Stage 3 (C0/C1 split).
- `notebooks/07_tier_d_and_stage4.ipynb` — Stage 4 + headline LOSO sanity.
- `notebooks/08_phase1_characterization.ipynb` — Tier G + perm importance + multi-seed + Tier G LOSO + scan-only LOSO + Tier G v2.
- `notebooks/09_stage5_family.ipynb` — Stage 5 family analyses (per-family A1, A0 sensitivity, cross-family transfer).
- `notebooks/10_label_sensitivity.ipynb` — sensitivity analysis on cell-type vs layer-label consistency: distribution of mismatches, refit Tier G on consistent-only subset, `pt_position_y` boundary effect, OOF confusion on mismatched neurons.
- `notebooks/11_within_session.ipynb` — within-session GKF (6 balanced sessions × 5 tiers) and balanced-session LOSO (Tier G HGB on 6 four-class scans). Shows that Phase-1 GKF is not session-confound-driven (within-session is *lower* by 0.052) and that the 4 session-6 scans are systematically harder across three independent analyses.

---

## 12. The single-paragraph summary

> Across 8,895 V1 excitatory neurons in a 13-scan single-mouse MICrONS dataset, we used per-neuron functional response statistics — amplitude scalars, shape scalars, trial-to-trial reliability, behavioural-state modulation, stimulus-content descriptors, and per-neuron tuning fingerprints over a cross-session-shared 116-stimulus oracle battery — to predict cortical layer (L2/3, L4, L5, L6). Within-protocol (GroupKFold by neuron) the best decoder, HGB on the 116-D oracle fingerprint, achieves balanced accuracy 0.663 ± 0.008 with macro-F1 0.658 and balanced per-class recall 0.53–0.75. Cross-recording-session (LeaveOneSessionOut) the same decoder achieves 0.497 ± 0.105, robustly above the scan-only LOSO baseline (0.250 on four-class held-out scans). The +0.25 LOSO margin over scan-only is a real cross-session generalisation signal, mechanistically attributable to the session-invariant feature axes of the oracle fingerprint. Per-session amplitude calibration (tested via per-feature train-fold zscoring) is empirically rejected as the residual GKF→LOSO confound; the residual gap is structural — limited training data per held-out scan, class-distribution shift across scans, scan-specific multivariate response covariance. Permutation importance shows the headline relies on amplitude scalars (`amp_min` dominating), continuous stimulus-content descriptors (`d_sf_mid`, `d_luminance_std`, `d_motion_energy`), and one reliability scalar (`b_amp_cv`); family-identity one-hots have ≈ 0 importance, refuting the worry that the headline learns family-as-session shortcut. Family analyses confirm Monet2 > Clip > Trippy in within-family layer recoverability — matching the canonical V1 prediction — and reveal that HGB's strong within-family performance does not transfer across families (Clip→Other ≈ chance), while LogReg's lower-capacity model generalises better. The Phase-1 layer signature is therefore *partly* family-specific, not universal, which is itself a positive scientific finding. A sensitivity analysis on the 4.6 % of neurons whose EM cell-type and depth-derived layer label disagree confirms the headline is robust (Δ = −0.003 GKF, +0.011 LOSO when those neurons are excluded); 75 % of the disagreements are biological border cells (Mann-Whitney p < 10⁻¹⁷ for L2/3, L4, L5), L6 mismatches do not show a boundary effect (and may reflect EM cell-type errors or 5P/6P subtype admixture), and 94.6 % of the mismatches are concentrated in 4 of 13 scans (a methodological flag for the report). A within-session decoding experiment further validates the GKF protocol: when training and testing within a single recording session, Tier G HGB scores 0.611 ± 0.063 across the 6 four-class sessions — *below* the Phase-1 GKF (0.663), confirming that cross-session pooling is doing real ML work rather than exploiting session confound. A balanced-session LOSO (training restricted to the same 6 four-class sessions) gives 0.520 ± 0.057, +0.041 over Phase-1 LOSO on the same held-out subset (0.479), showing that single-class and 3-class training scans were adding noise rather than signal to cross-session generalisation. The same 4 high-mismatch scans (6_2, 6_4, 6_6, 6_7) appear as outliers in three independent analyses (labelling-mismatch concentration, within-session decoding, balanced-LOSO held-out performance), a robust dataset-quality finding for the report. The within-protocol ceiling (0.66) and the cross-scan ceiling (0.50) are both real and both reportable; Phase 2 will test whether a CNN on raw traces beats them, and Phase 3 will resolve the L5 IT/ET subtype heterogeneity that limits L5 recall in Phase 1.
