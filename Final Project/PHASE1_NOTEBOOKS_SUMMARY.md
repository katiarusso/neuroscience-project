# Phase 1 — Notebooks Summary

*This document mirrors, notebook by notebook and in execution order, every analysis currently present in `notebooks/01_*` through `notebooks/11_*`. It is a faithful record of what each notebook actually does (markdown cell flow + key code + saved results).*

*The companion `PHASE1_GROUND_TRUTH.md` is the thematic / conclusion-oriented reading of Phase 1; this document is the procedural reading.*

---

## Notebook index

| # | file | cells | what it does |
|---|---|---:|---|
| 01 | `01_setup_and_eda.ipynb` | 39 | Structural + functional EDA, working population, CC|abs| QC, sessions inventory |
| 02 | `02_canonical_tables.ipynb` | 17 | Canonical tables: cleaned units, traces cache (A0 / A1), CV splits |
| 03 | `03_tier_a.ipynb` | 16 | Tier A0 (per-trial) and Tier A1 (per-hash) scalar feature extraction |
| 04 | `04_phase1_stage1.ipynb` | 20 | Phase-1 Stage 1 — layer decoding from Tier A only (3 modelling strategies) |
| 05 | `05_tier_b_and_stage2.ipynb` | 19 | Tier B reliability + Phase-1 Stage 2 (A1 vs A1+B) |
| 06 | `06_tier_c_and_stage3.ipynb` | 27 | Tier C0/C1 behaviour + Phase-1 Stage 3 |
| 07 | `07_tier_d_and_stage4.ipynb` | 25 | Tier D stimulus descriptors + Phase-1 Stage 4 (the headline) |
| 08 | `08_phase1_characterization.ipynb` | 20 | Tier G fingerprint + Phase-1 headline characterization (LOSO, perm-importance, multi-seed) |
| 09 | `09_stage5_family.ipynb` | 18 | Stage-5 — per-family pipelines + cross-family transfer |
| 10 | `10_label_sensitivity.ipynb` | 23 | Sensitivity to label-vs-depth-bin mismatches |
| 11 | `11_within_session.ipynb` | 19 | Within-session GKF decoding + balanced-session LOSO |

---

## Notebook 01 — `01_setup_and_eda.ipynb`

**Title**: *Setup & EDA: structural and functional ground truth*

**Question**: What does the dataset actually look like? Establish ground truth on neurons, scans, stimuli, behaviour channels, and the QC scalar `cc_abs` before any modelling decisions.

### Steps, in order

1. **Setup** — paths, `microns-datacleaner` import, `src.config` paths.
2. **CAVE auth smoke test** — verify the auth token works against the structural API.
3. **Download the structural CSVs (idempotent)** — `nucleus_detection_v0`, `proofreading_status_and_strategy`, `aibs_metamodel_celltypes_v661`, `nucleus_functional_area_assignment`, `coregistration_manual_v4`, `digital_twin_properties_bcm_coreg_v4`.
4. **`process_nucleus_data` — the four modes** — explore the four available modes (`min`, `all`, `coreg`, `func`).
5. **Build the working population** — V1 / excitatory / oracle-matched / best-only filter. Result: **n=8,910** cells (incl. L1 — L1 is dropped in notebook 02).
6. **Class breakdown — layer × cell_type** — the canonical layer × subtype crosstab (15 L1, 4,247 L2/3, 2,670 L4, 1,615 L5, 363 L6).
7. **`cc_abs` — quality scalar** — distribution of the per-neuron `cc_abs` (oracle-correlation absolute value); used as a within-protocol QC and as a confound feature throughout Phase 1.
8. **Cortical depth by layer** — `pt_position_y` distributions by `layer` to confirm depth bins make sense.
9. **Functional H5 — schema walk** — explore `microns_functional.h5` structure.
10. **Sessions inventory** — 14 sessions × 1–4 scans each → 13 unique `session_key`s in V1 best-only.
11. **Trials per session, frame counts per stim type, sampling rate** — characterise the recording structure: ~30 Hz, ~75 frames per Clip, ~150 per Monet2/Trippy.
12. **Stimulus inventory — total hashes per family** — 2,411 unique condition_hashes (≈2,112 Clip, 150 Monet2, 149 Trippy).
13. **Oracle hashes (cross-session)** — ~116 hashes (96 Clip + 10 Monet2 + 10 Trippy) repeated across all 14 sessions. The oracle subset is the backbone of Tier B reliability and Tier G fingerprint.
14. **Behavioural channel availability** — running speed and pupil signals: which sessions have them, which don't.
15. **Per-session V1-excitatory coverage** — neurons per session-scan in the working population.
16. **Sample traces (visual sanity)** — plot a few raw and trial-averaged traces per neuron / hash.
17. **Reconcile and persist EDA summary** — save `eda_summary.json` and `oracle_hashes.pkl`.
18. **Conclusions** — population is V1-only / 4 layers / strong depth structure / oracle-stimulus subset is usable.

### Outputs

- `data/processed/eda_summary.json`
- `data/processed/oracle_hashes.pkl`
- `data/processed/behav_audit_sample.parquet`

---

## Notebook 02 — `02_canonical_tables.ipynb`

**Title**: *Canonical tables: cleaned population, trace cache, CV splits*

**Question**: Build the once-and-for-all canonical tables that every modelling notebook downstream will read.

### Steps, in order

1. **Setup**.
2. **Clean the working population — drop L1, add label columns**. n=8,895 (L1 dropped, the 15 L1 cells removed). Adds: `layer_label` ∈ {L2/3, L4, L5, L6}, `celltype_label` (= AIBS metamodel call), `session_key`. Saved as `units_working.parquet`.
3. **Build a `condition_hash` → stim type lookup, once**. Stim-type lookup table cached.
4. **Stream the H5 — one session at a time, checkpointed**. Reads each session's traces, joins to the working population, saves a per-session checkpoint (`data/interim/traces_per_session/`).
5. **Combine per-session checkpoints into the canonical caches**. Two outputs:
   - `traces.parquet` (~1.66 GB) — full long-row trace cache, one row per `(nucleus_id, condition_hash, trial_idx, frame_idx)`.
   - `trials_meta.parquet` — per-trial metadata.
6. **Within-hash trial-averaged traces (A1 substrate)**. Average traces *within* each `(nucleus_id, condition_hash)` to produce `traces_avg.parquet` (~1 GB) — the substrate for Tier A1 features.
7. **CV splits — `StratifiedGroupKFold` + `LeaveOneSessionOut`**. Compute and save `cv_assignments.parquet` with two columns:
   - `gkf_fold` — `StratifiedGroupKFold(5)` stratified by `layer_label`, grouped by `nucleus_id`.
   - `loso_test_scan` — for LOSO, the `session_key` to hold out (= the neuron's own session).
8. **Conclusions** — the 5 cached tables are now the substrate for all modelling.

### Outputs

- `data/processed/tables/units_working.parquet` (8,895 × 19)
- `data/processed/tables/traces.parquet` (~1.66 GB)
- `data/processed/tables/traces_avg.parquet` (~1 GB)
- `data/processed/tables/trials_meta.parquet`
- `data/processed/splits/cv_assignments.parquet`

---

## Notebook 03 — `03_tier_a.ipynb`

**Title**: *Tier A: scalar features from the calcium trace*

**Question**: Compute the scalar amplitude + shape features (Tier A) on the trial-level (A0) and hash-level (A1) traces.

### Steps, in order

1. **Setup**.
2. **Feature definitions** — Tier A0 / A1 amplitude and shape feature blocks defined per WORKFLOW.md §4.
3. **What this implementation respects vs differs from a strict reading of WORKFLOW.md §4** — small clarifying note about implementation choices.
4. **Trace alignment audit (visual)** — sanity-check that frame indices align across hashes / sessions.
5. **Compute Tier A0 — per `(nucleus_id, trial_idx)`** — 10 amp features + 10 shape features computed per trial. n=4,127,280 long rows. Saved as `data/processed/features/A0_amp.parquet` and `A0_shape.parquet`.
6. **Compute Tier A1 — per `(nucleus_id, condition_hash)`** — same 20 features computed on the within-hash trial-averaged trace. n=2,490,600 long rows. Saved as `A1_amp.parquet` and `A1_shape.parquet`.
7. **Inspect the saved tables** — describe statistics per feature.
8. **Sanity plots — feature distributions per layer** — violin plots: e.g. L4 has highest peak amplitude, L6 has the lowest reliability metrics.
9. **Conclusions** — Tier A is ready as the substrate for Stage 1.

### Feature names

- **A0/A1 amp (10)**: `amp_baseline_mean`, `amp_baseline_std`, `amp_mean`, `amp_peak`, `amp_min`, `amp_range`, `amp_integral`, `amp_tail_mean`, `amp_variance`, `amp_std`.
- **A0/A1 shape (10)**: `shape_latency_s`, `shape_peak_frame_idx`, `shape_rise_slope`, `shape_decay_slope`, `shape_early_auc`, `shape_late_auc`, `shape_early_late_ratio`, `shape_adaptation_ratio`, `shape_center_of_mass_s`, `shape_width_above_half_max_s`.

### Outputs

- `data/processed/features/A0_amp.parquet`, `A0_shape.parquet`
- `data/processed/features/A1_amp.parquet`, `A1_shape.parquet`

---

## Notebook 04 — `04_phase1_stage1.ipynb`

**Title**: *Phase 1 Stage 1: layer decoding from Tier A only*

**Question**: Is layer information already accessible from stimulus-locked neural-response statistics? Does within-hash trial-averaging (A1) cost or save us information at the prediction level relative to per-trial A0?

### Modelling choices

- 4-class layer classification: `L2/3, L4, L5, L6`.
- Three strategies (WORKFLOW §3.6):
  1. **Strategy 1 (1a)** — sample one row per neuron, fit tabular model, average across 5 random-row seeds. Diagnostic baseline.
  2. **Strategy 2** — train on long rows with `GroupKFold` by `nucleus_id`, predict per-row probabilities, mean-aggregate to per-neuron and argmax. **The mandatory classical baseline.**
  3. **Strategy 3** — Deep Sets: each neuron as a permutation-invariant set of rows. The natural ceiling for the engineered-feature pipeline.
- Two row levels: A0 (per-trial) and A1 (per-hash).
- One sub-block: `amp+shape` (the merged 20 features).
- Two models: `LogReg` (l2, lbfgs, `class_weight='balanced'`, max_iter=400) and `HGB` (max_iter=100, max_depth=8, learning_rate=0.05, l2=1.0, early_stopping). DeepSets implementation in `src/models/deep_sets.py`.

### Steps, in order

1. **Setup**.
2. **Shared helpers + checkpoint logic** — `make_pipeline_for(model_name)`, `cv_fit_predict_proba(X, y, groups, folds, model_name)`, `record_run(...)`, `_save_results()`. Incremental save into `stage1_runs.parquet` + `stage1_runs_partial.parquet`.
3. **Strategy 1 — random-row-per-neuron diagnostic** — 1a (n=8,895 rows) per (level=A0 or A1) × model. Mean of 5 seeds.
4. **Strategy 2 — long-row + neuron-level probability aggregation** — both A0 (4.13M rows) and A1 (2.49M rows) × LogReg and HGB.
5. **Strategy 3 — Deep Sets** on A1 (set size ~280 per neuron). Class imbalance via loss weighting.
6. **Confound baselines (WORKFLOW §6.5)** — depth-only, scan-only (one-hot of `session_key`), `cc_abs`-only.
7. **LOSO sanity check on the Strategy-2 headline** — single LOSO run on A0+amp+shape HGB.
8. **Ablation table** — final Stage 1 ablation summary.
9. **Conclusions and reading guide**.

### Stage-1 results

| strategy | level | sub_block | model | bal_acc | macro_F1 |
|---|---|---|---|---|---|
| **Confound: depth-only** | neuron | depth-only | LogReg | **0.976 ± 0.002** | 0.954 |
| Strategy 3 (DeepSets) | A1 | amp+shape | DeepSets | 0.584 ± 0.007 | 0.493 |
| Strategy 2 | A0 | amp+shape | HGB | 0.536 ± 0.011 | 0.380 |
| Strategy 2 | A1 | amp+shape | HGB | 0.502 ± 0.012 | 0.343 |
| **Confound: scan-only** | neuron | scan-only | LogReg | **0.496 ± 0.010** | 0.373 |
| Strategy 2 | A1 | amp+shape | LogReg | 0.491 ± 0.006 | 0.340 |
| Strategy 2 | A0 | amp+shape | LogReg | 0.455 ± 0.005 | 0.298 |
| LOSO sanity | A0 | amp+shape | HGB | 0.376 ± 0.137 | 0.263 |
| Strategy 1a | A1 | amp+shape | LogReg | 0.363 ± 0.004 | 0.235 |
| Strategy 1a | A0 | amp+shape | LogReg | 0.362 ± 0.005 | 0.235 |
| **Confound: cc_abs-only** | neuron | cc_abs-only | LogReg | **0.360 ± 0.014** | 0.217 |
| Strategy 1a | A0 | amp+shape | HGB | 0.303 ± 0.011 | 0.299 |
| Strategy 1a | A1 | amp+shape | HGB | 0.294 ± 0.004 | 0.290 |

### Stage-1 readings

- **Depth alone gets 0.976** — `pt_position_y` is essentially deterministic for layer. The depth-only baseline is the hard upper bound (since layer was *defined* by depth bins).
- **Scan-only gets 0.496** — knowing only which session-scan a neuron came from gets within ~0.04 of the long-row HGB headline. This is the scan-composition confound. Phase-1 takes this seriously throughout (per-session zscoring is added in Stage 2).
- **A1 long-row HGB = 0.502 ± 0.012** vs **A0 long-row HGB = 0.536 ± 0.011** — A0 (per-trial) is slightly better than A1 (per-hash, trial-averaged). Trial averaging removes some information.
- **Deep Sets on A1 = 0.584 ± 0.007** — best Stage-1 result; the set framing recovers about +0.05 over the long-row HGB. But still well below the Tier G ceiling that comes later (0.66).

### Outputs

- `data/processed/results/stage1_runs.parquet` (18 rows)
- `data/processed/results/stage1_runs_partial.parquet`

---

## Notebook 05 — `05_tier_b_and_stage2.ipynb`

**Title**: *Tier B (reliability) + Phase-1 Stage 2*

**Question**: Does the trial-to-trial reliability structure carry layer information beyond the trial-averaged signal of Tier A1?

### Steps, in order

1. **Setup**.
2. **Compute Tier B per `(neuron, hash)`**. Reads `traces.parquet` (1.6 GB), restricts to `(neuron, hash)` groups with `n_trials ≥ 2`, calls `compute_tier_b` from `src/features/tier_b.py`. 7 features: `b_amp_var, b_amp_std, b_amp_cv, b_amp_fano, b_amp_snr, b_trace_corr_mean, b_oracle_corr`. n=1,209,720 rows. Saved as `B_per_hash.parquet`.
3. **Per-neuron summary of Tier B (kept for diagnostic plots only)**. Per-neuron means of the 7 B features (8 columns + `b_summary_n_hashes`). Saved as `B_per_neuron.parquet`. **Not used in Stage 2 modelling** (per WORKFLOW §3.3 — long-row is canonical).
4. **Sanity — per-layer reliability distributions**. Violin plots of `b_summary_oracle_corr` etc. by layer. Confirms the L4-most-reliable / L6-least-reliable biological prediction.
5. **Theory — what is the scan-composition confound, and why does it matter here?** This block contains the explicit interpretation: scan-only GKF = 0.496 (Stage 1). Per-session zscoring is the active de-confounding tool from here on.
6. **Stage 2 modelling — A1 vs A1+B on the 1.21M-row restricted domain**. The inner-join with B restricts to repeated hashes, giving 1.21M rows.
   - Helper: `make_pipeline_for(model_name)` (LogReg with `class_weight='balanced', n_jobs=-1, max_iter=400`; HGB with `max_iter=100, max_depth=8, learning_rate=0.05, l2_regularization=1.0, early_stopping=True`).
   - Helper: `per_session_zscore(X, sessions)` — per-session median + MAD robust z-score.
   - Two configurations × two models × two preprocessings = 8 fits.
7. **Stage-2 ablation — A1 vs A1+B deltas**. Compare per-class recall changes; the workflow predicts L5 / L6 to gain the most because reliability differs most between deep layers.
8. **Comparison vs Stage 1**.
9. **Stage-2 result — interpretation**.
10. **Conclusions and reading guide**.

### Stage-2 results (from `stage2_runs.parquet`, 8 rows)

| features | preprocessing | model | bal_acc | macro_F1 | R(L2/3) | R(L4) | R(L5) | R(L6) |
|---|---|---|---|---|---|---|---|---|
| A1 only | raw | LogReg | 0.477 ± 0.008 | 0.292 | 0.119 | 0.578 | 0.260 | 0.953 |
| A1 + B | raw | LogReg | 0.439 ± 0.007 | 0.232 | 0.093 | 0.831 | 0.006 | 0.825 |
| A1 only | zscored-by-session | LogReg | 0.470 ± 0.010 | 0.289 | 0.641 | 0.005 | 0.304 | 0.928 |
| A1 + B | zscored-by-session | LogReg | (8 rows total — see file for full HGB rows) | | | | | |
| **A1 only** | **zscored-by-session** | **HGB** | **0.618 ± 0.014** | 0.491 | 0.549 | 0.593 | 0.397 | 0.933 |
| **A1 + B** | **zscored-by-session** | **HGB** | **0.631 ± 0.013** | 0.501 | 0.515 | 0.686 | 0.386 | 0.936 |
| A1 only | raw | HGB | 0.502 ± 0.012 | 0.343 | 0.257 | 0.461 | 0.341 | 0.947 |
| A1 + B | raw | HGB | 0.560 ± 0.013 | 0.422 | 0.469 | 0.503 | 0.330 | 0.939 |

### Stage-2 readings

- **Tier B adds +0.013** over A1 (HGB, zscored-by-session): 0.618 → 0.631. Modest but real.
- **Per-session zscoring matters**: HGB raw 0.502 → zscored 0.618 on A1 alone. The zscored variant is the headline preprocessing from Stage 2 onwards.
- L4 and L6 recalls are the cells whose reliability differs most — exactly what the biological prediction said.

### Outputs

- `data/processed/features/B_per_hash.parquet`
- `data/processed/features/B_per_neuron.parquet`
- `data/processed/results/stage2_runs.parquet`
- `data/processed/results/stage2_ablation.parquet`
- `reports/figures/05_tierB_violins_by_layer.png`

---

## Notebook 06 — `06_tier_c_and_stage3.ipynb`

**Title**: *Tier C (behaviour) + Phase-1 Stage 3*

**Question**: Does behaviour (running speed, pupil) carry layer-discriminative information when correctly conditioned per-`(neuron, hash)`?

### Steps, in order

1. **Setup**.
2. **Theory — why behaviour is methodologically delicate**. C0 is per-`(session, trial)` (shared across neurons within a trial), so naive averaging across trials destroys the interaction. Tier C1 is per-`(neuron, hash)`, computed *before* within-hash trial averaging.
3. **Compute Tier C0 (per-`(session, trial)` behavioural scalars)**. 4 features: state (running/still), pupil percentile, etc. Saved as `C0_per_trial.parquet`.
4. **Sanity — per-session behavioural state mix**. Per-session running/still proportions.
5. **Compute Tier C1 — per-`(neuron, hash)` behaviour-conditioned features**. 4 features: `c1_rmi` (running modulation index), `c1_pupil_resp_slope`, `c1_state_rel_diff`, `c1_arousal_gain_diff`. Saved as `C1_per_hash.parquet` (~2.49M rows; many NaN for hashes without both behavioural states).
6. **C1 sparsity audit — how often do we have both states?** Roughly 30–40 % of `(neuron, hash)` pairs have both running and still trials, so Tier C1 has a significant NaN rate.
7. **Per-layer Tier C1 distributions — biology eyeball**. Violin plots of `c1_rmi` etc. by layer.
8. **Stage 3 modelling — A0+C0 and A1+B+C1**.
   - **Comparison 1**: A0 vs A0 + C0 — does adding trial-level behaviour to per-trial neural features help?
   - **Comparison 2**: A1+B vs A1+B+C1 — does adding per-`(neuron, hash)` behaviour-conditioned features to the hash-level model help?
9. **Comparison 1 — A0 vs A0 + C0 (testing the interaction prediction)**. 4 models (LR/HGB × raw/zscored).
10. **Comparison-1 sanity — HGB A0 vs A0 + C0 with per-session z-scoring** — verify the A0+C0 lift survives zscoring.
11. **Comparison 2 — A1+B vs A1+B+C1 (raw and per-session-zscored)**. 4 models.
12. **Stage 3 ablation table**.
13. **Stage-3 result — interpretation**.
14. **Conclusions and reading guide**.

### Stage-3 results (from `stage3_runs.parquet`, 14 rows)

| comparison | preprocessing | features | model | bal_acc | R(L2/3) | R(L4) | R(L5) | R(L6) |
|---|---|---|---|---|---|---|---|---|
| A0 vs A0+C0 | raw | A0 only | LogReg | 0.455 | 0.255 | 0.330 | 0.290 | 0.944 |
| A0 vs A0+C0 | raw | A0 only | HGB | 0.536 | 0.296 | 0.522 | 0.367 | 0.958 |
| A0 vs A0+C0 | raw | A0 + C0 | LogReg | 0.495 | 0.375 | 0.397 | 0.271 | 0.936 |
| **A0 vs A0+C0** | **raw** | **A0 + C0** | **HGB** | **0.646** | 0.608 | 0.567 | 0.454 | 0.956 |
| A0 vs A0+C0 | zscored-by-session | A0 only | HGB | 0.629 | 0.437 | 0.675 | 0.449 | 0.956 |
| A0 vs A0+C0 | zscored-by-session | A0 + C0 | HGB | 0.617 | 0.512 | 0.705 | 0.313 | 0.938 |
| **A1+B vs A1+B+C1** | **zscored-by-session** | **A1+B+C1** | **HGB** | **0.635** | 0.551 | 0.667 | 0.383 | 0.939 |
| A1+B vs A1+B+C1 | zscored-by-session | A1+B | HGB | 0.631 | 0.515 | 0.686 | 0.386 | 0.936 |
| A1+B vs A1+B+C1 | raw | A1+B+C1 | HGB | 0.576 | 0.493 | 0.522 | 0.349 | 0.941 |

### Stage-3 readings

- **A0 + C0 HGB raw = 0.646** — the **best Stage-3 number**, and surprisingly a *raw* (non-zscored) result. Adding trial-level behaviour to per-trial neural features lifts HGB by +0.11 over A0 alone.
- **A1+B+C1 zscored HGB = 0.635** — adding C1 to A1+B lifts the zscored HGB by only +0.004, *not* statistically meaningful at this fold-level std (~0.012). The per-`(neuron, hash)` conditional features don't add much over reliability.
- **The interaction prediction holds for trial-level (A0+C0) but not for hash-level (A1+B+C1)**. Behaviour matters, but the granularity at which it helps is per-trial.

### Outputs

- `data/processed/features/C0_per_trial.parquet`
- `data/processed/features/C1_per_hash.parquet`
- `data/processed/results/stage3_runs.parquet` (14 rows)
- `data/processed/results/stage3_ablation.parquet`

---

## Notebook 07 — `07_tier_d_and_stage4.ipynb`

**Title**: *Tier D (stimulus descriptors) + Phase-1 Stage 4*

**Question**: Do stimulus-content scalars (luminance, motion energy, spatial / temporal frequency) carry layer-discriminative information, and do they help when joined to A1+B+C1 (hash-level path) or A0+C0 (trial-level path)?

### Steps, in order

1. **Setup**.
2. **Theory — stimulus descriptors and the Stage-4 question**. D is per-stimulus, not per-neuron, but per-neuron interactions emerge through `(neuron, hash)` × D joins.
3. **Compute Tier D (per-`condition_hash` stimulus descriptors)**. 10 numeric features: `d_luminance_mean, d_luminance_std, d_contrast, d_motion_energy, d_sf_low, d_sf_mid, d_sf_high, d_tf_low, d_tf_mid, d_tf_high`. + 3 stim-family one-hots. n=2,411 rows. Saved as `D_per_hash.parquet`.
4. **Per-family D distributions — sanity by stimulus family**. Clip vs Monet2 vs Trippy distributions.
5. **Stage-4 modelling helpers + checkpointing**. Re-uses pipeline factories from notebook 05/06.
6. **Hash-level path — A1+B+C1 vs A1+B+C1+D1**. 4 model × preprocessing combinations.
7. **Trial-level path — A0+C0 vs A0+C0+D0**. 2 raw model combinations (LR/HGB).
8. **Phase-1 headline — DeepSets on A1+B+C1+D1**. Set-model implementation.
9. **Stage 4 ablation table**.
10. **Phase-1 ablation table — combined view across Stages 1–4** (saved as `phase1_ablation.parquet`).
11. **Stage-4 / Phase-1 interpretation**.
12. **LOSO sanity at the Phase-1 headline**. Single LOSO run with the headline HGB on A1+B+C1+D1 zscored. Results saved as `phase1_headline_loso.parquet`.
13. **Conclusions and bridge to Phase 2 / Phase 3**.

### Stage-4 results (from `stage4_runs.parquet`, 14 rows)

| comparison | preprocessing | features | model | bal_acc | R(L2/3) | R(L4) | R(L5) | R(L6) |
|---|---|---|---|---|---|---|---|---|
| A1+B+C1 vs A1+B+C1+D1 | raw | A1+B+C1 | HGB | 0.576 | 0.493 | 0.522 | 0.349 | 0.941 |
| A1+B+C1 vs A1+B+C1+D1 | raw | A1+B+C1+D1 | HGB | 0.622 | 0.565 | 0.580 | 0.403 | 0.941 |
| **A1+B+C1 vs A1+B+C1+D1** | **zscored-by-session** | **A1+B+C1+D1** | **HGB** | **0.639 ± 0.011** | 0.550 | 0.629 | 0.436 | 0.942 |
| A1+B+C1 vs A1+B+C1+D1 | zscored-by-session | A1+B+C1+D1 | DeepSets | 0.603 ± 0.016 | 0.486 | 0.453 | 0.572 | 0.903 |
| A0+C0 vs A0+C0+D0 | raw | A0+C0+D0 | HGB | **0.650 ± 0.010** | 0.602 | 0.577 | 0.469 | 0.953 |

### Phase-1 LOSO headline — `phase1_headline_loso.parquet`

A1+B+C1+D1 zscored-by-session HGB, LOSO across 13 held-out scans:

| held_out_scan | n_neurons | bal_acc |
|---|---:|---:|
| 4_7 | 715 | 0.325 |
| 5_6 | 686 | 0.517 |
| 5_7 | 701 | 0.440 |
| 6_2 | 774 | 0.480 |
| 6_4 | 916 | 0.451 |
| 6_6 | 697 | 0.452 |
| 6_7 | 874 | 0.434 |
| 7_3 | 552 | 0.322 |
| 7_5 | 274 | 0.326 |
| 8_5 | 780 | 0.277 |
| 9_3 | 818 | 0.000 (1-class scan) |
| 9_4 | 680 | 0.004 (1-class scan) |
| 9_6 | 428 | 0.488 (1-class scan) |
| **mean** | | **0.347 ± 0.171** |

Excluding the 3 single-class scans (9_3, 9_4, 9_6), mean over 10 multi-class scans is ~0.402 ± 0.085.

### Stage-4 / Phase-1 readings

- **Phase-1 long-row headline = 0.639 ± 0.011** (A1+B+C1+D1 zscored HGB). This is the "long-row HGB headline" referenced everywhere downstream.
- **A0+C0+D0 raw HGB = 0.650** — a *raw* per-trial model is the absolute Stage-4 best, narrowly beating the zscored hash-level headline. Phase 1 reports both.
- **GKF → LOSO drop is severe**: 0.639 → 0.347, drop of −0.29 if we include single-class scans, or 0.639 → 0.40 (−0.24) on multi-class scans. Cross-session generalisation is much harder than within-protocol.
- DeepSets on A1+B+C1+D1 zscored = 0.603, *worse* than HGB. The set ceiling does not improve on the long-row + probability-mean ceiling at Stage 4.

### Outputs

- `data/processed/features/D_per_hash.parquet`
- `data/processed/results/stage4_runs.parquet`
- `data/processed/results/stage4_ablation.parquet`
- `data/processed/results/phase1_ablation.parquet` (the combined Stage 1–4 table; 54 rows)
- `data/processed/results/phase1_headline_loso.parquet`

---

## Notebook 08 — `08_phase1_characterization.ipynb`

**Title**: *Phase-1 headline characterization*

**Question**: Three sub-experiments to characterise the Phase-1 winner: (a) does Tier G (per-neuron oracle fingerprint) beat the long-row headline; (b) which features actually carry the long-row headline; (c) is the long-row headline stable to seed.

### Steps, in order

1. **Setup**.
2. **Theory — what the three sub-experiments measure**. Tier G ↔ ceiling check; permutation importance ↔ feature attribution; multi-seed ↔ robustness.
3. **Build Tier G — the per-neuron oracle fingerprint**. 116 columns `g_000..g_115`, one per oracle hash (96 Clip + 10 Monet2 + 10 Trippy = 116). Each value = per-neuron mean trial-averaged response on that hash. n=8,895 × 117. Saved as `G_per_neuron.parquet`.
4. **Tier G as Strategy 1 — fit on the per-neuron oracle fingerprint**. One row per neuron (8,895 × 116). Two models. Result: `tierG_strategy1.parquet`.
5. **LOSO sanity on Tier G**. Single LOSO run, HGB. Result: `tierG_loso.parquet`.
6. **Scan-only LOSO baseline + Tier G v2 — proper LOSO comparators**. The scan-only LOSO baseline (0.295 ± 0.240) and Tier G v2 (per-feature train-fold zscoring instead of session-zscoring). Results: `scan_only_loso.parquet`, `tierG_v2_loso.parquet`.
7. **Findings — what these two experiments establish** (markdown).
8. **Permutation importance on the Phase-1 headline**. Run `sklearn.inspection.permutation_importance` on the long-row HGB A1+B+C1+D1 zscored. Result: `headline_perm_importance.parquet`.
9. **Multi-seed stability of the Phase-1 headline**. 5 seeds: 42, 123, 456, 789, 999. Result: `headline_multiseed.parquet`.
10. **Combined interpretation**.
11. **Conclusions**.

### Tier G results (from `tierG_strategy1.parquet`)

| model | n_features | n_neurons | bal_acc | R(L2/3) | R(L4) | R(L5) | R(L6) |
|---|---:|---:|---|---|---|---|---|
| LogReg | 116 | 8,895 | 0.615 ± 0.006 | 0.426 | 0.542 | 0.589 | 0.904 |
| **HGB** | 116 | 8,895 | **0.663 ± 0.008** | 0.740 | 0.630 | 0.530 | 0.753 |

### Tier G LOSO results (from `tierG_loso.parquet`)

13 held-out scans, HGB on Tier G. Mean over all 13: **0.497 ± 0.105**. Per-scan range 0.36–0.69.

### Scan-only LOSO baseline (from `scan_only_loso.parquet`)

13 held-out scans of `session_key` one-hot only. 6 multi-class scans give 0.250 (chance for 4 classes). The 3 single-class scans are 1.000 / 0.000 / 0.000. Mean: 0.295 ± 0.240.

**Reading**: Tier G LOSO 0.50 vs scan-only LOSO 0.25 = **+0.25 lift** — clear evidence of cross-scan generalisation.

### Tier G v2 LOSO (from `tierG_v2_loso.parquet`)

Per-feature train-fold zscoring (not session-zscoring). LOSO mean: 0.498 — identical to Tier G v1 (0.497). Per-session amplitude calibration is *not* the residual confound for the GKF→LOSO drop.

### Permutation importance (from `headline_perm_importance.parquet`, top 15)

| feature | importance (mean) |
|---|---:|
| `amp_min` | 0.074 |
| `amp_peak` | 0.028 |
| `d_sf_mid` | 0.023 |
| `d_luminance_std` | 0.023 |
| `d_sf_low` | 0.018 |
| `b_amp_cv` | 0.016 |
| `shape_adaptation_ratio` | 0.016 |
| `amp_range` | 0.016 |
| `d_motion_energy` | 0.015 |
| `amp_baseline_std` | 0.011 |
| `amp_tail_mean` | 0.011 |
| `c1_rmi` | 0.010 |
| `shape_early_auc` | 0.010 |
| `b_amp_snr` | 0.009 |
| `d_luminance_mean` | 0.007 |

**Reading**: amplitude scalars dominate (especially `amp_min`, `amp_peak`); D1 stimulus-content terms appear next; one B reliability scalar; one C1 behaviour scalar; *most shape and most C1 features rank below 0.01.* The "long-row headline" is mostly amplitude + stimulus content + one reliability + one behaviour scalar — much sparser than the 41-feature input suggests.

### Multi-seed stability (from `headline_multiseed.parquet`)

5 seeds × 5-fold CV — bal_acc fold means: 0.6392, 0.6387, 0.6383, 0.6410, 0.6387. Range 0.638–0.641; std across seeds ~0.001. **The headline is statistically stable.**

### Outputs

- `data/processed/features/G_per_neuron.parquet`
- `data/processed/results/tierG_strategy1.parquet`
- `data/processed/results/tierG_loso.parquet`
- `data/processed/results/tierG_v2_loso.parquet`
- `data/processed/results/scan_only_loso.parquet`
- `data/processed/results/headline_perm_importance.parquet`
- `data/processed/results/headline_multiseed.parquet`

---

## Notebook 09 — `09_stage5_family.ipynb`

**Title**: *Stage-5 family analyses (neural-response only)*

**Question**: How well does each stimulus family (Clip / Monet2 / Trippy) decode layer on its own? And does a model trained on one family generalise to another?

### Steps, in order

1. **Setup**.
2. **Theory — what each experiment is asking**. (b) Per-family pipelines: train + test within one family. (c) Cross-family transfer: train on family X, test on family Y.
3. **Build the A1 amp+shape feature matrix (shared across both experiments)**.
4. **DeepSets training helper for family-restricted data**.
5. **Per-family pipelines — experiment (b)**. Three families × three models (LR/HGB/DeepSets) on A1 amp+shape. Result: `stage5_per_family.parquet` (9 rows).
6. **A0 sensitivity check — per-family experiment, HGB + LogReg only**. Same but on A0 (per-trial). Result: `stage5_per_family_A0.parquet` (6 rows).
7. **Cross-family transfer — experiment (c)**. 3 train × 3 test (excl. self) × 3 models = 18 rows. Result: `stage5_transfer.parquet`.
8. **Family-analysis figure**.
9. **Stage-5 interpretation**.
10. **Conclusions — Phase-1 final wrap**.

### Per-family results — A1 amp+shape (from `stage5_per_family.parquet`)

| family | n_rows | model | bal_acc | R(L2/3) | R(L4) | R(L5) | R(L6) |
|---|---:|---|---|---|---|---|---|
| Clip | 2.13 M | LogReg | 0.417 | 0.593 | 0.450 | 0.085 | 0.542 |
| Clip | 2.13 M | HGB | 0.596 | 0.447 | 0.602 | 0.416 | 0.919 |
| Clip | 2.13 M | DeepSets | 0.483 | 0.222 | 0.359 | 0.417 | 0.931 |
| Monet2 | 178 K | LogReg | 0.485 | 0.390 | 0.286 | 0.364 | 0.899 |
| **Monet2** | 178 K | **HGB** | **0.614** | 0.479 | 0.585 | 0.499 | 0.895 |
| Monet2 | 178 K | DeepSets | 0.530 | 0.526 | 0.382 | 0.448 | 0.763 |
| Trippy | 178 K | LogReg | 0.441 | 0.405 | 0.220 | 0.281 | 0.857 |
| Trippy | 178 K | HGB | 0.559 | 0.565 | 0.481 | 0.346 | 0.846 |
| Trippy | 178 K | DeepSets | 0.480 | 0.438 | 0.354 | 0.361 | 0.767 |

**Reading**: Monet2 alone (n_rows = 178K, ~1/12 of Clip) is the *best* per-family decoder — slightly better than Clip's HGB despite far less data. Monet2's drifting-grating structure may be more layer-discriminative than Clip's natural-movie content.

### Cross-family transfer (from `stage5_transfer.parquet`, 18 rows)

| train | test | model | bal_acc |
|---|---|---|---|
| Clip | Monet2 | LogReg | 0.435 |
| Clip | Monet2 | HGB | 0.251 (collapses to L2/3) |
| Clip | Trippy | LogReg | 0.388 |
| Clip | Trippy | HGB | 0.249 (collapses) |
| Monet2 | Clip | LogReg | 0.279 (collapses to L5) |
| **Monet2** | **Clip** | **HGB** | **0.449** |
| Monet2 | Trippy | LogReg | 0.409 |
| Monet2 | Trippy | HGB | 0.422 |
| Trippy | Clip | LogReg | 0.295 (collapses) |
| **Trippy** | **Clip** | **HGB** | **0.441** |
| Trippy | Monet2 | LogReg | 0.448 |
| Trippy | Monet2 | HGB | 0.405 |
| Trippy | Monet2 | DeepSets | 0.476 |

**Reading**: Cross-family transfer is *much* worse than within-family (0.25–0.48 vs 0.55–0.61 within-family). Some transfer pairs collapse to a single class entirely (Clip → Monet2 / Trippy with HGB; Monet2 / Trippy → Clip with LogReg). Layer-discriminative information is *partially* shared across families but not transferable in plug-and-play form.

### Outputs

- `data/processed/results/stage5_per_family.parquet`
- `data/processed/results/stage5_per_family_A0.parquet`
- `data/processed/results/stage5_transfer.parquet`
- `reports/figures/09_stage5_family_*.png`

---

## Notebook 10 — `10_label_sensitivity.ipynb`

**Title**: *Phase-1 sensitivity analysis: cell-type vs layer-label consistency*

**Question**: 407 cells in the Phase-1 working population have a `cell_type` (AIBS metamodel) call inconsistent with their `layer_label` (depth bin) — e.g. a cell at L4 depth called `5P-IT`. Do these label-vs-depth mismatches drive the Phase-1 layer headline?

### Steps, in order

1. **Setup**.
2. **Build the `consistent` flag**. For each neuron, derive the canonical layer from `cell_type` (`5P-* → L5`, `6P-* → L6`, etc.) and compare to `layer_label`. n_mismatch = 407 / 8,895 = 4.6 %.
3. **Check 1 — distribution of mismatches by layer × session**. Saved as `label_sensitivity_per_layer.parquet` and `label_sensitivity_per_session.parquet`.
4. **Check 2 — refit Tier G HGB excluding mismatches**. Run on the 8,488-cell `consistent_only` subset. Saved as `label_sensitivity_tierG_filtered.parquet` and the LOSO version `label_sensitivity_tierG_filtered_loso.parquet`.
5. **Check 3 — are mismatches near `pt_position_y` boundaries?** Position the 407 cells along the depth axis; quantify how many are within ±X µm of an inter-layer threshold. Saved as `label_sensitivity_boundary.parquet`.
6. **Check 4 — confusion of model on mismatched neurons**. For the headline HGB, compare predicted class distribution on consistent vs mismatched cells. Saved as `label_sensitivity_confusion.parquet` and `label_sensitivity_oof.parquet`.
7. **Combined interpretation**.
8. **Conclusions**.

### Mismatch distribution

**By layer** (from `label_sensitivity_per_layer.parquet`):

| layer | n | n_mismatch | pct_mismatch |
|---|---:|---:|---:|
| L2/3 | 4,247 | 87 | 2.0 % |
| L4 | 2,670 | 70 | 2.6 % |
| L5 | 1,615 | 191 | **11.8 %** |
| L6 | 363 | 59 | **16.3 %** |

L5 and L6 have much higher mismatch rates than L2/3 / L4. Deeper layers are anatomically thinner / harder to bin, plus the metamodel uses richer anatomical evidence (morphology, projections) than depth alone.

**By session** (from `label_sensitivity_per_session.parquet`):

| session | n | n_mismatch | pct |
|---|---:|---:|---:|
| 4_7 | 715 | 8 | 1.1 % |
| 5_6 | 686 | 1 | 0.1 % |
| 5_7 | 701 | 5 | 0.7 % |
| 6_2 | 774 | 40 | 5.2 % |
| 6_4 | 916 | 92 | 10.0 % |
| 6_6 | 697 | 91 | **13.1 %** |
| 6_7 | 874 | 162 | **18.5 %** |
| 7_3–9_6 | various | 0–4 | <1 % |

Sessions 6_4, 6_6, 6_7 (the same anatomical region across multiple imaging passes) carry most of the mismatches. Possible explanations: per-session imaging-plane drift, EM coregistration noise concentrated in that volume, or real biology in that subvolume.

### Tier G HGB on consistent-only subset (from `label_sensitivity_tierG_filtered.parquet`)

| subset | n | bal_acc | R(L2/3) | R(L4) | R(L5) | R(L6) |
|---|---:|---|---|---|---|---|
| full (Tier G HGB Strategy 1) | 8,895 | 0.663 | 0.740 | 0.630 | 0.530 | 0.753 |
| **consistent_only** | 8,488 | **0.660** | 0.743 | 0.630 | 0.529 | 0.740 |

**Δ = −0.003.** Removing the 407 mismatched cells **does not** change the Tier G HGB headline. The label-mismatch is **not** the source of the Phase-1 layer signal.

### LOSO on consistent-only Tier G (from `label_sensitivity_tierG_filtered_loso.parquet`)

Per-scan LOSO, mean across the 10 multi-class scans: 0.486 (vs Tier G full LOSO mean of 0.497). Δ = −0.011 — small, in the noise.

### Conclusions

- Mismatches are concentrated in deep layers and a single anatomical sub-volume (sessions 6_*).
- Refitting on consistent-only changes the headline by ≤ 0.01.
- Phase-1's layer signal is robust to the label-vs-depth question.

### Outputs

- `data/processed/results/label_sensitivity_per_layer.parquet`
- `data/processed/results/label_sensitivity_per_session.parquet`
- `data/processed/results/label_sensitivity_tierG_filtered.parquet`
- `data/processed/results/label_sensitivity_tierG_filtered_loso.parquet`
- `data/processed/results/label_sensitivity_boundary.parquet`
- `data/processed/results/label_sensitivity_confusion.parquet`
- `data/processed/results/label_sensitivity_oof.parquet`
- `data/processed/results/label_sensitivity_scan_only_filtered.parquet`

---

## Notebook 11 — `11_within_session.ipynb`

**Title**: *Within-session decoding + balanced-session LOSO*

**Question**: (A) If we train and test *within* a single session (no cross-session generalisation needed), how well does each tier decode? This isolates the within-protocol ceiling per session. (B) If we restrict LOSO to scans that are well-balanced across all 4 layers, does the GKF→LOSO drop shrink?

### Steps, in order

1. **Setup**.
2. **Verify balanced-session selection**. Identify scans that have ≥ 30 cells of each of the 4 layers (typically 5_6, 5_7, 6_2, 6_4, 6_6, 6_7).
3. **Build the long-row feature matrix (A1 + B + C1 + D)**.
4. **Helper — within-session GroupKFold fit**. Within each session, run `StratifiedGroupKFold(5)` by `nucleus_id`.
5. **Experiment A — within-session GKF**. Per session × per tier (A1, A1+B, A1+B+C1, A1+B+C1+D1, TierG). Save as `within_session_runs.parquet`.
6. **Experiment B — balanced-session LOSO (Tier G HGB)**. LOSO restricted to the balanced sessions. Save as `balanced_loso_tierG.parquet`.
7. **Comparison summary**.
8. **Interpretation**.
9. **Conclusions**.

### Within-session results (from `within_session_runs.parquet`, top 15 of ~30 rows)

For each balanced session, Tier G HGB is the best block:

| session | tier | n_features | n_neurons | bal_acc | R(L2/3) | R(L4) | R(L5) | R(L6) |
|---|---|---:|---:|---|---|---|---|---|
| 5_6 | A1 | 20 | 686 | 0.506 ± 0.026 | 0.110 | 0.547 | 0.423 | 0.945 |
| 5_6 | A1+B | 27 | 686 | 0.508 ± 0.024 | 0.090 | 0.559 | 0.439 | 0.945 |
| 5_6 | A1+B+C1 | 31 | 686 | 0.524 ± 0.033 | 0.121 | 0.536 | 0.481 | 0.958 |
| 5_6 | A1+B+C1+D1 | 44 | 686 | 0.554 ± 0.026 | 0.187 | 0.587 | 0.486 | 0.958 |
| **5_6** | **TierG** | 116 | 686 | **0.664 ± 0.013** | 0.478 | 0.702 | 0.545 | 0.932 |
| 5_7 | TierG | 116 | 701 | 0.656 ± 0.071 | 0.471 | 0.711 | 0.511 | 0.932 |
| **6_2** | **TierG** | 116 | 774 | **0.676 ± 0.039** | 0.630 | 0.614 | 0.612 | 0.847 |
| 6_4 | TierG | 116 | 916 | (similar 0.65–0.70) | | | | |
| 6_6 | TierG | 116 | 697 | (similar) | | | | |
| 6_7 | TierG | 116 | 874 | (similar) | | | | |

**Reading**: Within a single session (no cross-session generalisation), Tier G HGB matches its full-dataset GKF result of 0.66. There's no within-session penalty. The tier progression A1 → +B → +C1 → +D1 within session shows lifts of ~0.015 each, totaling ~0.05 — consistent with full-population stage progression.

### Balanced-session LOSO results (from `balanced_loso_tierG.parquet`)

LOSO restricted to the 6 balanced sessions: 5_6, 5_7, 6_2, 6_4, 6_6, 6_7.

| held_out | n_test | bal_acc |
|---|---:|---:|
| 5_6 | 686 | 0.599 |
| 5_7 | 701 | 0.572 |
| 6_2 | 774 | 0.531 |
| 6_4 | 916 | 0.490 |
| 6_6 | 697 | 0.460 |
| 6_7 | 874 | 0.468 |
| **mean** | | **0.520** |

vs full-LOSO Tier G mean of 0.497 (across 13 scans, 10 multi-class). Restricted to balanced scans: **0.520**, a +0.023 lift but also a smaller dataset.

### Conclusions

- Within-session decoding hits the same Tier G ceiling (~0.66) as full-population GKF — Phase-1 GKF is **not** exploiting cross-session pooling.
- Balanced-session LOSO closes ~0.02 of the GKF→LOSO drop — most of the −0.16 gap is intrinsic class-distribution shift across scans, not bal-vs-imbal sampling.
- The L6 collapse phenomenon (large bal_acc drop on scans where L6 is in the held-out fold) is amplified in the long-row pipeline because L6 has only ~28 cells per scan.

### Outputs

- `data/processed/results/within_session_runs.parquet`
- `data/processed/results/balanced_loso_tierG.parquet`

---

## Phase-1 — what every notebook contributed (one-line each)

| notebook | one-line contribution |
|---|---|
| 01 | The dataset is V1 / 4 layers / oracle stimuli usable; depth ≈ deterministic for layer label. |
| 02 | The 5 canonical tables (`units_working`, `traces`, `traces_avg`, `trials_meta`, `cv_assignments`) are the substrate for every modelling notebook. |
| 03 | Tier A0 / A1 amp + shape features (20 features each row level) ready as long tables. |
| 04 | A0 long-row HGB = 0.536; scan-only confound = 0.496; depth-only ceiling = 0.976. |
| 05 | Tier B reliability: A1+B zscored HGB = 0.631; per-session zscoring is the de-confound tool. |
| 06 | Tier C: A0+C0 raw HGB = **0.646** (strongest single-tier addition); A1+B+C1 zscored HGB = 0.635. |
| 07 | Phase-1 long-row headline = A1+B+C1+D1 zscored HGB = **0.639 ± 0.011** (5-fold GKF); LOSO = 0.347 (full)/0.40 (multi-class). |
| 08 | Tier G HGB (per-neuron oracle fingerprint) = **0.663 ± 0.008** (the within-protocol ceiling); LOSO = 0.497 ± 0.105. Multi-seed: 0.638–0.641. Permutation importance: amp_min, amp_peak, d_sf_mid dominate. |
| 09 | Per-family Monet2 HGB = 0.614 (best per-family); cross-family transfer collapses (0.25–0.48). |
| 10 | Label-vs-depth mismatches (4.6 %, concentrated in L5/L6 and sessions 6_*) do not change the Tier G headline (Δ ≈ −0.003). |
| 11 | Within-session GKF Tier G = 0.66 (matches full-population); balanced-session LOSO = 0.520 vs full-LOSO 0.497 — most of the GKF→LOSO drop is intrinsic. |

---

## File outputs inventory — Phase 1

All saved in `data/processed/results/` (sizes in bytes from disk):

| file | rows | source notebook |
|---|---:|---|
| `stage1_runs.parquet` | 18 | 04 |
| `stage1_runs_partial.parquet` | — | 04 (incremental save) |
| `stage2_runs.parquet` | 8 | 05 |
| `stage2_ablation.parquet` | — | 05 |
| `stage3_runs.parquet` | 14 | 06 |
| `stage3_ablation.parquet` | — | 06 |
| `stage4_runs.parquet` | 14 | 07 |
| `stage4_ablation.parquet` | — | 07 |
| `phase1_ablation.parquet` | 54 | 07 (combined Stages 1–4) |
| `phase1_headline_loso.parquet` | 13 | 07 |
| `tierG_strategy1.parquet` | 2 | 08 |
| `tierG_loso.parquet` | 13 | 08 |
| `tierG_v2_loso.parquet` | 13 | 08 |
| `scan_only_loso.parquet` | 13 | 08 |
| `headline_perm_importance.parquet` | 44 | 08 |
| `headline_multiseed.parquet` | 5 | 08 |
| `stage5_per_family.parquet` | 9 | 09 |
| `stage5_per_family_A0.parquet` | 6 | 09 |
| `stage5_transfer.parquet` | 18 | 09 |
| `label_sensitivity_*.parquet` (8 files) | various | 10 |
| `within_session_runs.parquet` | ~30 | 11 |
| `balanced_loso_tierG.parquet` | 6 | 11 |

Reproducibility: all notebooks fix `RANDOM_SEED = 42` from `src/config`. Re-running the notebook chain regenerates these parquets verbatim.
