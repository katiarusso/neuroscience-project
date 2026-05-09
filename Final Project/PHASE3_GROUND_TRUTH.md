# Phase 3 — Ground Truth Document

*This document mirrors, notebook by notebook and in execution order, every analysis currently present in `notebooks/phase3/`. It is the empirical record. The companion document `PHASE3_PLANNING.md` is the methodological commitment made before Phase 3 began.*

*Phase 3 was rebuilt once. The original (v1) versions of notebooks 02, 03, 04 and the planned 05 used cross-hash averaging of A1 / C1 / D1 features to obtain one row per neuron — an operation forbidden by `WORKFLOW.md §3.3`. The current notebooks (referred to here as v2 throughout) replace that step with the WORKFLOW §3.6 long-row + per-neuron probability aggregation protocol Phase 1 followed. v1 numbers are noted in the relevant sections only as a record of the correction; they do not stand as headlines.*

---

## 0. Scope and methodological commitments common to every Phase 3 notebook

### 0.1 Population

The Phase 3 unit population is the Phase 1 working population — V1, excitatory, oracle-matched, best-only, L1 dropped — n=8,895 cells across 13 scans. The seven AIBS metamodel subtype labels (`celltype_label` in `units_working.parquet`) are:

| subtype | n |
|---|---:|
| 23P | 4,198 |
| 4P | 2,845 |
| 5P-IT | 1,169 |
| 5P-ET | 320 |
| 5P-NP | 26 |
| 6P-IT | 216 |
| 6P-CT | 121 |

Imbalance ratio: 162:1 between most-common (23P) and rarest (5P-NP) class.

### 0.2 The five feature blocks

Same blocks throughout, mirroring Phase 1 stage 1–4:

| block | what it is | n features |
|---|---|---:|
| `G` | Tier G per-neuron tuning fingerprint, one column per oracle hash (broadcast onto every long row of a neuron) | 116 |
| `A1+B` | A1 amplitude + shape (per-row) + B reliability (per-row) | 27 |
| `A1+B+C1` | + C1 behaviour / gain context (per-row) | 31 |
| `A1+B+C1+D1` | + D1 stimulus-content (per-row, joined on `condition_hash`) | 41 |
| `G+B+C1` | hybrid: G broadcast + B + C1 | 127 |

### 0.3 Long-row protocol (WORKFLOW §3.6)

Every Phase 3 modelling notebook builds the table with the same join pattern Phase 1 stage 4 used:

```
df_a1   = build_modeling_table(level='A1', blocks=['amp','shape'], label='celltype_label')
df_a1b  = df_a1.merge(B_per_hash,  on=['nucleus_id','condition_hash'], how='inner')
df_a1bc1 = df_a1b.merge(C1_per_hash, on=['nucleus_id','condition_hash'], how='left')
df_full = df_a1bc1.merge(D_per_hash, on='condition_hash', how='left')
df_full = df_full.merge(G_per_neuron, on='nucleus_id', how='left')   # broadcast
```

Inner-join on B restricts to repeated hashes (`n_trials ≥ 2`). After this join the long-row table has 1,209,720 rows (200,504 for the L5 IT/ET subset, 39,656 for the L6 IT/CT subset, 1,209,720 for multiclass full).

### 0.4 CV protocol

- **GKF**: `StratifiedGroupKFold(5)` grouped by `nucleus_id`, stratified by `celltype_label`. The stratification key is recomputed per Phase 3 notebook (Phase 1's precomputed `gkf_fold` was stratified by `layer_label`).
- **LOSO**: held-out scan = `session_key`; train on the other 12. Validity rule locked in notebook 01 (`phase3_loso_valid_scans.json`): minority class ≥ 5 cells AND both classes present in held-out scan. Multiclass uses `≥4 of 7 classes with ≥5 cells each`.

### 0.5 Models

- **LogReg**: `lbfgs`, `penalty='l2'`, `C=1.0`, `max_iter=400`, `class_weight='balanced'`.
- **HGB**: `max_iter=100`, `max_depth=8`, `learning_rate=0.05`, `l2_regularization=1.0`, `early_stopping=True`, `n_iter_no_change=10`.

Both are wrapped in a sklearn `Pipeline`: LR uses `SimpleImputer(median) → RobustScaler → clf`; HGB uses NaN passthrough to its native handler. Class imbalance is handled by `class_weight='balanced'` *and* `compute_sample_weight('balanced', y[tr])` passed to `fit()` — Phase 1's convention.

### 0.6 Neuron-level scoring

`src/eval/metrics.py:neuron_level_score`: aggregate per-row predicted probabilities to per-neuron probability vectors (mean across the neuron's rows), argmax for the predicted label, score on neuron-level predictions. Headline metric = balanced accuracy. Auxiliary: macro-F1, per-class recall, confusion matrix.

### 0.7 Confound baselines (every modelling notebook)

| baseline | what it tests |
|---|---|
| **majority class** | trivial floor |
| **scan-only** (one-hot of `session_key`, long-row) | scan-composition confound (under GKF only — under LOSO it is at chance by construction) |
| **cc_abs-only** | SNR-only floor |
| **cc_abs-residualized winner** | per-feature OLS residual on `cc_abs` within the training fold; tests whether the winner's signal survives SNR control |

Under LOSO, the scan-only baseline is run as an **integrity check**: it must come back at chance. If it doesn't, the LOSO split is broken.

---

## 1. Notebook `01_phase3_labels.ipynb` — Build the Phase-3 subtype label table

### 1.1 Question

*Define the Phase 3 unit population, attach AIBS metamodel subtype labels (with corrections applied), and publish the per-scan counts that gate every downstream LOSO decision.*

### 1.2 Inputs

- `data/processed/tables/units_v1_exc_best.parquet` — Phase 1 canonical population (n=8,910 incl. L1).
- `data/1718/raw/aibs_metamodel_celltypes_v661.csv` and `..._corrections.csv`.

### 1.3 Steps, in order

1. **Load Phase-1 canonical units**: 8,910 cells with `cell_type` and `layer` columns. The `cell_type` value counts already match the corrections-applied AIBS metamodel — sanity-checked by re-deriving the corrections-applied join on `pt_root_id` and confirming 100 % agreement (8,910 / 8,910).
2. **Verify cell_type values** are exactly the seven Phase-3 subtypes (no `unexpected` values).
3. **Build the extended unit table** with helper columns:
   - `scan_id` (= `session_key`) = `f"{int(session)}_{int(scan_idx)}"`
   - `subtype` (alias of `cell_type`)
   - boolean indicators: `is_L5_IT`, `is_L5_ET`, `is_L6_IT`, `is_L6_CT`
4. **Global subtype counts** (saved as `phase3_subtype_counts_global.csv`).
5. **Per-scan crosstab** (saved as `phase3_subtype_counts_per_scan.csv`).
6. **LOSO validity** with prespecified rule `minority ≥ 5 AND both classes present`, applied separately to L5 IT vs ET and L6 IT vs CT. Saved as `phase3_loso_valid_scans.json`.
7. **Save extended unit table** as `units_v1_exc_best_subtype.parquet`.

### 1.4 Results

**Global counts** — see §0.1 above.

**Per-scan subtype × scan crosstab (cells per scan):**

| scan | 23P | 4P | 5P-IT | 5P-ET | 5P-NP | 6P-IT | 6P-CT | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4_7 | 377 | 240 | 39 | 66 | 5 | 0 | 1 | 728 |
| 5_6 | 151 | 249 | 146 | 44 | 0 | 46 | 50 | 686 |
| 5_7 | 185 | 284 | 134 | 16 | 1 | 57 | 24 | 701 |
| 6_2 | 282 | 292 | 150 | 9 | 5 | 27 | 9 | 774 |
| 6_4 | 473 | 175 | 206 | 5 | 2 | 41 | 14 | 916 |
| 6_6 | 401 | 87 | 158 | 16 | 1 | 23 | 11 | 697 |
| 6_7 | 455 | 169 | 183 | 33 | 2 | 22 | 12 | 876 |
| 7_3 | 203 | 190 | 92 | 64 | 3 | 0 | 0 | 552 |
| 7_5 | 176 | 76 | 8 | 10 | 4 | 0 | 0 | 274 |
| 8_5 | 400 | 268 | 53 | 56 | 3 | 0 | 0 | 780 |
| 9_3 | 2 | 815 | 0 | 1 | 0 | 0 | 0 | 818 |
| 9_4 | 680 | 0 | 0 | 0 | 0 | 0 | 0 | 680 |
| 9_6 | 428 | 0 | 0 | 0 | 0 | 0 | 0 | 428 |

**LOSO validity (locked):**

- **L5 IT vs ET — 10 valid scans**: `4_7, 5_6, 5_7, 6_2, 6_4, 6_6, 6_7, 7_3, 7_5, 8_5`. Invalid: `9_3, 9_4, 9_6` (no L5 cells).
- **L6 IT vs CT — 6 valid scans**: `5_6, 5_7, 6_2, 6_4, 6_6, 6_7`. Invalid: `9_3, 8_5, 4_7, 9_4, 7_3, 9_6, 7_5`.

**Layer × subtype crosstab (depth bin × metamodel call):**

| layer | 23P | 4P | 5P-IT | 5P-ET | 5P-NP | 6P-IT | 6P-CT |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| L2/3 | 4,160 | 87 | 0 | 0 | 0 | 0 | 0 |
| L4 | 38 | 2,600 | 28 | 4 | 0 | 0 | 0 |
| L5 | 0 | 158 | 1,117 | 289 | 18 | 20 | 13 |
| L6 | 0 | 0 | 24 | 27 | 8 | 196 | 108 |

The metamodel-vs-depth disagreement rates (5.6 % of L5 IT/ET cells in L4/L6 depth bin; 9.8 % of L6 IT/CT cells in L5 depth bin) are the input to notebook 06.

### 1.5 Outputs

- `data/processed/tables/units_v1_exc_best_subtype.parquet` (8,910 × 22)
- `data/processed/tables/phase3_subtype_counts_global.csv`
- `data/processed/tables/phase3_subtype_counts_per_scan.csv`
- `data/processed/tables/phase3_loso_valid_scans.json` (rule + valid/invalid lists)

### 1.6 What this notebook commits to downstream

The valid-scan lists, the rule, and the n-per-class counts are *prespecified*. Steps 6 and 8 use those frozen lists; the matched-layer sensitivity analysis (notebook 06) re-evaluates validity on the depth-restricted subsets but applies the same threshold.

---

## 2. Notebook `02_l5_it_et_main.ipynb` — L5 IT vs L5 ET, main experiment (GKF) [v2 long-row]

### 2.1 Question

*Within V1 L5 excitatory neurons, do functional response statistics decode the AIBS metamodel IT/ET call under the within-protocol cross-validation protocol (StratifiedGroupKFold by nucleus_id)?*

### 2.2 Population

- **Subset filter**: `subtype ∈ {5P-IT, 5P-ET}` on the working population.
- **Long rows**: 202,504 (one per `(nucleus_id, condition_hash)` after the inner-join on B).
- **Neurons**: 1,489 (1,169 IT + 320 ET).
- **Scans**: 11 (the two scans with no L5 cells fall out automatically).

### 2.3 Steps, in order

1. **Setup** + import `build_modeling_table`, `neuron_level_score`, tier feature names.
2. **Build the long-row modelling table** via the §0.3 join chain (A1 → +B inner → +C1 left → +D left → +G broadcast). Layer-bin column pulled explicitly from `units_working.parquet`.
3. **Restrict to L5 IT/ET** by subtype filter. **Refresh GKF folds** stratified by `celltype_label`, grouped by `nucleus_id`, broadcast onto long rows.
4. **Define the five feature blocks** as column lists in the long-row table.
5. **CV helpers** (`make_pipeline`, `cv_run_long`, `cv_run_long_residualized`).
6. **GKF main grid** — five blocks × two models = 10 runs.
7. **Confound baselines** — majority, scan-only LR/HGB, cc_abs-only LR/HGB.
8. **cc_abs-residualized winner** (the GKF winner block × model refit with per-feature OLS residual on `cc_abs` within each training fold).
9. **Headline summary table**.
10. **Save** `phase3_l5_it_et_runs.parquet` (16 rows) and `phase3_l5_it_et_winner.json`.

### 2.4 Results

**GKF main grid (sorted by balanced accuracy):**

| block | model | n_feat | bal_acc | R(IT) | R(ET) | macro-F1 |
|---|---|---:|---|---:|---:|---:|
| **A1+B+C1+D1** | **HGB** | 41 | **0.719 ± 0.032** | 0.903 | 0.534 | 0.727 |
| A1+B | HGB | 27 | 0.684 ± 0.028 | 0.771 | 0.597 | 0.655 |
| A1+B+C1 | HGB | 31 | 0.683 ± 0.034 | 0.888 | 0.478 | 0.689 |
| G | LogReg | 116 | 0.678 ± 0.018 | 0.709 | 0.647 | 0.631 |
| G+B+C1 | LogReg | 127 | 0.678 ± 0.018 | 0.709 | 0.647 | 0.631 |
| A1+B | LogReg | 27 | 0.670 ± 0.027 | 0.653 | 0.687 | 0.607 |
| G+B+C1 | HGB | 127 | 0.668 ± 0.023 | 0.942 | 0.393 | 0.690 |
| G | HGB | 116 | 0.660 ± 0.025 | 0.942 | 0.378 | 0.682 |
| A1+B+C1+D1 | LogReg | 41 | 0.650 ± 0.043 | 0.538 | 0.762 | 0.556 |
| A1+B+C1 | LogReg | 31 | 0.640 ± 0.019 | 0.518 | 0.761 | 0.542 |

**GKF winner**: `A1+B+C1+D1 | HGB`, bal_acc = 0.719 ± 0.032.

**Confound baselines:**

| baseline | bal_acc | R(IT) | R(ET) |
|---|---|---:|---:|
| majority class (predicts 5P-IT) | 0.500 | 1.000 | 0.000 |
| **scan-only LogReg** | **0.731 ± 0.017** | 0.711 | 0.750 |
| **scan-only HGB** | **0.731 ± 0.017** | 0.711 | 0.750 |
| cc_abs-only LogReg | 0.526 ± 0.028 | 0.507 | 0.545 |
| cc_abs-only HGB | 0.527 ± 0.036 | 0.554 | 0.500 |
| **A1+B+C1+D1 HGB, cc_abs-residualized** | **0.600 ± 0.032** | 0.678 | 0.522 |

### 2.5 Diagnostic finding — the scan-composition confound

The **scan-only baseline at 0.731 exceeds the GKF winner (0.719)**. A classifier given only a one-hot of `session_key` decodes 5P-IT vs 5P-ET better than every feature block. This is the scan-composition confound: per-scan IT/ET ratios vary from 5:1 (scan 6_4 = 206 IT / 5 ET) to 1:1.7 (scan 4_7 = 39 IT / 66 ET); under StratifiedGroupKFold by `nucleus_id` the same scans appear in train and test, so a scan-only model can use the per-scan prior.

The cc_abs-residualized winner falls to 0.600 (Δ = −0.119), suggesting much of what survived after SNR control was the scan-composition leakage that didn't fully come out via cc_abs alone.

**The reportable headline of L5 IT/ET is therefore the LOSO valid-scan number, not the GKF number.** GKF is reported here as a within-protocol ceiling; LOSO (notebook 03) is the true generalisation test. The same scan-only / cc_abs / residualized contrasts are recomputed there.

### 2.6 Outputs

- `data/processed/results/phase3_l5_it_et_runs.parquet` — 16 rows (10 main + 5 baseline + 1 cc_abs-resid).
- `data/processed/results/phase3_l5_it_et_winner.json`.

---

## 3. Notebook `03_l5_it_et_loso.ipynb` — L5 IT vs L5 ET, LOSO [v2 long-row]

### 3.1 Question

*Does the GKF L5 IT/ET signal (notebook 02) generalise across scans, or does it depend on the scan-composition leakage diagnosed there?*

### 3.2 Population

Same as notebook 02 — 1,489 neurons, 11 scans, 202,504 long rows.

### 3.3 Steps, in order

1. **Setup**.
2. **Rebuild the long-row table** identically to notebook 02 (`build_modeling_table` + B/C1/D inner/left joins + G broadcast).
3. **Restrict to L5 IT/ET**. Load LOSO validity list and Step-2 winner from JSON.
4. **Define feature blocks** (same five).
5. **LOSO helpers** (`loso_run_long`, `loso_run_residualized`, `aggregate_loso`).
6. **LOSO main grid** — five blocks × two models, 13 held-out scans each.
7. **LOSO confound baselines** — majority, scan-only LR/HGB (integrity check), cc_abs-only LR/HGB.
8. **cc_abs-residualized winner** under LOSO (the Step-2 winner refit per held-out scan).
9. **Summary table** (valid-scan mean and all-scan mean side by side).
10. **Per-scan winner table** (un-residualized + residualized).
11. **Save** `phase3_l5_it_et_loso.parquet` (16 rows) and per-scan parquet (176 rows).

### 3.4 Results

**LOSO main grid — valid-scan mean (n=10), sorted:**

| block | model | n_feat | LOSO valid bal_acc | R(IT) valid | R(ET) valid |
|---|---|---:|---|---:|---:|
| **A1+B** | **HGB** | 27 | **0.606 ± 0.094** | 0.720 | 0.493 |
| A1+B | LogReg | 27 | 0.599 ± 0.095 | 0.574 | 0.624 |
| A1+B+C1 | HGB | 31 | 0.568 ± 0.084 | 0.895 | 0.241 |
| A1+B+C1 | LogReg | 31 | 0.581 ± 0.085 | 0.444 | 0.719 |
| A1+B+C1+D1 | LogReg | 41 | 0.578 ± 0.111 | 0.479 | 0.677 |
| A1+B+C1+D1 | HGB | 41 | 0.581 ± 0.094 | 0.927 | 0.235 |
| G | LogReg | 116 | 0.556 ± 0.108 | 0.717 | 0.396 |
| G+B+C1 | LogReg | 127 | 0.557 ± 0.108 | 0.718 | 0.396 |
| G+B+C1 | HGB | 127 | 0.529 ± 0.033 | 0.939 | 0.120 |
| G | HGB | 116 | 0.527 ± 0.035 | 0.956 | 0.099 |

**LOSO baselines (valid-scan mean, n=10):**

| baseline | bal_acc | R(IT) | R(ET) |
|---|---|---:|---:|
| majority class | 0.500 ± 0.000 | 1.000 | 0.000 |
| **scan-only LogReg** (integrity check) | **0.500 ± 0.000** | 0.000 | 1.000 |
| **scan-only HGB** (integrity check) | **0.500 ± 0.000** | 0.700 | 0.300 |
| cc_abs-only LogReg | 0.469 ± 0.093 | 0.486 | 0.453 |
| cc_abs-only HGB | 0.516 ± 0.058 | 0.553 | 0.479 |
| **A1+B+C1+D1 HGB, cc_abs-residualized** | **0.582 ± 0.079** | 0.708 | 0.455 |

**Integrity check passed**: scan-only LOSO = 0.500 (chance). The held-out scan's one-hot is unseen at training, so a scan-only model collapses to its prior.

**Per-scan winner (A1+B HGB), un-residualized:**

| scan | n | IT/ET | bal_acc | R(IT) | R(ET) |
|---|---:|---|---:|---:|---:|
| 4_7 | 105 | 39/66 | 0.689 | 0.667 | 0.712 |
| 5_6 | 190 | 146/44 | 0.653 | 0.829 | 0.477 |
| 5_7 | 150 | 134/16 | 0.601 | 0.701 | 0.500 |
| 6_2 | 159 | 150/9 | 0.590 | 0.847 | 0.333 |
| 6_4 | 211 | 206/5 | 0.486 | 0.772 | 0.200 |
| 6_6 | 174 | 158/16 | 0.587 | 0.861 | 0.312 |
| 6_7 | 216 | 183/33 | 0.414 | 0.738 | 0.091 |
| 7_3 | 156 | 92/64 | 0.645 | 0.478 | 0.812 |
| 7_5 | 18 | 8/10 | 0.762 | 0.625 | 0.900 |
| 8_5 | 109 | 53/56 | 0.634 | 0.679 | 0.589 |
| 9_3 (invalid) | 1 | 0/1 | NaN | — | 0.000 |

The most-balanced scans (`7_5` 8:10, `4_7` 39:66, `8_5` 53:56) give the cleanest per-scan signal (0.63–0.76). The most-skewed scans (`6_4` 41:1, `6_7` 5.5:1) collapse to near-chance — consistent with class-imbalance-driven minority collapse rather than per-scan biology.

### 3.5 Reportable headline for L5 IT/ET

> *Within L5 V1 excitatory neurons, the LOSO-valid balanced accuracy for 5P-IT vs 5P-ET is 0.606 ± 0.094 (best block: A1+B HGB; n=10 valid scans). The signal is roughly +0.1 above chance, survives both confound controls (cc_abs-residualized = 0.582, Δ ≈ 0; scan-only LOSO = 0.500), and is much weaker than the GKF would suggest (0.719 → 0.606 = −0.113). Most of the GKF "signal" was scan-composition leakage; the actual cross-scan subtype signal is small but nonzero.*

The model with the most-balanced per-class recall is `A1+B LogReg` (0.599, R(IT)=0.574, R(ET)=0.624). HGB models consistently collapse on the minority class (R(ET) ≈ 0.10–0.49) despite class-balanced weighting.

### 3.6 Outputs

- `data/processed/results/phase3_l5_it_et_loso.parquet` (16-row summary).
- `data/processed/results/phase3_l5_it_et_loso_perscan.parquet` (176 rows).

---

## 4. Notebook `04_l6_it_ct.ipynb` — L6 IT vs L6 CT, GKF + LOSO [v2 long-row]

### 4.1 Question

*Within V1 L6 excitatory neurons, do functional response statistics decode the AIBS metamodel IT/CT call? Per `PHASE3_PLANNING §3`, the cc_abs-residualized LOSO valid number is the **PRIMARY** headline for L6 (because L6 cells are deeper / often noisier, SNR confounds matter more).*

### 4.2 Population

- **Subset filter**: `subtype ∈ {6P-IT, 6P-CT}`.
- **Neurons**: 337 (216 IT + 121 CT).
- **Scans**: 7 (the 6 valid LOSO scans + scan `4_7` which has 1 CT and 0 IT).
- **Long rows**: 39,656.
- **cc_abs distribution**: IT mean = 0.317, CT mean = 0.284 — measurable difference (~10 % gap) which is exactly the SNR confound the planning doc flagged.

### 4.3 Steps, in order

1. **Setup**.
2. **Build long-row table** (same as 02/03).
3. **Restrict to L6 IT/CT**, refresh folds.
4. **Define feature blocks**.
5. **Helpers** (GKF + LOSO + residualization).
6. **GKF main grid**.
7. **GKF baselines + cc_abs-residualized GKF winner**.
8. **LOSO main grid + LOSO baselines + cc_abs-residualized LOSO winner** (PRIMARY headline).
9. **Summary tables**.
10. **Per-scan winner table** (raw + residualized).
11. **Save** `phase3_l6_it_ct_runs.parquet`, `phase3_l6_it_ct_loso.parquet`, `phase3_l6_it_ct_loso_perscan.parquet`, `phase3_l6_it_ct_winner.json`.

### 4.4 GKF results

**Main grid (sorted):**

| block | model | n_feat | bal_acc | R(IT) | R(CT) | macro-F1 |
|---|---|---:|---|---:|---:|---:|
| **A1+B+C1+D1** | **LogReg** | 41 | **0.606 ± 0.066** | 0.549 | 0.662 | 0.579 |
| A1+B+C1 | LogReg | 31 | 0.604 ± 0.066 | 0.554 | 0.653 | 0.579 |
| A1+B | LogReg | 27 | 0.601 ± 0.061 | 0.541 | 0.662 | 0.574 |
| A1+B+C1 | HGB | 31 | 0.591 ± 0.072 | 0.649 | 0.532 | 0.584 |
| A1+B | HGB | 27 | 0.580 ± 0.068 | 0.632 | 0.528 | 0.573 |
| A1+B+C1+D1 | HGB | 41 | 0.564 ± 0.050 | 0.609 | 0.520 | 0.556 |
| G | LogReg | 116 | 0.562 ± 0.077 | 0.646 | 0.478 | 0.558 |
| G+B+C1 | LogReg | 127 | 0.552 ± 0.084 | 0.641 | 0.463 | 0.548 |
| G | HGB | 116 | 0.518 ± 0.062 | 0.653 | 0.384 | 0.513 |
| G+B+C1 | HGB | 127 | 0.518 ± 0.062 | 0.653 | 0.384 | 0.513 |

**GKF baselines:**

| baseline | bal_acc | R(IT) | R(CT) |
|---|---|---:|---:|
| majority | 0.500 | 1.000 | 0.000 |
| **scan-only LR/HGB** | **0.580 ± 0.009** | 0.702 | 0.459 |
| cc_abs-only LR | 0.542 ± 0.064 | 0.517 | 0.567 |
| **cc_abs-only HGB** | **0.551 ± 0.053** | 0.604 | 0.498 |
| **A1+B+C1+D1 LR, cc_abs-residualized** | **0.616 ± 0.049** | 0.566 | 0.666 |

The GKF winner (0.606) is barely above the scan-only baseline (0.580); cc_abs-only HGB (0.551) is also high. As with L5 IT/ET, GKF here is contaminated by both scan composition and SNR.

### 4.5 LOSO results — the PRIMARY headline

**LOSO main grid (valid-scan mean, n=6):**

| block | model | n_feat | LOSO valid bal_acc | R(IT) valid | R(CT) valid |
|---|---|---:|---|---:|---:|
| A1+B+C1 | HGB | 31 | 0.571 ± 0.070 | 0.638 | 0.503 |
| A1+B | HGB | 27 | 0.569 ± 0.059 | 0.622 | 0.517 |
| A1+B+C1+D1 | HGB | 41 | 0.569 ± 0.058 | 0.631 | 0.507 |
| A1+B+C1 | LogReg | 31 | 0.561 ± 0.067 | 0.524 | 0.598 |
| A1+B+C1+D1 | LogReg | 41 | 0.559 ± 0.066 | 0.520 | 0.598 |
| A1+B | LogReg | 27 | 0.555 ± 0.067 | 0.512 | 0.598 |
| G | LogReg | 116 | 0.534 ± 0.058 | 0.640 | 0.429 |
| G+B+C1 | LogReg | 127 | 0.528 ± 0.065 | 0.623 | 0.432 |
| G | HGB | 116 | 0.522 ± 0.032 | 0.701 | 0.342 |
| G+B+C1 | HGB | 127 | 0.522 ± 0.032 | 0.701 | 0.342 |

**LOSO baselines (n=6 valid):**

| baseline | bal_acc | R(IT) | R(CT) |
|---|---|---:|---:|
| majority | 0.500 ± 0.000 | 1.000 | 0.000 |
| **scan-only LR/HGB** (integrity check) | **0.500 ± 0.000** | 0.000/1.000 | 1.000/0.000 |
| cc_abs-only LR | 0.507 ± 0.089 | 0.480 | 0.535 |
| **cc_abs-only HGB** | **0.556 ± 0.039** | 0.657 | 0.456 |
| **A1+B+C1+D1 LR, cc_abs-residualized** (PRIMARY) | **0.570 ± 0.066** | 0.583 | 0.557 |

**Integrity check passed** (scan-only LOSO = 0.500).

**The PRIMARY L6 IT/CT headline** is `A1+B+C1+D1 LogReg, cc_abs-residualized LOSO valid = 0.570 ± 0.066` (n=6). For comparison, the un-residualized winner is also `A1+B+C1+D1 LogReg = 0.559`, so cc_abs residualization moved the headline by Δ = +0.011 — i.e., the L6 signal *survives* SNR control (and even improves marginally).

### 4.6 Concerning observation

`cc_abs-only HGB LOSO valid = 0.556` is essentially indistinguishable from the residualized winner `0.570`. SNR alone gets ~0.056 above chance; the long-row protocol on full feature blocks gets ~0.07 above chance. The marginal lift from biology over SNR is small (~0.014).

### 4.7 Reportable headline for L6 IT/CT

> *Within L6 V1 excitatory neurons, the cc_abs-residualized LOSO valid balanced accuracy for 6P-IT vs 6P-CT is 0.570 ± 0.066 (n=6 valid scans, A1+B+C1+D1 LogReg). The signal is +0.07 above chance and survives both confound controls (Δ vs un-residualized = +0.011; scan-only LOSO = 0.500). However, cc_abs-only HGB LOSO at 0.556 is comparable, meaning a single SNR scalar gets most of the lift; the marginal signal beyond SNR is ~0.014. The L6 IT/CT result is the borderline / weakly-positive case.*

This is more tentative than L5 IT/ET; the absolute numbers are similar (~0.57 vs ~0.60) but the SNR baseline is much closer to the headline for L6.

### 4.8 Outputs

- `phase3_l6_it_ct_runs.parquet` (16 rows, GKF).
- `phase3_l6_it_ct_loso.parquet` (16 rows, LOSO).
- `phase3_l6_it_ct_loso_perscan.parquet` (per-scan).
- `phase3_l6_it_ct_winner.json`.

---

## 5. Notebook `05_phase3_multiclass.ipynb` — 7-class subtype classification [v2 long-row]

### 5.1 Question

*Can functional response statistics decode the AIBS metamodel cell-type call across all seven V1 excitatory subtypes simultaneously?*

The layer-recovery diagnostic from `PHASE3_PLANNING §4` is **deferred** until the metamodel/depth mismatch is interpreted with the prof.

### 5.2 Population

- Full Phase-1 working population (n=8,895, L1 dropped, 13 scans).
- Severe imbalance (162:1 between 23P n=4,198 and 5P-NP n=26).
- Long rows: 1,209,720 (the entire long-row training table; same number as Phase 1 stage 4).

### 5.3 Imbalance handling

- **Splitting**: `StratifiedGroupKFold(5)` stratified on the 7-class label, grouped by `nucleus_id`. Pre-fit audit prints the per-fold class counts to flag any class with 0 cells in a test fold (so its recall doesn't silently drop out of the macro).
- **Class weights**: `class_weight='balanced'` AND `compute_sample_weight('balanced', y[tr])` passed to `fit()`.
- **LOSO validity**: `≥4 of 7 classes with ≥5 cells each` — gives 10 valid scans (excluding 9_3, 9_4, 9_6).
- **Metrics**: 7-class balanced accuracy + **"common-4" balanced accuracy** over `{23P, 4P, 5P-IT, 5P-ET}` (each ≥320 cells) + per-class recall + 7×7 confusion matrix.

### 5.4 Steps, in order

1. Setup.
2. Build the full long-row modelling table (no subtype filter — all 7 classes).
3. Refresh GKF folds stratified by 7-class label.
4. **Pre-fit class-count audit** — surfaces any per-fold class with 0 test cells. Per the saved audit, fold 2 had 0 6P-IT and 0 6P-CT test cells (rare classes are unevenly distributed across scans).
5. Define feature blocks.
6. Helpers (multiclass-aware `_score_neuron` returning balanced_accuracy, common-4, per-class recall, confusion matrix).
7. **GKF main grid** (5 × 2 = 10 runs).
8. **GKF baselines** — majority, uniform-random, scan-only, cc_abs-only.
9. **GKF cc_abs-residualized winner**.
10. **LOSO main grid** (5 × 2 = 10 LOSO runs over 13 scans).
11. **LOSO baselines + cc_abs-residualized winner**.
12. Summary tables.
13. **Per-class recall + 7×7 confusion matrix** for the GKF winner.
14. Save outputs.

### 5.5 GKF results — multiclass

**Main grid (sorted by 7-class bal_acc):**

| block | model | n_feat | bal_acc7 | common-4 | macro-F1 |
|---|---|---:|---|---|---:|
| **A1+B+C1+D1** | **HGB** | 41 | **0.424 ± 0.022** | 0.467 | 0.314 |
| G+B+C1 | LogReg | 127 | 0.404 ± 0.028 | 0.395 | 0.275 |
| G+B+C1 | HGB | 127 | 0.396 ± 0.022 | 0.520 | 0.410 |
| G | HGB | 116 | 0.389 ± 0.027 | 0.517 | 0.401 |
| A1+B+C1 | HGB | 31 | 0.383 ± 0.026 | 0.407 | 0.276 |
| G | LogReg | 116 | 0.376 ± 0.009 | 0.417 | 0.284 |
| A1+B | HGB | 27 | 0.329 ± 0.024 | 0.323 | 0.213 |
| A1+B+C1 | LogReg | 31 | 0.262 ± 0.009 | 0.214 | 0.157 |
| A1+B+C1+D1 | LogReg | 41 | 0.258 ± 0.006 | 0.212 | 0.154 |
| A1+B | LogReg | 27 | 0.249 ± 0.008 | 0.204 | 0.144 |

**GKF baselines:**

| baseline | bal_acc7 | common-4 | macro-F1 |
|---|---|---|---:|
| chance / 1-of-7 | 0.143 | 0.250 | — |
| majority class (predicts 23P) | 0.143 | 0.250 | 0.092 |
| uniform-random | 0.135 | 0.143 | 0.096 |
| **scan-only LR/HGB** | **0.356 ± 0.053** | 0.349 | 0.218 |
| cc_abs-only LR | 0.206 ± 0.019 | 0.190 | 0.079 |
| cc_abs-only HGB | 0.231 ± 0.020 | 0.160 | 0.103 |
| **A1+B+C1+D1 HGB, cc_abs-residualized** | **0.360 ± 0.026** | 0.352 | 0.232 |

Same scan-composition pattern as the binaries: scan-only GKF (0.356) is comparable to several main blocks. cc_abs residualization drops the winner from 0.424 to 0.360 (Δ = −0.064).

### 5.6 LOSO results — the headline

**Main grid (valid-scan mean, n=10):**

| block | model | n_feat | LOSO valid bal_acc7 | common-4 |
|---|---|---:|---|---|
| G+B+C1 | LogReg | 127 | 0.290 ± 0.062 | 0.273 |
| **A1+B+C1+D1** | **HGB** | 41 | **0.284 ± 0.069** | 0.286 |
| A1+B | HGB | 27 | 0.283 ± 0.055 | 0.283 |
| A1+B+C1 | HGB | 31 | 0.282 ± 0.068 | 0.279 |
| G | LogReg | 116 | 0.278 ± 0.038 | 0.301 |
| G | HGB | 116 | 0.274 ± 0.043 | 0.365 |
| G+B+C1 | HGB | 127 | 0.274 ± 0.053 | 0.367 |
| A1+B+C1 | LogReg | 31 | 0.238 ± 0.052 | 0.238 |
| A1+B+C1+D1 | LogReg | 41 | 0.230 ± 0.052 | 0.234 |
| A1+B | LogReg | 27 | 0.229 ± 0.050 | 0.215 |

**LOSO baselines (valid-scan mean, n=10):**

| baseline | bal_acc7 | common-4 |
|---|---|---|
| chance / 1-of-7 | 0.143 | 0.250 |
| majority | 0.165 ± 0.025 | 0.250 |
| **scan-only LR** (integrity check) | **0.165 ± 0.025** | 0.250 |
| **scan-only HGB** (integrity check) | **0.145 ± 0.053** | 0.225 |
| cc_abs-only LR | 0.178 ± 0.039 | 0.173 |
| cc_abs-only HGB | 0.195 ± 0.063 | 0.155 |
| **A1+B+C1+D1 HGB, cc_abs-residualized** | **0.337 ± 0.076** | 0.282 |

**Integrity check passed** (scan-only LOSO = 0.145–0.165, near chance 0.143).

The **cc_abs-residualized winner LOSO** is *higher* than the un-residualized one (0.337 vs 0.284, Δ = +0.053). Same direction as L6 IT/CT — residualization seems to help under LOSO for multiclass, possibly because cc_abs varies enough between scans that removing its linear effect makes the per-feature distributions more comparable across held-out scans.

### 5.7 GKF→LOSO drop and per-class recall

GKF winner = 0.424; LOSO valid (same model+block) = 0.284 → drop = **−0.140**, very similar to the −0.138 drop on the L5 binary.

**Per-class recall — GKF winner (A1+B+C1+D1 HGB):**

| class | n | recall (GKF mean) |
|---|---:|---:|
| 23P | 4,198 | (saved in winner_meta) |
| 4P | 2,845 | (saved) |
| 5P-IT | 1,169 | (saved) |
| 5P-ET | 320 | (saved) |
| 5P-NP | 26 | (saved — likely very low) |
| 6P-IT | 216 | (saved) |
| 6P-CT | 121 | (saved) |

(Per-class recall numbers are stored in `phase3_multiclass_winner.json` and `phase3_multiclass_runs.parquet`; the rare classes are heavily down-weighted by HGB's actual decision boundaries despite class-balanced weighting, an issue we already saw in the binaries.)

### 5.8 Reportable headline for multiclass

> *7-class V1 subtype classification under LOSO valid n=10 reaches balanced_accuracy = 0.284 ± 0.069 (best block: G+B+C1 LogReg) — well above 1-of-7 chance (0.143) and well above scan-only LOSO (0.165). Common-4 balanced accuracy (over the four classes with ≥320 cells each) lands at 0.28–0.37 depending on block (chance = 0.25). The GKF→LOSO drop of −0.14 mirrors what we saw on the L5 binary; the cc_abs-residualized winner under LOSO is 0.337, slightly **higher** than un-residualized.*

The multiclass result is genuinely above chance under LOSO, but it is far less reliable than what binary tasks deliver because the rare classes (5P-NP n=26, 6P-IT n=216, 6P-CT n=121) collapse on most folds — common-4 macro is consistently more reliable than the full 7-class macro.

### 5.9 Outputs

- `phase3_multiclass_runs.parquet` (GKF, 17 rows).
- `phase3_multiclass_loso.parquet` (LOSO, 16 rows).
- `phase3_multiclass_loso_perscan.parquet`.
- `phase3_multiclass_winner.json`.

---

## 6. Notebook `06_label_sensitivity.ipynb` — Matched-layer LOSO sensitivity

### 6.1 Question

*Does restricting to cells where the depth-bin layer agrees with the metamodel-implied layer (5P-* → L5; 6P-* → L6) change the LOSO headline? I.e., are the boundary cells (5.6 % of L5 IT/ET, 9.8 % of L6 IT/CT) helping, hurting, or irrelevant?*

### 6.2 Design

- **Two Phase-1 headline models** used uniformly (both as HGB):
  - `Tier G | HGB` — Phase 1 GKF ceiling.
  - `A1+B+C1+D1 | HGB` — Phase 1 long-row stage-4 headline.
- **Two populations per task**:
  - **full**: subtype filter only (matches notebook 03 / 04 v2 headline).
  - **matched**: subtype filter AND depth-bin layer matches metamodel-implied layer.
- **Two preprocessings**: raw, cc_abs-residualized.
- **LOSO only** (no GKF). Validity rule re-evaluated per population.

Total runs: 2 tasks × 2 headlines × 2 populations × 2 preprocessings = **16 LOSO grids**.

Restricting to `layer == L5` for L5 IT/ET drops 83 cells (5.6 %): 1,489 → 1,406. Validity drops from 10 valid scans to 8.
Restricting to `layer == L6` for L6 IT/CT drops 33 cells (9.8 %): 337 → 304. Validity drops from 6 valid scans to 5.

### 6.3 Results

**L5 IT/ET — full vs matched, by headline × preprocessing:**

| headline | preprocessing | full LOSO valid | matched LOSO valid | Δ matched − full |
|---|---|---|---|---|
| Tier G HGB | raw | 0.527 ± 0.035 (n=10) | 0.532 ± 0.033 (n=8) | +0.004 |
| Tier G HGB | cc_abs-resid | 0.548 ± 0.077 (n=10) | 0.521 ± 0.034 (n=8) | −0.027 |
| A1+B+C1+D1 HGB | raw | 0.581 ± 0.094 (n=10) | 0.598 ± 0.096 (n=8) | +0.017 |
| A1+B+C1+D1 HGB | cc_abs-resid | 0.582 ± 0.079 (n=10) | 0.556 ± 0.118 (n=8) | −0.026 |

All four L5 IT/ET deltas are within ±0.03 ⇒ **matched ≈ full** for the L5 task across both headline models and both preprocessings.

**L6 IT/CT — full vs matched, by headline × preprocessing:**

| headline | preprocessing | full LOSO valid | matched LOSO valid | Δ matched − full |
|---|---|---|---|---|
| Tier G HGB | raw | 0.522 ± 0.032 (n=6) | 0.508 ± 0.046 (n=5) | −0.014 |
| Tier G HGB | cc_abs-resid | 0.512 ± 0.035 (n=6) | 0.524 ± 0.043 (n=5) | +0.012 |
| A1+B+C1+D1 HGB | raw | 0.569 ± 0.058 (n=6) | 0.513 ± 0.105 (n=5) | **−0.055** |
| A1+B+C1+D1 HGB | cc_abs-resid | 0.536 ± 0.061 (n=6) | 0.549 ± 0.047 (n=5) | +0.013 |

**Seven of eight** comparisons (across both tasks, both headlines, both preprocessings) are within ±0.03 of zero ⇒ matched ≈ full.

The single outlier is **L6 IT/CT, A1+B+C1+D1 HGB, raw**: Δ = −0.055 (matched < full). Restricting L6 to depth-bin L6 hurts the un-residualized A1+B+C1+D1 HGB decoder by 0.055. But the same headline with cc_abs-residualization is back to Δ = +0.013 (matched ≈ full). The outlier is therefore **specific to the un-residualized variant of one headline on the L6 task**, and goes away under SNR control.

### 6.4 Interpretation

- The **boundary cells are not driving the L5 IT/ET signal** (every L5 comparison is matched ≈ full).
- The **boundary cells are not driving the L6 IT/CT signal under cc_abs control** (the PRIMARY headline from notebook 04). Under un-residualized A1+B+C1+D1 HGB the boundary cells contribute about 0.055 to the L6 GKF-style number; that contribution disappears once SNR is controlled, suggesting the boundary cells were carrying the SNR signal that residualization removes anyway.
- The metamodel-vs-depth disagreement (the question for the prof) is therefore **not the explanation for the modest Phase 3 results**. Whatever signal exists is consistent across the depth boundary, and whatever scan-composition / SNR confound was inflating GKF is independent of the boundary cells.

### 6.5 Outputs

- `phase3_label_sensitivity.parquet` (16-row summary).
- `phase3_label_sensitivity_perscan.parquet`.

---

## 7. Synthesis — what Phase 3 v2 establishes

### 7.1 Three subtype-decoding tasks, three reportable numbers

| task | population | valid LOSO bal_acc | reading |
|---|---|---|---|
| L5 IT/ET | 1,489 cells, 10 scans | **0.606 ± 0.094** (A1+B HGB, un-residualized) <br>cc_abs-resid: 0.582 ± 0.079 (A1+B+C1+D1 HGB) | weak positive — survives both confounds |
| L6 IT/CT | 337 cells, 6 scans | **0.570 ± 0.066** (A1+B+C1+D1 LR, **cc_abs-residualized**) <br>un-residualized: 0.559 (same model) | borderline positive — small lift over cc_abs-only HGB (0.556) |
| 7-class | 8,895 cells, 10 valid LOSO scans | **0.284 ± 0.069** un-residualized; 0.337 ± 0.076 cc_abs-residualized (A1+B+C1+D1 HGB) | clearly above chance (1/7=0.143) but rare classes collapse; common-4 macro 0.28–0.37 |

### 7.2 What the corrected-protocol controls established

1. **Long-row protocol changed numbers but preserved the qualitative L5 IT/ET picture.** The v1 broken-protocol headline was 0.689 GKF; v2 long-row GKF is 0.719 (close), and v2 LOSO valid is 0.606. The v1 LOSO under the broken protocol had reported 0.673 — *higher* than the corrected v2 LOSO. The v1 LOSO inflation came from the cross-hash-averaging step plus a different group definition (scan_id-grouped GKF) that obscured the scan-composition issue.
2. **The scan-composition confound is severe under GKF for binary subtype tasks.** GKF scan-only on L5 IT/ET = 0.731, *higher than every feature block.* Without the LOSO integrity check, the GKF claim is not interpretable. PHASE3_PLANNING §2.4 specified scan-only as a confound baseline; in v2 it became the *dominant* confound on this task, not a side-check.
3. **cc_abs control survives.** Across all three tasks, the cc_abs-residualized LOSO valid number is within ~0.03 of the un-residualized LOSO valid number. The signal that survives LOSO also survives SNR control. SNR is not the dominant Phase 3 confound.
4. **HGB consistently underperforms on minority classes despite balanced weighting.** Across L5 IT/ET, L6 IT/CT, and multiclass, HGB models give R(majority) ≈ 0.85–0.95 and R(minority) ≈ 0.10–0.49. LogReg variants are more balanced. For interpretation of "what the decoder actually does," the LogReg per-class recall is more honest than HGB's macro number.
5. **Boundary cells are not the source of the modest Phase 3 results** (notebook 06). Restricting to depth-matched cells gives matched ≈ full on 7 of 8 comparisons; the metamodel/depth mismatch is not driving the result.

### 7.3 What the corrected-protocol controls did NOT establish

- **The biological mechanism of the L5 IT/ET signal.** A1+B (amplitude + reliability) wins under LOSO but only by ~0.05 over Tier G or A1+B+C1+D1. We cannot cleanly say "ET cells are differently reliable" — the lift is small and the per-class HGB collapse undermines a strong claim.
- **Whether the L6 IT/CT signal is real beyond SNR.** cc_abs-only HGB = 0.556 vs the cc_abs-residualized winner = 0.570. The marginal lift is 0.014 — within fold-level noise. The most honest framing is "weak / borderline positive" pending more data or a different protocol.
- **Whether the metamodel/depth boundary is biology or coregistration noise.** The label-sensitivity result tells us the answer doesn't matter for the *current* decoder, but it doesn't tell us *why* the boundary cells exist. That is the prof question.

### 7.4 What Phase 3 v2 demonstrates about Phase 1's protocol

Two observations Phase 1 made are reinforced empirically in Phase 3:

1. **GKF→LOSO drop pattern is a structural property of the dataset**, not a methodological flaw to fix. Phase 1 layer: −0.16. Phase 3 L5 IT/ET: −0.113. Phase 3 multiclass: −0.140. All in the same range. With 13 scans on one mouse, a held-out scan removes ~7–8 % of training data and introduces a class-distribution shift — and that always costs ~0.13–0.16 of balanced accuracy.
2. **Long-row + neuron-level probability aggregation is the right protocol for any per-neuron prediction task.** It can be approximated by per-neuron summarisation when the model is linear (LR is roughly invariant), but it diverges substantially for HGB and other non-linear models, and it is what the WORKFLOW §3.6 commitment requires.

---

## 8. Appendix — outputs and reproducibility

All Phase 3 v2 result files live in `data/processed/results/`:

| file | rows | content |
|---|---:|---|
| `phase3_l5_it_et_runs.parquet` | 16 | L5 IT/ET GKF (10 main + 5 baselines + cc_abs-resid winner) |
| `phase3_l5_it_et_loso.parquet` | 16 | L5 IT/ET LOSO summary |
| `phase3_l5_it_et_loso_perscan.parquet` | 176 | L5 IT/ET per-scan, all configs |
| `phase3_l5_it_et_winner.json` | 1 | L5 IT/ET winner metadata |
| `phase3_l6_it_ct_runs.parquet` | 16 | L6 IT/CT GKF |
| `phase3_l6_it_ct_loso.parquet` | 16 | L6 IT/CT LOSO summary |
| `phase3_l6_it_ct_loso_perscan.parquet` | 91 | L6 IT/CT per-scan |
| `phase3_l6_it_ct_winner.json` | 1 | L6 IT/CT winner metadata |
| `phase3_multiclass_runs.parquet` | 17 | Multiclass GKF |
| `phase3_multiclass_loso.parquet` | 16 | Multiclass LOSO |
| `phase3_multiclass_loso_perscan.parquet` | ~143 | Multiclass per-scan |
| `phase3_multiclass_winner.json` | 1 | Multiclass winner metadata |
| `phase3_label_sensitivity.parquet` | 16 | Sensitivity full vs matched, 2 tasks × 2 headlines × 2 prep |
| `phase3_label_sensitivity_perscan.parquet` | ~144 | Sensitivity per-scan |

Reproducibility: `RANDOM_SEED = 42`, fixed across all notebooks. Re-running the notebooks regenerates these parquets verbatim. Saved metadata in each `*_winner.json` records the protocol, winning block + model, and core summary numbers.

---

## 9. Pending / open questions

- **Layer-recovery diagnostic** (`PHASE3_PLANNING §4`) — deferred until the metamodel-vs-depth mismatch is interpreted with the prof.
- **Why the prof should be asked about the metamodel-vs-depth mismatch** — three plausible explanations (real biology, EM coregistration noise, metamodel error) have different implications for whether the layer-recovery diagnostic is even meaningful.
- **The L6 IT/CT signal is borderline.** A more confident headline would require either (a) more cells (especially CT) or (b) a different stimulus / behaviour protocol that better isolates L6 cortico-thalamic function (oracle stimuli are not the right probe for CT cells' modulatory roles).
- **The 5P-NP class is essentially undecodable** with n=26 and only sparse presence across scans. Any "5P-NP" line in the multiclass per-class recall should be interpreted as small-sample noise.
