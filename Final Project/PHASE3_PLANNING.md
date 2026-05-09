# Phase 3 — Planning Document

*This document is the rigorous bridge from Phase 1 (layer decoding) to Phase 3 (cell-type / subtype decoding within V1 excitatory neurons). It is written before any Phase-3 code so that every design decision is justified against what Phase 1 established. Phase 2 (raw-trace CNN) is paused; Phase 3 is run independently on the same engineered features and the same canonical unit population. The companion document `PHASE1_GROUND_TRUTH.md` is the empirical record that Phase 3 must respect.*

---

## 0. The framing claim Phase 3 is allowed to make

Phase 3 asks **whether functional response statistics carry information about excitatory subtype identity, within layer**. The headline question is L5 IT vs L5 ET — the cleanest, most biologically meaningful binary in V1 excitatory neurons. Secondary tests are L6 IT vs L6 CT and full multiclass subtype recovery.

The conservative framing is:

- If L5 IT/ET works, we are allowed to say: *"Within L5, functional response statistics contain information associated with IT/ET identity."* We are **not** allowed to say *"we decoded genetic / morphological type from function."* The IT/ET subtype call comes from the AIBS metamodel and is an anatomical/transcriptomic prior; what we measure is whether *function*, as captured in our oracle stimulus protocol, is *consistent* with that prior.
- If L5 IT/ET fails, the result is also meaningful: L5 subtype identity may be more anatomical/transcriptomic than visible in this calcium / stimulus protocol, or the relevant differences may require stimuli / behaviours not present in the dataset.
- Either way, the likely biological axes — should the decoder succeed — are behavioural-state modulation, gain, response reliability, and possibly temporal integration. The feature-block ablation (§3) is what discriminates among these.

Anything stronger requires direct anatomical or transcriptomic measurement we do not have.

---

## 1. Population definition (Step 1)

The Phase 3 unit population is the **same** as the Phase 1 headline population: V1, excitatory, oracle-matched, best unit per neuron. This is the file `data/processed/tables/units_v1_exc_best.parquet` (n = 8,910). We do not re-derive this; we extend it.

Subtype labels come from `aibs_metamodel_celltypes_v661.csv` with `aibs_metamodel_celltypes_v661_corrections.csv` applied. The corrections-applied call is **already present** in the `cell_type` column of `units_v1_exc_best.parquet` (verified by 100 % agreement against the corrections-applied join on `pt_root_id`). The seven subtype labels in scope are:

`23P, 4P, 5P-IT, 5P-ET, 5P-NP, 6P-IT, 6P-CT`

Step 1 publishes:
1. **Global counts** of each subtype in the Phase 3 population.
2. **Per-scan crosstab** of `scan_id × subtype`.
3. **LOSO validity lists** for the L5 IT/ET and L6 IT/CT binaries (rule below).
4. The extended unit table `data/processed/tables/units_v1_exc_best_subtype.parquet` with helper columns (`scan_id`, `subtype`, `is_L5_IT`, `is_L5_ET`, `is_L6_IT`, `is_L6_CT`).

**Note on layer vs subtype.** The `layer` column in the Phase 1 table is a depth bin computed from `pt_position_y`. The `cell_type` column is the AIBS metamodel call. They mostly agree but disagree at layer boundaries (e.g. some L4-binned cells are called `5P-IT` and vice versa). For Phase 3 we use **the metamodel `cell_type`** to define groups, not the depth bin — the metamodel call is the one that encodes the IT/ET/CT/NP distinction we care about. The depth bin is reported only as a sanity covariate.

---

## 2. Headline experiment: L5 IT vs L5 ET (Step 2–7)

### 2.1 Group definition

`5P-IT` (n ≈ 1,169) vs `5P-ET` (n ≈ 320) — **all** cells with these `cell_type` calls, regardless of which depth bin they fell into. Filtering by `layer == L5` on top of the metamodel call would discard valid edge cases at the layer boundary. `5P-NP` is **excluded from this binary task only**, not from the dataset.

This is a binary, severely imbalanced (~3.7:1) task. Class weights must be balanced. Balanced accuracy is the headline metric, but we report per-class precision / recall and a confusion matrix alongside, because balanced accuracy can stay flat while a model collapses to majority.

### 2.2 Feature blocks (Step 3)

Five feature configurations, in increasing capacity, mirroring Phase 1:

| Block | What it is | Phase-1 origin |
|---|---|---|
| **G** | Tier G per-neuron fingerprint (oracle-stimulus tuning) | Phase 1 headline |
| **A1+B** | A1 amplitude + B reliability | Stage 2 |
| **A1+B+C1** | + C1 behaviour / gain context | Stage 3 |
| **A1+B+C1+D1** | + D1 stimulus-content interactions | Stage 4 |
| **G+B+C1** | hybrid: tuning fingerprint + reliability + behaviour | new in Phase 3 |

### 2.3 Models and protocol (Step 4)

Logistic regression first (interpretable coefficients), HGB second (capacity ceiling). Same preprocessing pipeline as Phase 1 stage 1 (`SimpleImputer(median) → RobustScaler → classifier`). `class_weight='balanced'` for both. `StratifiedGroupKFold(5)` grouped by `scan_id`. Neuron-level metric is the per-neuron probability with a single best row per neuron, no aggregation needed (we already have one row per neuron in the Phase 3 table).

### 2.4 Confound baselines (Step 5)

Five baselines reported alongside every block:

1. **Chance / majority** — predict majority class for everyone.
2. **Scan-only** — one-hot of `scan_id` only, no neuron features. Tests whether subtype is scan-confounded.
3. **`cc_abs`-only** — single-feature LR/HGB on response-strength scalar. Tests whether SNR alone gives the model.
4. **`cc_abs`-residualized winning block** — residualize each feature on `cc_abs` (and per-scan mean amplitude) before fitting. Stronger SNR-confound test.
5. **Session-zscored vs raw comparison** — refit the winning block with per-scan robust z-scoring on the input features.

A model that does not beat the `cc_abs`-only baseline by a meaningful margin is not a subtype model.

### 2.5 LOSO with prespecified validity (Step 6)

LOSO grouped by `scan_id`. **Validity rule, locked from Step 1 counts**: a held-out scan is valid for the L5 IT/ET LOSO if (a) both classes present and (b) minority class (ET) ≥ 5 cells. We compute this list **before** any LOSO run and report it in Step 1. Folds where the held-out scan is invalid are excluded from the LOSO mean and shown separately as "invalid-fold cells held out". The reportable LOSO is the **valid-scan mean ± std**, not the all-scan mean.

### 2.6 Interpretation by feature family (Step 7)

- If **Tier G wins**: subtype information is encoded mainly in the stimulus-specific tuning fingerprint.
- If **+C1 improves** L5 IT/ET: behaviour/gain modulation is a subtype marker. (This is the *most biologically interesting* outcome — L5 ET cells are known to be more behaviour-modulated than L5 IT in some preparations.)
- If **+B improves**: reliability / noise structure differs between IT and ET.
- If **only +D1 helps**: be careful — it may mean subtype-specific stimulus-content interactions, or it may reflect stimulus / session structure leaking through D1. We cross-check with the `cc_abs`-residualized variant.

---

## 3. L6 IT vs L6 CT (Step 8)

Same protocol as L5 IT/ET, with two changes:

- **Stronger `cc_abs` control.** L6 cells are deeper, often noisier; SNR confounds are more dangerous. We always report the `cc_abs`-residualized number as the *primary* L6 IT/CT result, not as a sensitivity analysis.
- **Tighter LOSO validity.** Per Step 1 per-scan counts, several scans have very few `6P-IT` or `6P-CT` cells. The same minority-≥-5-cells rule applies; we expect fewer valid LOSO scans than L5 IT/ET.

---

## 4. Multiclass subtype + layer-recovery diagnostic (Step 9)

Full 7-class subtype classification on the same population, with the same five feature blocks, but treated as **secondary**. The diagnostic value is in the layer-recovery test:

1. Take per-cell predicted subtype.
2. Map subtype back to layer (`23P → L2/3`, `4P → L4`, `5P-* → L5`, `6P-* → L6`).
3. Compare aggregated layer prediction against Phase 1 layer-decoder performance on the same population.

Three possible outcomes:

- **Multiclass cannot recover layer well**: subtype model is not a good model of anything. Layer is more robust than subtype in this dataset.
- **Multiclass recovers layer but fails within-layer subtype**: clean scientific result — function carries layer more robustly than subtype.
- **Multiclass succeeds at both**: subtype information is genuinely accessible; the within-layer binaries should also succeed.

---

## 5. Files Phase 3 will produce

```
notebooks/phase3/
  01_phase3_labels.ipynb              ← Step 1 (this notebook)
  02_l5_it_et_main.ipynb              ← Steps 2–5
  03_l5_it_et_loso.ipynb              ← Step 6
  04_l6_it_ct.ipynb                   ← Step 8
  05_phase3_multiclass.ipynb          ← Step 9

data/processed/tables/
  units_v1_exc_best_subtype.parquet   ← extended unit table
  phase3_subtype_counts_global.csv
  phase3_subtype_counts_per_scan.csv
  phase3_loso_valid_scans.json

data/processed/results/
  phase3_l5_it_et_runs.parquet
  phase3_l5_it_et_loso.parquet
  phase3_l6_it_ct_runs.parquet
  phase3_l6_it_ct_loso.parquet
  phase3_multiclass_runs.parquet
  phase3_multiclass_layer_recovery.parquet
```

---

## 6. What Phase 3 will not claim

- It will not claim to have decoded "genetic" or "morphological" type from function.
- It will not interpret a Tier G win as direct evidence for IT/ET kinetic differences without the C1 / B feature-family follow-up.
- It will not average meaningless LOSO folds where the held-out scan has only one class or fewer than 5 minority cells.
- It will not silently confound SNR with subtype identity — every reportable number passes the `cc_abs` control.

If L5 IT/ET fails, Phase 3 reports the failure as the result. The dataset, stimulus protocol, and population are fixed; the absence of decodable subtype identity within them is itself a publishable observation.
