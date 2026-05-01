# Final Project — Implementation Plan

This folder is the **clean, from-scratch implementation** of the project specified in `WORKFLOW.md`. Nothing here imports from the exploratory notebooks in `katia/BIN/`, `katia/1.APPROACH/`, `tommy/`, `sofi/`, or `marco/`. Those folders are reference and history; the implementation lives here.

The README is a structural plan, not a build log: it defines where things go, what each file is responsible for, and the contracts between notebooks. The actual code is created later, one notebook at a time.

---

## 1. Operating principles

1. **Reusable logic lives in `src/`** as importable Python modules. Anything used by more than one notebook, or anything non-trivial, becomes a function in `src/`.
2. **Notebooks are scoped, not monolithic.** Each notebook does *one thing*. It declares its inputs and outputs, runs end-to-end in a reasonable time, and writes its outputs to `data/processed/` so the next notebook can pick them up.
3. **Heavy compute is cached to disk.** Feature extraction, RF estimation, and CV runs write parquet/pickle/npz files. No notebook should re-run a 30-minute computation every time it is opened.
4. **Reports are reproducible from the cache.** The figures and tables in the final report regenerate from `data/processed/` without re-running training.
5. **The workflow document is law.** If a design choice in code disagrees with `WORKFLOW.md`, update `WORKFLOW.md` first, then change the code.
6. **No precomputed tuning parameters.** Every tuning-like quantity (preferred orientation, OSI/DSI, RF) is computed here from raw traces and stimulus video, not read off the structural CSVs (see `WORKFLOW.md` §3.5).

---

## 2. Folder structure

```
Final Project/
├── README.md                         # this file
├── WORKFLOW.md                       # methodology — reasoning behind every choice
├── requirements.txt                  # pinned dependencies
├── environment.yml                   # optional conda env for reproducibility
│
├── data/
│   ├── raw/                          # symlink or path constant pointing at the
│   │                                 # source MICrONS data (read-only, never written)
│   ├── interim/                      # transient checkpoints, free to delete
│   └── processed/                    # canonical cached artefacts (see §4)
│       ├── tables/                   # neuron-level and (neuron × hash) tables
│       ├── features/                 # tier_A..tier_H feature parquet files
│       ├── splits/                   # CV fold assignments
│       └── results/                  # per-experiment scores and predictions
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # paths, random seed, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py                   # microns-datacleaner wrappers, H5 readers
│   │   ├── filters.py                # V1, excitatory, matched, best-only
│   │   ├── tables.py                 # neuron-level and neuron×hash table builders
│   │   └── splits.py                 # GroupKFold, leave-one-session-out
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tier_a_response_stats.py
│   │   ├── tier_b_reliability.py
│   │   ├── tier_c_tuning.py
│   │   ├── tier_d_behavior.py
│   │   ├── tier_e_temporal.py
│   │   ├── tier_f_rf.py
│   │   ├── tier_g_oracle_fingerprint.py
│   │   └── tier_h_foundation.py      # extension; only if Tier H is included
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear.py                 # logistic regression, ridge, linear SVM, LDA
│   │   ├── trees.py                  # random forest, gradient boosting
│   │   ├── temporal.py               # 1D-CNN on per-hash traces
│   │   └── multimodal.py             # multi-branch CNN (neural + behavior + stim)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                # balanced accuracy, macro F1, AUC, confusion
│   │   ├── baselines.py              # majority, random, cc_abs, session, depth
│   │   ├── cv.py                     # CV runners with grouped + LOSO modes
│   │   └── ablation.py               # tier-removal ablation
│   └── viz/
│       ├── __init__.py
│       └── plots.py                  # confusion matrices, ablation bars, etc.
│
├── notebooks/                        # numbered execution order — see §3
│   ├── 00_setup_and_paths.ipynb
│   ├── 01_eda_structural.ipynb
│   ├── 02_eda_functional.ipynb
│   ├── 03_build_neuron_table.ipynb
│   ├── 04_build_neuron_hash_table.ipynb
│   ├── 05_define_splits.ipynb
│   │
│   ├── 10_features_tier_a_response_stats.ipynb
│   ├── 11_features_tier_b_reliability.ipynb
│   ├── 12_features_tier_c_tuning.ipynb
│   ├── 13_features_tier_d_behavior.ipynb
│   ├── 14_features_tier_e_temporal.ipynb
│   ├── 15_features_tier_f_rf.ipynb
│   ├── 16_features_tier_g_oracle_fingerprint.ipynb
│   ├── 17_features_tier_h_foundation.ipynb        # extension
│   │
│   ├── 20_phase1_baseline_linear.ipynb            # Stage 1 of WORKFLOW
│   ├── 21_phase1_with_behavior.ipynb              # Stage 2
│   ├── 22_phase1_nonlinear.ipynb                  # Stage 3
│   ├── 23_phase1_temporal.ipynb                   # Stage 4
│   ├── 24_phase1_multimodal_cnn.ipynb             # Stage 5
│   ├── 25_phase1_rf_features.ipynb                # Stage 6
│   │
│   ├── 30_phase2_L5_subtype.ipynb
│   ├── 31_phase2_L6_subtype.ipynb
│   │
│   ├── 40_ablation_table.ipynb
│   ├── 41_interpretability_linear.ipynb
│   ├── 42_interpretability_trees.ipynb
│   ├── 43_interpretability_cnn.ipynb
│   ├── 44_confound_audit.ipynb                    # session/quality/depth baselines
│   │
│   └── 50_report_figures.ipynb
│
├── reports/
│   ├── figures/                      # PDF/PNG figures, regenerated from cache
│   ├── tables/                       # markdown / CSV tables for the report
│   └── final_report.md
│
└── tests/
    ├── test_filters.py               # filter logic returns the expected counts
    ├── test_splits.py                # no neuron leaks across folds
    └── test_features.py              # tier outputs have correct shape and dtype
```

---

## 3. Notebook plan and contracts

Each notebook starts with a short markdown cell that declares:

- **Question.** The single thing this notebook is supposed to settle.
- **Inputs.** The list of paths in `data/processed/` (or `data/raw/`) it reads. No reads from `data/interim/` from a different notebook.
- **Outputs.** The list of paths it writes. Every output goes through a path constant defined in `src/config.py`.
- **Runtime estimate.** So we know what to expect.
- **Dependencies on src/.** Which `src/` modules it imports.

The numbering encodes order: a notebook may only depend on outputs from notebooks with a lower number.

### 00–09 Setup and exploratory data analysis

| # | Notebook | Question |
|---|----------|----------|
| 00 | `00_setup_and_paths.ipynb` | Verify paths in `src/config.py` resolve, check `microns-datacleaner` version, list available structural CSVs and the H5 file. Run-once sanity. |
| 01 | `01_eda_structural.ipynb` | Class counts (layer, subtype) under each combination of filters; quality (`cc_abs`) distribution per class; matched vs unmatched counts. Reproduces and confirms the Section-2 numbers. |
| 02 | `02_eda_functional.ipynb` | Sessions × trials × stimuli structure; oracle hash count; trial length per stimulus family; behavioral channel availability. |
| 03 | `03_build_neuron_table.ipynb` | Build the canonical V1-excitatory-matched-best-only neuron table (Option A row, plus session/scan/unit_id keys to reach the H5). Output: `tables/neurons.parquet`. |
| 04 | `04_build_neuron_hash_table.ipynb` | Build the (neuron × hash) table with trial-averaged calcium traces, behavior summaries, and trial counts. Output: `tables/neuron_hash.parquet`. This is the unit of analysis (WORKFLOW §3.2 option C). |
| 05 | `05_define_splits.ipynb` | Compute and save the CV fold assignments — both `GroupKFold` by `nucleus_id` and leave-one-session-out (WORKFLOW §6). Output: `splits/cv_folds.parquet`, `splits/loso_folds.parquet`. |

### 10–17 Feature building (one notebook per tier)

Each `1X_features_*` notebook reads `tables/neurons.parquet` and `tables/neuron_hash.parquet`, calls the corresponding `src/features/tier_*.py` module, and writes a `features/tier_*.parquet` file with a fixed schema:

```
nucleus_id (index) | feature_1 | feature_2 | ... | feature_K
```

One row per neuron, K features for that tier. Notebooks 10–16 cover Tiers A–G; notebook 17 is the optional Tier H foundation embedding.

### 20–25 Phase 1 — layer decoding (one notebook per stage)

Each `2X_phase1_*` notebook reads selected feature parquets, joins them on `nucleus_id`, attaches the layer label, and runs the modeling stage described in the corresponding section of WORKFLOW (Section 5). It writes:

- `results/<stage>_scores.parquet` — per-fold balanced accuracy, macro F1, per-class recall.
- `results/<stage>_predictions.parquet` — out-of-fold predictions, for downstream confusion matrices.
- `results/<stage>_models/` — pickled fitted models (small models only; CNNs save weights only).

The decision rule between stages (when to escalate, when to stop) is enforced inside the notebooks: each notebook ends with a markdown cell that compares its score to the previous stage and explicitly states whether the next stage is justified.

### 30–31 Phase 2 — within-layer subtype

Same pipeline restricted to L5 or L6, with bootstrap CIs and explicit small-sample handling. These notebooks run only after Phase 1 has stabilized.

### 40–44 Cross-cutting analyses

| # | Notebook | Question |
|---|----------|----------|
| 40 | `40_ablation_table.ipynb` | For the best Phase-1 stage, retrain with each tier removed; produce the tier-ablation table that is the headline scientific output. |
| 41 | `41_interpretability_linear.ipynb` | Standardized coefficients per class; per-tier coefficient inspection. |
| 42 | `42_interpretability_trees.ipynb` | Permutation importance, SHAP on a subsample. |
| 43 | `43_interpretability_cnn.ipynb` | First-layer filters; temporal occlusion; channel ablation. Only if Stage 4 or 5 ran. |
| 44 | `44_confound_audit.ipynb` | Side-by-side: full model vs cc_abs-only vs session-only vs depth-only baseline. The non-negotiable defense of the project. |

### 50 Report figures

`50_report_figures.ipynb` reads `results/` and `figures/` are saved to `reports/figures/`. The final report (`reports/final_report.md`) regenerates from the cache without re-training anything.

---

## 4. Canonical artefacts in `data/processed/`

These are the files that constitute the project's contract with itself. Anything else in `data/` is local scratch.

```
data/processed/
├── tables/
│   ├── neurons.parquet                       # one row per nucleus_id (best-only V1 exc matched)
│   └── neuron_hash.parquet                   # one row per (nucleus_id, hash) with trial-averaged trace
├── features/
│   ├── tier_a_response_stats.parquet
│   ├── tier_b_reliability.parquet
│   ├── tier_c_tuning.parquet
│   ├── tier_d_behavior.parquet
│   ├── tier_e_temporal.parquet               # may be wide or store basis projections
│   ├── tier_f_rf.parquet                     # RF descriptors; the kernels themselves go to features/rf_kernels.npz
│   ├── tier_g_oracle_fingerprint.parquet
│   └── tier_h_foundation.parquet             # optional
├── splits/
│   ├── cv_folds.parquet                      # nucleus_id -> fold_id
│   └── loso_folds.parquet                    # nucleus_id -> session_id (held out)
└── results/
    ├── phase1_stage1_scores.parquet
    ├── phase1_stage1_predictions.parquet
    ├── ...
    ├── phase2_L5_scores.parquet
    └── ablation_table.parquet
```

---

## 5. `src/` conventions

- **Type hints on every public function.**
- **Docstrings** describe the contract (inputs, outputs, side effects), not the implementation.
- **No notebook code in `src/`**: no `display(...)`, no `plt.show()` inside library functions. Plotting helpers live in `src/viz/` and return figures.
- **Paths only through `src/config.py`.** A constant like `NEURON_HASH_PATH = PROCESSED / "tables" / "neuron_hash.parquet"` is defined once and imported everywhere. Notebooks never hard-code paths.
- **Random seeds only through `src/config.py`.** A single `SEED = 42` constant; functions that randomize accept an explicit `random_state=SEED` argument.
- **One feature tier = one module.** `src/features/tier_a_response_stats.py` exposes a single function `compute_tier_a(neuron_hash_table) -> pd.DataFrame` that returns the tier's feature parquet content. The feature notebook is then a thin wrapper that loads, computes, saves.
- **Caching pattern.** Each tier function checks if its output parquet exists; if yes, returns it; if not, computes and writes. Notebooks pass `force=True` when they want to recompute.

---

## 6. Reproducibility

- `requirements.txt` pins `pandas`, `numpy`, `scikit-learn`, `h5py`, `torch`, `microns-datacleaner`, `xgboost` / `lightgbm`, `shap`, `matplotlib`, `seaborn`.
- `src/config.py` is the single source of truth for the random seed.
- The structural CSVs and the functional H5 are read-only; we never write back to `data/raw/`.
- Intermediate outputs are deterministic functions of inputs, so deleting `data/processed/` and re-running the notebooks 00 → 50 reproduces every number in the report.

---

## 7. Naming conventions

- **Notebooks.** `NN_topic_short_name.ipynb`. Tens digit = phase (0X setup/EDA, 1X features, 2X Phase 1, 3X Phase 2, 4X cross-cutting, 5X report).
- **Feature parquets.** `tier_<lowercase letter>_<short description>.parquet`.
- **Result parquets.** `phase<N>_stage<M>_<artefact>.parquet`.
- **Notebooks emit the same artefact name as their cached parquet** so the dependency is obvious.

---

## 8. Extending the project

- **New tier.** Add `src/features/tier_X_*.py`, add a `1X_features_*` notebook, regenerate `data/processed/features/`, add a row to the ablation script.
- **New model.** Add `src/models/*.py`, add a `2X_phase1_*` notebook, write a results parquet with the same schema as the others so the ablation table picks it up automatically.
- **New target (e.g. inhibitory subtypes).** Restrict in the relevant notebook (e.g. `30_*` analogue), keep the rest of the pipeline unchanged.

---

## 9. What this folder explicitly does not include

- Code from `katia/BIN/`, `katia/1.APPROACH/`, `katia/functional_eda.ipynb`, `katia/build_functional_eda_notebook.py`. Those are reference notebooks for the methodology, not the implementation.
- Code from `tommy/`, `sofi/`, `marco/`. Group members' experiment folders are read-only inputs to discussion, not dependencies of this folder.
- Precomputed tuning parameters (`pref_ori`, `pref_dir`, `gOSI`, `gDSI`, published oracle correlation) from the structural CSVs. We compute every tuning-like quantity from raw traces and stimulus video.
- Predicted cell-type tables (`baylor_*`, `cg_cell_type_calls`, `cell_type_multifeature_combo`). The label vocabulary is `aibs_metamodel_celltypes_v661.csv` with corrections.

---

## 10. Build order — first concrete steps

When the actual implementation starts, the order is:

1. `src/config.py`, `requirements.txt`, the directory skeleton.
2. `src/data/load.py` (datacleaner wrapper, H5 reader) + `src/data/filters.py` (V1/exc/matched/best-only).
3. Notebook 00 → 02 to confirm we reproduce the dataset numbers in `WORKFLOW.md` §2.
4. `src/data/tables.py` and notebooks 03–05 to produce `tables/neurons.parquet`, `tables/neuron_hash.parquet`, and `splits/`.
5. Tier A and a Stage-1 baseline (notebooks 10 and 20) end-to-end before any other tier, to validate the full pipeline on a minimal feature set.
6. Then tiers B → G one at a time, each followed by re-running Stage 1 to update the ablation row.
7. Phase 1 stages 2–6, then Phase 2, then cross-cutting analyses, then the report.

This order means the project always has a working end-to-end pipeline; new tiers and stages are increments on a green build, never long-lived branches.
