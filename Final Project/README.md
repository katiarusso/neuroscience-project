# Final Project — Implementation sketch

This folder is the **clean, from-scratch implementation** of the project specified in `WORKFLOW.md`. The README is a *sketch*, not a binding plan: it captures a reasonable starting structure and a small set of working principles, but the actual layout will evolve as the project is built. If a different folder shape, fewer notebooks, or a different cache structure turn out to fit the work better, change them — `WORKFLOW.md` is the source of truth, this README is a starting point.

The exploratory notebooks in `katia/` and `tommy/` are reference and history; the implementation lives here, and is independent of those folders.

---

## 1. Working principles

These are easier to commit to than any specific folder layout. They hold even if §2's tree changes.

1. **Reusable logic lives in modules**, not as cells copied between notebooks. If two notebooks need the same function, that function moves into a Python module under `src/` (or wherever we decide module code lives).
2. **Each notebook is scoped to one question.** A short header at the top declares what the notebook is for, what it reads, and what it writes — that's enough; the exact field names in the header can vary.
3. **Heavy compute is cached.** Feature extraction, RF estimation, and CV runs should not re-execute every time a notebook is reopened. Where the cache lives is less important than that it exists.
4. **Reports regenerate from the cache.** Final figures and tables should reproduce without retraining.
5. **Random seed and path constants are centralized somewhere.** No hard-coded paths in notebooks; one seed, threaded through.
6. **`WORKFLOW.md` is the source of truth.** If the code disagrees with it, update the workflow first, then change the code.
7. **Confound baselines are not optional.** Every result is reported alongside the session-only, `cc_abs`-only, and depth-only baselines from `WORKFLOW.md` §6.5, plus a leave-one-session-out sanity check.
8. **No precomputed tuning parameters, and no orientation features.** `pref_ori`, `pref_dir`, `gOSI`, `gDSI` — whether read off the structural CSVs or computed from the raw data — are out of scope per the project brief and `WORKFLOW.md` §3.5 + §4 Tier F.
9. **Build the smallest end-to-end pipeline first.** A working route from raw data to a single layer-decoding score is more valuable than seven half-built feature tiers; new tiers and stages are increments on a pipeline that already runs.

---

## 2. A possible folder structure (sketch — not binding)

The tree below is one reasonable starting layout. Treat it as a draft. It is fine — and probably necessary at some point — to deviate: combine modules, split notebooks differently, drop pieces that turn out to be overkill, add things that turn out to be needed.

```
Final Project/
├── README.md                         # this file
├── WORKFLOW.md                       # methodology — the binding document
├── requirements.txt
│
├── data/
│   ├── raw/                          # source MICrONS data (read-only)
│   ├── interim/                      # transient checkpoints
│   └── processed/                    # cached artefacts (tables, features, splits, results)
│
├── src/                              # reusable Python modules
│   ├── config.py                     # paths, random seed, constants
│   ├── data/                         # loaders, filters, table builders, splits
│   ├── features/                     # one module per tier (A–G; H optional)
│   ├── models/                       # linear / trees / temporal / multimodal
│   ├── eval/                         # metrics, baselines, CV, ablation
│   └── viz/                          # plotting helpers
│
├── notebooks/                        # see §3 for a sketch of which notebooks
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
└── tests/                            # lightweight, only where it pays off
```

This is illustrative. We may end up with fewer `src/` subfolders, with notebooks grouped differently, or with `tests/` skipped entirely. None of that breaks the project as long as the principles in §1 hold.

---

## 3. A possible notebook breakdown

This is a sketch of what notebooks could exist. Whether each one is its own notebook, a section inside a larger notebook, or skipped entirely is decided as we build. The goal is to keep notebooks scoped — not to enforce any particular numbering scheme.

A reasonable starting set, organized by phase:

- **Setup and EDA.** Path setup, structural EDA (class counts, quality), functional EDA (sessions × trials × stimuli, oracle hashes), and building the canonical neuron tables.
- **Feature building.** One notebook (or section) per tier we actually build: Tier A (response stats), Tier B (reliability), Tier C (per-stimulus tuning), Tier D (behavioral modulation indices), Tier E (temporal shape), Tier G (oracle fingerprint). Tier F and Tier H are optional and deferred.
- **Phase 1 — layer decoding.** Stages 1–5 of `WORKFLOW.md` §5: linear baselines on engineered tiers, add behavioral interactions, nonlinear models, temporal models, multi-modal CNN. Stage 6 (RF) is optional and last.
- **Phase 2 — cell type and within-layer subtype.** Per `WORKFLOW.md` §7's hybrid: within-layer binaries (5P-IT vs 5P-ET, 6P-IT vs 6P-CT), within-L5 3-way diagnostic, full multi-class cell-type prediction.
- **Cross-cutting.** Tier ablation table (the headline scientific output), confound audit (cc_abs / session / depth baselines), per-model interpretability.
- **Report.** Final figures and tables.

A loose numbering convention (e.g. `0X_setup`, `1X_features`, `2X_phase1`, `3X_phase2`, `4X_analysis`, `5X_report`) helps readers follow execution order, but it is a convenience, not a contract.

If a notebook ends up too large or too small, split or merge it. The principle that matters is that each notebook has a single clear purpose and writes outputs that other notebooks can pick up.

---

## 4. Cached artefacts (suggestion)

A reasonable starting layout for `data/processed/`:

- `tables/` — the neuron-level table and the (neuron × hash) trial-averaged table.
- `features/` — one parquet per tier, each one row per neuron.
- `splits/` — CV fold assignments, both grouped-by-neuron and leave-one-session-out.
- `results/` — per-experiment scores, predictions, and the ablation table.

This is a useful default because it makes the project regenerable from cache. Other layouts (e.g. one big features dataframe instead of one per tier) are also fine if they fit better.

---

## 5. Module conventions (defaults, not laws)

If we end up writing reusable modules in `src/`, the defaults below are a sensible starting point. Deviate where the work argues for something else.

- Type hints on public functions where it helps readability.
- Docstrings that describe the contract (inputs, outputs, side effects) where the function is non-obvious.
- Plotting helpers return figures rather than calling `plt.show()` inside library code.
- One feature tier per module, each exposing a `compute_tier_X(...)` function.
- Caching: each tier function can check whether its output already exists and skip recompute, with a `force=True` override.

---

## 6. Reproducibility

- A pinned `requirements.txt` (or `environment.yml`) for the dependencies actually used.
- A single random seed defined in one place.
- Structural CSVs and the functional H5 are read-only.
- Deleting the cache and re-running notebooks in order should reproduce the report.

---

## 7. What this folder explicitly does not include

- Code from `katia/BIN/`, `katia/1.APPROACH/`, `katia/functional_eda.ipynb`, `katia/build_functional_eda_notebook.py`. Reference, not implementation.
- Code from `tommy/`, `sofi/`, `marco/`. Reference, not implementation.
- Precomputed tuning parameters (`pref_ori`, `pref_dir`, `gOSI`, `gDSI`, published oracle correlation) from the structural CSVs.
- Predicted cell-type tables (`baylor_*`, `cg_cell_type_calls`, `cell_type_multifeature_combo`). The label vocabulary is `aibs_metamodel_celltypes_v661.csv` with corrections.

---

## 8. Build order — a starting suggestion

A reasonable first sequence of moves, subject to revision once the work begins:

1. Minimal infrastructure: paths, dependencies, the directory skeleton.
2. Data loading and filtering: V1 / excitatory / matched / best-only.
3. Confirm the dataset numbers in `WORKFLOW.md` §2 reproduce.
4. Build the canonical neuron table and the (neuron × hash) table; define the CV splits.
5. **Tier A + Stage 1 end-to-end on a minimal feature set** — get a working pipeline from raw data to a layer-decoding score before adding anything else.
6. Add tiers one at a time (B, then C, D, E, G), re-running Stage 1 (and Stage 2 once Tier D exists) after each.
7. Phase 1 Stages 2–5; then Phase 2's three deliverables; then ablation, confound audit, interpretability; then report.
8. Tier F and Tier H are optional and live after the main project is defended.

The point of this order is that the project always has a working end-to-end pipeline. New tiers and stages become increments on a green build rather than long-lived branches that may not converge.
