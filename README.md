# Decoding cortical layer from functional fingerprints in MICrONS V1 excitatory neurons

Neuroscience final project.

## Overview

Cortical excitatory neurons are organised into layers (L2/3, L4, L5, L6) that
differ in their anatomy and connectivity. This project asks a complementary
question: **how much of a neuron's layer identity is written into its *function*
— the way it responds to visual stimuli and behaviour — rather than its
anatomy alone?**

Using the MICrONS V1 functional-connectomics dataset, we frame this as a
4-class layer-decoding problem (chance = 0.25): given only functional descriptors
of a neuron, predict which cortical layer it belongs to. Decoding accuracy above
chance means layer membership leaves a measurable signature in the response
properties; comparing feature sets and models tells us *where* that signature
lives and *how* it is best read out.

## Approach

The analysis is built around three axes:

- **Feature sets.** Engineered tabular tiers — response amplitude and shape (A1),
  reliability (B), behaviour (C1) and stimulus descriptors (D1) — are compared
  against a high-dimensional oracle "functional fingerprint" (Tier G) that serves
  as an upper bound on the available layer information.
- **Models.** Linear (logistic regression), gradient-boosted trees
  (HistGradientBoosting), a set-based **Deep Sets** model and a **CNN** that reads
  the raw response traces, spanning simple per-neuron classifiers up to
  architectures that pool information across a neuron's recordings.
- **Evaluation protocols.** Within-session GroupKFold (GKF) measures the ceiling
  when train and test come from the same recording; leave-one-session-out (LOSO)
  is the stricter test of whether the layer code generalises to *new* sessions.
  The gap between them quantifies session-to-session shift.

On top of the main benchmark, the project runs a set of **controls**: label-
consistency filtering (cell-type-implied vs depth-derived layer), confound checks
(scan identity, response quality), seed-stability runs, and temporal controls
that shuffle the time axis of the CNN input to test whether fine temporal
structure — as opposed to coarse response amplitude — is what carries the signal.

## Repository structure

```
neuroscience-project/
├── Main/
│   ├── notebooks/        # analysis pipeline (numbered notebooks + phase2/, phase3/)
│   ├── src/              # shared code: data loading, feature extraction, models
│   ├── requirements.txt  # Python dependencies
│   └── .gitignore
├── report.pdf            # final written report
└── README.md
```

The numbered notebooks in `Main/notebooks/` are meant to be run in order: dataset
construction and label-consistency filtering, the model × feature-set ablation,
the within-session and LOSO benchmarks, and finally the controls and stability
analyses. The `phase2/` include the CNN extensions, including the temporal-control experiments. The `phase3/` contain additional exploratory analysis on subtype decoding.

## Report

The full write-up — methods, results and appendices (dataset composition, full
result plots, the broader 13-session extension, and side exploratory analyses
such as per-stimulus-family and excitatory-subtype decoding) — is in
[`report.pdf`](report.pdf).

