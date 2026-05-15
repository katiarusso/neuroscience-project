"""Project configuration: paths, seed, binding constants.

Why this file looks the way it does
-----------------------------------
`microns-datacleaner` builds its target paths by f-string concatenation
against a class-level ``homedir = Path().resolve()`` captured at *the
package's* import time::

    # microns_datacleaner/mic_datacleaner.py
    homedir = Path().resolve()                                       # class attr
    self.data_storage = f"{self.homedir}/{self.datadir}/{self.version}"
    os.makedirs(self.data_storage, exist_ok=True)

Two failure modes follow from that, and both have bitten this project:

1. **Absolute ``datadir`` -> rogue ``Users/`` folder.** If we hand the cleaner
   ``"/Users/<you>/.../Final Project/data"``, the f-string produces
   ``"/Users/<you>/.../notebooks//Users/<you>/.../Final Project/data/1718"``.
   POSIX collapses ``//`` to ``/``, so the package then ``os.makedirs`` a path
   that, *interpreted by the filesystem*, plants a ``Users/<your-home>/...``
   directory chain inside ``notebooks/``. That is the spurious ``Users/``
   folder that appears in ``notebooks/`` after running section 3.

2. **Wrong ``homedir`` capture.** Whatever cwd was active when
   ``microns_datacleaner`` was *first* imported is locked in for the lifetime
   of the kernel. Notebook 01 imports the package in section 1 *before* it
   imports this module, so by the time we run, ``homedir`` is already
   ``Final Project/notebooks``. A bare ``datadir="data"`` would then land
   inside ``notebooks/data`` instead of ``Final Project/data``.

The fix is three parts and lives entirely in this file:

- ``os.chdir(_PROJECT_ROOT)`` so every relative path in this module (and
  every relative path the cleaner uses afterwards) anchors to the project
  root regardless of where Jupyter / VS Code launched the kernel.
- Monkey-patch ``MicronsDataCleaner.homedir`` and
  ``MicronsFunctionalReader.homedir`` to ``_PROJECT_ROOT`` so the
  already-cached value picked up at notebook section 1 is corrected.
- Expose ``DATA_DIR`` as a *bare* relative ``Path("data")``, never an
  absolute path. ``str(DATA_DIR) == "data"``, which is what the cleaner
  expects and what makes the f-string above resolve to
  ``<project_root>/data/<version>``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Anchor the project root from this file's own location, NOT from the cwd.
# `config.py` lives at `Final Project/src/config.py`, so two `.parent`s up.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Move cwd to the project root. All relative paths in this module, and any
# relative path the cleaner subsequently builds against its (now-patched)
# homedir, will resolve here. Importing this module is the canonical "set up
# the project" step in every notebook -- the chdir is part of that contract.
os.chdir(_PROJECT_ROOT)

# Repair the package's class-level ``homedir`` if the package was imported
# from a different cwd before this module ran (it is, in notebook 01). We
# touch attributes on already-loaded classes; new instances will read the
# patched value. Wrapped in try/except so this file still imports cleanly in
# environments where ``microns-datacleaner`` is not installed.
try:
    from microns_datacleaner.mic_datacleaner import MicronsDataCleaner as _MDC
    _MDC.homedir = _PROJECT_ROOT
except Exception:
    pass
try:
    from microns_datacleaner.functionalreader import MicronsFunctionalReader as _MFR
    _MFR.homedir = _PROJECT_ROOT
except Exception:
    pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# IMPORTANT: ``DATA_DIR`` is a *relative* Path. ``str(DATA_DIR) == "data"``,
# which is what we hand to ``MicronsDataCleaner(datadir=...)`` and to
# ``MicronsFunctionalReader(datadir=...)``. Do NOT redefine this as an
# absolute path -- doing so reintroduces the leading-slash bug described in
# this file's docstring (the rogue ``Users/`` chain inside ``notebooks/``).
#
# Joins like ``DATA_DIR / "interim"`` stay relative; they resolve correctly
# because we chdir'd to the project root above.
DATA_DIR: Final[Path] = Path("data")

INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
FUNCTIONAL_DIR: Final[Path] = DATA_DIR / "functional"
FUNCTIONAL_H5: Final[Path] = FUNCTIONAL_DIR / "microns_functional.h5"

PROCESSED_TABLES_DIR: Final[Path] = PROCESSED_DIR / "tables"
PROCESSED_FEATURES_DIR: Final[Path] = PROCESSED_DIR / "features"
PROCESSED_SPLITS_DIR: Final[Path] = PROCESSED_DIR / "splits"
PROCESSED_RESULTS_DIR: Final[Path] = PROCESSED_DIR / "results"
PROCESSED_PREDICTIONS_DIR: Final[Path] = PROCESSED_DIR / "predictions"

FIGURES_DIR: Final[Path] = Path("reports") / "figures"


def ensure_dirs() -> None:
    """Create the standard directory tree if it does not already exist."""
    for p in (
        INTERIM_DIR,
        PROCESSED_DIR,
        FUNCTIONAL_DIR,
        FIGURES_DIR,
        PROCESSED_TABLES_DIR,
        PROCESSED_FEATURES_DIR,
        PROCESSED_SPLITS_DIR,
        PROCESSED_RESULTS_DIR,
        PROCESSED_PREDICTIONS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42


# ---------------------------------------------------------------------------
# Dataset version (WORKFLOW.md §2.1 pins 1718)
# ---------------------------------------------------------------------------
MICRONS_VERSION: Final[int] = 1718


# ---------------------------------------------------------------------------
# Working-population filters (binding from WORKFLOW.md §1.2)
# ---------------------------------------------------------------------------
TARGET_BRAIN_AREA: Final[str] = "V1"
TARGET_CLASSIFICATION_SYSTEM: Final[str] = "excitatory_neuron"
FUNCTIONAL_DATA_MODE: Final[str] = "best_only"


# ---------------------------------------------------------------------------
# Feature-engineering bans (binding from WORKFLOW.md §3.5 and §4 Tier F)
# ---------------------------------------------------------------------------
ORIENTATION_FEATURE_DENYLIST: Final[tuple[str, ...]] = (
    "pref_ori",
    "pref_dir",
    "gOSI",
    "gDSI",
    "global_dir",
    "global_ori",
    "orientation_selectivity",
    "direction_selectivity",
)

DIGITAL_TWIN_INPUT_DENYLIST: Final[tuple[str, ...]] = (
    "pref_ori",
    "pref_dir",
    "gOSI",
    "gDSI",
    "oracle",
)

LABEL_TABLE_DENYLIST: Final[tuple[str, ...]] = (
    "baylor_log_reg_cell_type_coarse_v1",
    "baylor_gnn_cell_type_fine_model_v2",
    "cg_cell_type_calls",
    "cell_type_multifeature_combo",
)


# ---------------------------------------------------------------------------
# Stimulus families
# ---------------------------------------------------------------------------
STIMULUS_FAMILIES: Final[tuple[str, ...]] = ("Clip", "Monet2", "Trippy")
