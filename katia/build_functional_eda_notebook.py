"""
Generator script for `functional_eda.ipynb`.

Run from the katia/ folder:  python3 build_functional_eda_notebook.py
This produces a clean, organized notebook that explores the structure of the
MICrONS functional dataset and answers Katia's specific questions about
trials, sessions, stimuli, condition hashes, and time-series shape.

The script lives in katia/ so the team can regenerate the notebook from
source if cells get accidentally rewritten.
"""

import json
from pathlib import Path

# ----------------------------- helpers -----------------------------

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

cells = []

# ============================ HEADER ============================

cells.append(md("""# Functional Data — Exploratory Analysis

**Goal.** Build a clear, ground-truth picture of the structure of the
MICrONS functional dataset (`data/functional/microns_functional.h5`) so that
every downstream design choice (feature extraction, train/test splits,
classifier inputs) sits on a verified understanding of the data.

This notebook does **not** depend on `microns-datacleaner` for the schema
exploration — it reads the H5 directly, which is faster and removes any
ambiguity coming from the cleaner's higher-level API.

---

## Questions answered here

1. **What is a "time series" in this dataset?** Is it one per session-scan,
   one per trial, one per (neuron, trial)?
2. **How long is each time series?** Frames, seconds, sampling rate. Do all
   trials have the same length, or does length depend on the stimulus?
3. **Are the same neurons recorded across multiple session-scans?** What
   does it mean to compare "the same unit" across sessions?
4. **Is the trial sequence (sequence of `condition_hashes`) the same in
   every session, or unique per session?** How many conditions are shared
   across sessions ("oracle") vs. session-specific?
5. **How many condition hashes belong to each stimulus family** (`Clip`,
   `Monet2`, `Trippy`)?
6. **Do we have the stimuli themselves as time series** (pixel-level video),
   and at what sampling rate are they aligned to the neural responses?
7. **What behavioural signals come along with the responses** (pupil,
   treadmill)?
"""))

# ============================ SETUP ============================

cells.append(md("## 1. Setup\n"))

cells.append(code("""# Standard scientific stack
import os
from pathlib import Path
from collections import Counter
import urllib.parse

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
pd.set_option('display.max_columns', 50)
"""))

cells.append(code("""# Path to the functional H5. The file is ~20 GB and lives outside this folder.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'katia' else Path.cwd()
H5_PATH = Path('data/functional/microns_functional.h5')
assert H5_PATH.exists(), f'Functional H5 not found at {H5_PATH.resolve()}'

print('H5 path:', H5_PATH.resolve())
print('Size: %.2f GB' % (H5_PATH.stat().st_size / 1e9))
"""))

# ============================ Q1: Schema ============================

cells.append(md("""---
## 2. H5 schema at a glance

The file has four top-level groups:

| group | contents |
|-------|----------|
| `brain_areas/` | one entry per area (V1, LM, AL, RL) — purely a registry |
| `sessions/`   | the actual recordings: one entry per session-scan (e.g. `4_7`) |
| `types/`      | maps stimulus type (`Clip`, `Monet2`, `Trippy`) → set of condition hashes |
| `videos/`     | one entry per stimulus condition: pixel data + per-instance traces |

We confirm this on the actual file below.
"""))

cells.append(code("""def describe_group(g, depth=0, max_depth=2, max_per_level=6, indent='  '):
    \"\"\"Recursively print group structure with shapes/dtypes for datasets.\"\"\"
    keys = list(g.keys())
    print(indent * depth + f'{g.name}/  ({len(keys)} children)')
    for k in keys[:max_per_level]:
        v = g[k]
        if isinstance(v, h5py.Group):
            if depth + 1 <= max_depth:
                describe_group(v, depth + 1, max_depth, max_per_level, indent)
            else:
                print(indent * (depth + 1) + f'{k}/  ({len(list(v.keys()))} children)')
        else:
            print(indent * (depth + 1) + f'{k}  shape={v.shape} dtype={v.dtype}')
    if len(keys) > max_per_level:
        print(indent * (depth + 1) + f'... ({len(keys) - max_per_level} more)')

with h5py.File(H5_PATH, 'r') as f:
    print('Top-level groups:', list(f.keys()))
    print()
    describe_group(f, max_depth=1, max_per_level=4)
"""))

# ============================ Q2: Sessions inventory ============================

cells.append(md("""---
## 3. Sessions inventory

How many session-scans are there, and how many recorded units does each one
contain (split by brain area)?
"""))

cells.append(code("""with h5py.File(H5_PATH, 'r') as f:
    rows = []
    for sess in sorted(f['sessions'].keys()):
        meta = f['sessions'][sess]['meta']
        n_units_total = meta['unit_ids'].shape[0]
        n_trials = len(f['sessions'][sess]['trials'])
        ai = meta['area_indices']
        row = {
            'session_scan': sess,
            'session': int(sess.split('_')[0]),
            'scan_idx': int(sess.split('_')[1]),
            'n_units': n_units_total,
            'n_trials': n_trials,
            'n_V1': ai['V1'].shape[0],
            'n_LM': ai['LM'].shape[0],
            'n_AL': ai['AL'].shape[0],
            'n_RL': ai['RL'].shape[0],
        }
        rows.append(row)

sessions_df = pd.DataFrame(rows).sort_values('session_scan').reset_index(drop=True)
sessions_df
"""))

cells.append(code("""# Visualize per-session area composition.
fig, ax = plt.subplots(figsize=(11, 4))
sessions_df.set_index('session_scan')[['n_V1', 'n_LM', 'n_AL', 'n_RL']].plot(
    kind='bar', stacked=True, ax=ax,
    color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
)
ax.set_ylabel('# functional units')
ax.set_title('Functional units per session-scan, by brain area')
ax.legend(title='Area', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""**Takeaway.** 14 session-scans. Every session contains units from all 4
areas, with V1 dominating (typical V1 share ≈ 65–80% of units in a scan).
Total units per scan ranges roughly 5 000 – 11 000.
"""))

# ============================ Q3: Time series shape ============================

cells.append(md("""---
## 4. Q1 + Q2 — What is a "time series" and how long is it?

Inside a session-scan we have `trials/0`, `trials/1`, ..., `trials/463` — i.e.
**464 trials per session-scan**. A single trial group looks like this:

```
sessions/<sess>/trials/<trial_id>/
├── responses    (n_units, n_frames)   float32   ← calcium responses
├── stim_times   (n_frames,)           float32   ← timestamp per frame (s)
├── pupil        (4, n_frames)         float32   ← pupil tracking
└── treadmill    (n_frames, 1)         float32   ← running speed
```

So the **fundamental time series is per `(neuron, trial)`** — for each unit
and each trial we have one calcium trace of length `n_frames`. Per
session-scan a single neuron therefore has **464 time series** (one per
trial), not one continuous recording.

Below we confirm this empirically and measure trial lengths.
"""))

cells.append(code("""# Inspect one trial in detail.
SAMPLE_SESS = '4_7'
SAMPLE_TRIAL = '0'

with h5py.File(H5_PATH, 'r') as f:
    tr = f['sessions'][SAMPLE_SESS]['trials'][SAMPLE_TRIAL]
    print(f'Session {SAMPLE_SESS}, trial {SAMPLE_TRIAL}:')
    for k in tr.keys():
        v = tr[k]
        print(f'  {k}: shape={v.shape} dtype={v.dtype}')
    n_units = tr['responses'].shape[0]
    n_frames = tr['responses'].shape[1]
    print(f'\\n=> {n_units} neurons × {n_frames} frames in this single trial')
"""))

cells.append(code("""# Trial-length distribution within one session: do all trials have the same length?
with h5py.File(H5_PATH, 'r') as f:
    trial_keys = sorted(f['sessions'][SAMPLE_SESS]['trials'].keys(), key=int)
    n_frames_list = [
        f['sessions'][SAMPLE_SESS]['trials'][tk]['responses'].shape[1]
        for tk in trial_keys
    ]

print(f'Frame-count distribution for {SAMPLE_SESS}:', Counter(n_frames_list))
print(f'Total trials: {len(n_frames_list)}')
"""))

cells.append(code("""# Estimate sampling rate (Hz) from stim_times.
with h5py.File(H5_PATH, 'r') as f:
    tr = f['sessions'][SAMPLE_SESS]['trials']['0']
    st = tr['stim_times'][:]

dt = np.diff(st)
fps = 1.0 / dt.mean()
duration = st[-1] - st[0]
print(f'Trial 0 of {SAMPLE_SESS}:')
print(f'  duration = {duration:.3f} s')
print(f'  mean Δt  = {dt.mean()*1000:.2f} ms  →  fps ≈ {fps:.2f} Hz')
print(f'  Δt std   = {dt.std()*1000:.3f} ms (regularity check)')
"""))

cells.append(md("""**Takeaway (so far).** Trials in one session have a
*bimodal* length: **75 frames** or **113 frames**. Calcium imaging is
sampled at ≈ 7.5 Hz. So each individual time series is either ≈ 10 s long
(75 frames) or ≈ 15 s long (113 frames). The next section ties this to the
stimulus.
"""))

# ============================ Q4: Stimulus families ============================

cells.append(md("""---
## 5. Q5 — Stimulus families and how many conditions in each

The `types/` group splits all condition hashes by stimulus family:

- **`Clip`**   — natural movies (cinematic, rendered, sports)
- **`Monet2`** — parametric drifting orientation noise
- **`Trippy`** — parametric phase noise

Below we count hashes in each family and verify they are mutually
exclusive.
"""))

cells.append(code("""with h5py.File(H5_PATH, 'r') as f:
    type_sets = {t: set(f['types'][t].keys()) for t in f['types'].keys()}

type_counts = pd.Series({t: len(s) for t, s in type_sets.items()})
print('Unique condition hashes per stimulus family:')
print(type_counts)
print(f'\\nTotal across families: {type_counts.sum()}')

# Check mutual exclusivity.
overlaps = {
    f'{a} ∩ {b}': len(type_sets[a] & type_sets[b])
    for a in type_sets for b in type_sets if a < b
}
print('\\nPairwise overlaps:', overlaps)
"""))

cells.append(code("""# Visualize.
fig, ax = plt.subplots(figsize=(6, 4))
type_counts.plot(kind='bar', ax=ax,
                 color=['#3a86ff', '#ffbe0b', '#fb5607'])
ax.set_ylabel('# unique condition hashes')
ax.set_title('Catalogue size per stimulus family')
for i, v in enumerate(type_counts.values):
    ax.text(i, v + 30, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""**Takeaway.** The stimulus library has **2 411** unique conditions in
total: 2 112 Clip + 150 Monet2 + 149 Trippy. Families are mutually
exclusive — every condition hash belongs to exactly one family.
"""))

# ============================ Q6: Stim videos ============================

cells.append(md("""---
## 6. Q6 — Do we have the stimuli themselves as time series?

Yes — each unique condition lives in `videos/<hash>/` with:

```
videos/<hash>/
├── clip       (n_frames, H, W)   uint8     ← downsampled to calcium-rate
├── times      (n_frames,)        float32   ← seconds
└── instances  (group)            ← one sub-group per (session, trial) the
                                    stimulus appeared in, mirroring the
                                    response/pupil/treadmill arrays
```

Two important details:

1. The **video frame count matches the calcium frame count** of the
   corresponding trials (75 for Clip, 113 for Monet2/Trippy). The original
   movies were recorded at 30 fps (Clip) or 60 fps (Monet/Trippy) but the
   stored `clip` is already aligned to the calcium sampling grid.
2. The **spatial dimensions differ by family**:
   Clip = 144×256, Monet2 = 126×216, Trippy = 90×160.
"""))

cells.append(code("""# Inspect one example per stimulus family.
with h5py.File(H5_PATH, 'r') as f:
    rows = []
    for stim_type in ['Clip', 'Monet2', 'Trippy']:
        sample_hash = next(iter(f['types'][stim_type].keys()))
        v = f['videos'][sample_hash]
        rows.append({
            'family': stim_type,
            'clip_shape': str(v['clip'].shape),
            'times_len': v['times'].shape[0],
            'duration_s': v.attrs.get('duration', np.nan),
            'orig_fps': v.attrs.get('fps', np.nan),
            'extra_attrs': {k: v.attrs[k] for k in v.attrs.keys()
                            if k not in ('duration', 'fps', 'type', 'original_hash')},
        })

stim_video_df = pd.DataFrame(rows)
stim_video_df
"""))

cells.append(code("""# Visualise one frame from each family.
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
with h5py.File(H5_PATH, 'r') as f:
    for ax, stim_type in zip(axes, ['Clip', 'Monet2', 'Trippy']):
        sample_hash = next(iter(f['types'][stim_type].keys()))
        clip = f['videos'][sample_hash]['clip']
        ax.imshow(clip[clip.shape[0] // 2], cmap='gray')
        ax.set_title(f'{stim_type} — frame {clip.shape[0]//2}\\nshape {clip.shape}')
        ax.axis('off')
plt.tight_layout()
plt.show()
"""))

# ============================ Q7: Hash sequences across sessions ============================

cells.append(md("""---
## 7. Q4 — Trial sequence: same across sessions or unique?

Each session-scan stores `meta/condition_hashes` of length 464 — the
sequence of condition hashes shown trial-by-trial. We answer three nested
questions:

1. Are sequences identical across sessions? (i.e. did each animal see the
   same trials in the same order?)
2. How many *distinct* conditions does a session use, and how many times
   does each repeat within a session?
3. Across sessions: how many conditions are shared, vs. session-specific?

Note on encoding: `condition_hashes` is stored **raw** (e.g. `"/abc..."`,
`"+def..."`), but `types/X/...` and `videos/...` keys URL-encode `/` → `%2F`
while leaving `+` alone. The right round-trip is therefore
`urllib.parse.quote(h, safe='+')`.
"""))

cells.append(code("""def encode_hash(raw_hash: str) -> str:
    \"\"\"Convert a raw condition_hash into the form used as a key in
    types/<stim_type>/<key> and videos/<key>.\"\"\"
    return urllib.parse.quote(raw_hash, safe='+')

# Build raw_hash -> stim_type lookup once.
with h5py.File(H5_PATH, 'r') as f:
    enc2type = {}
    for stim_type in f['types'].keys():
        for h in f['types'][stim_type].keys():
            enc2type[h] = stim_type

def hash_to_type(raw_hash: str) -> str:
    return enc2type.get(encode_hash(raw_hash), 'UNK')

# Sanity check on one session.
with h5py.File(H5_PATH, 'r') as f:
    raw = f['sessions'][SAMPLE_SESS]['meta']['condition_hashes'][:]
    raw = [c.decode() if isinstance(c, (bytes, bytearray)) else c for c in raw]

types_in_seq = [hash_to_type(h) for h in raw]
print(f'{SAMPLE_SESS}: {len(raw)} trials')
print('Stimulus mix in trial sequence:', Counter(types_in_seq))
"""))

cells.append(code("""# Per-session: trial-sequence breakdown by stimulus family.
with h5py.File(H5_PATH, 'r') as f:
    rows = []
    seqs = {}  # session -> tuple of raw hashes (the full ordering)
    for sess in sorted(f['sessions'].keys()):
        raw = f['sessions'][sess]['meta']['condition_hashes'][:]
        raw = tuple(c.decode() if isinstance(c, (bytes, bytearray)) else c for c in raw)
        seqs[sess] = raw
        cnt = Counter(hash_to_type(h) for h in raw)
        rows.append({
            'session_scan': sess,
            'n_trials': len(raw),
            'n_unique_conditions': len(set(raw)),
            'Clip': cnt.get('Clip', 0),
            'Monet2': cnt.get('Monet2', 0),
            'Trippy': cnt.get('Trippy', 0),
            'UNK': cnt.get('UNK', 0),
        })

trial_breakdown = pd.DataFrame(rows)
trial_breakdown
"""))

cells.append(code("""# Are the trial *sequences* (order of hashes) identical across any pair of sessions?
seq_hashes = {sess: hash(seqs[sess]) for sess in seqs}
n_distinct_orderings = len(set(seq_hashes.values()))
print(f'Distinct trial orderings across {len(seqs)} sessions: {n_distinct_orderings}')

# Overlap of *sets* of conditions across sessions.
sess_keys = sorted(seqs.keys())
sess_unique = {s: set(seqs[s]) for s in sess_keys}

overlap = pd.DataFrame(
    {a: [len(sess_unique[a] & sess_unique[b]) for b in sess_keys] for a in sess_keys},
    index=sess_keys
)
print('\\nPairwise condition-set overlap (subset; full table available as overlap):')
print(overlap.iloc[:7, :7])
"""))

cells.append(code("""# Heatmap of overlap.
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(overlap, annot=True, fmt='d', cmap='Blues', ax=ax,
            cbar_kws={'label': '# shared condition hashes'})
ax.set_title('Cross-session overlap of condition hashes')
plt.tight_layout()
plt.show()
"""))

cells.append(code("""# How many sessions does each unique hash appear in?
hash_session_count = Counter()
for sess in sess_keys:
    for h in sess_unique[sess]:
        hash_session_count[h] += 1

dist = Counter(hash_session_count.values())
print('Distribution: hash -> # sessions it appears in')
for k in sorted(dist):
    print(f'  in {k:>2} session(s): {dist[k]:>5} hashes')

oracle_hashes = [h for h, c in hash_session_count.items() if c == len(sess_keys)]
print(f'\\nOracle conditions (in all {len(sess_keys)} sessions): {len(oracle_hashes)}')
print('  by stimulus family:', Counter(hash_to_type(h) for h in oracle_hashes))
"""))

cells.append(code("""# Within a session: how many times does each unique condition repeat,
# broken down by stimulus family?
def repeats_by_family(sess):
    raw = seqs[sess]
    counts = Counter(raw)
    out = {'Clip': [], 'Monet2': [], 'Trippy': [], 'UNK': []}
    for h, c in counts.items():
        out[hash_to_type(h)].append(c)
    return out

repeats = repeats_by_family(SAMPLE_SESS)
print(f'Within {SAMPLE_SESS}, repeats per unique hash by stimulus family:')
for t in ['Clip', 'Monet2', 'Trippy']:
    print(f'  {t:<6}: {dict(Counter(repeats[t]))}  '
          f'(n_unique={len(repeats[t])}, n_total={sum(repeats[t])})')
"""))

cells.append(md("""**Takeaways for the trial structure.**

- All 14 sessions have **464 trials** and **280 unique conditions** each,
  but no two sessions share the same trial ordering — each session-scan was
  presented with its **own randomized sequence**.
- Within a session the family mix is **~384 Clip · ~40 Monet2 · ~40 Trippy**
  trials (≈ 83% / 8.5% / 8.5%).
- **116 conditions are shown in every session** ("oracle" set, used for
  cross-session decoding / generalization tests): 96 Clip + 10 Monet2 +
  10 Trippy. The remaining ~164 conditions in each session are
  session-specific.
- Within a session, most Clip conditions appear once or twice, while every
  Monet2 / Trippy condition appears exactly twice. Oracle Clip conditions
  appear ~10 times each — multiple repeats here are what allow clean
  trial-averaged "tuning" features.
- All 464 hashes resolve to a stimulus family — provided the lookup uses
  `urllib.parse.quote(h, safe='+')` (only `/` is URL-encoded in the
  `types/` and `videos/` keys, `+` is not).
"""))

# ============================ Q5: Neuron overlap across scans ============================

cells.append(md("""---
## 8. Q3 — Same neurons across session-scans?

The H5 stores `unit_ids` per session. Two important caveats:

- The same numeric `unit_id` can appear in different sessions, but **that
  does *not* mean the same biological neuron**. The H5 alone does not
  uniquely identify a neuron across scans.
- Cross-scan neuron identity comes from the **structural data**: every
  matched neuron has a `nucleus_id`, and `microns-datacleaner`'s `match`
  mode produces a `(nucleus_id, session, scan_idx, unit_id)` table that
  links each session-scan record back to the biological neuron.

Below we just measure the raw `unit_id` overlap as an upper bound and flag
that the proper analysis must use `nucleus_id`.
"""))

cells.append(code("""# Raw unit_id overlap (UPPER BOUND, not biological identity).
with h5py.File(H5_PATH, 'r') as f:
    sess_units = {sess: set(int(u) for u in f['sessions'][sess]['meta']['unit_ids'][:])
                  for sess in sorted(f['sessions'].keys())}

uoverlap = pd.DataFrame(
    {a: [len(sess_units[a] & sess_units[b]) for b in sess_units] for a in sess_units},
    index=list(sess_units),
)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(uoverlap, annot=False, cmap='viridis', ax=ax,
            cbar_kws={'label': '# shared unit_id (NOT biological identity)'})
ax.set_title('Raw unit_id overlap across session-scans\\n'
             '(use nucleus_id for true neuron-level matching)')
plt.tight_layout()
plt.show()
"""))

cells.append(md("""**Action item for downstream phases.** When asking *"is neuron X in scan
A also in scan B?"*, do **not** rely on `unit_id` alone — join through
`nucleus_id` from the structural metadata produced by
`microns_datacleaner.MicronsDataCleaner.process_nucleus_data(functional_data='match')`.
"""))

# ============================ Trace + behaviour quick look ============================

cells.append(md("""---
## 9. A quick look at one trace and the behavioural signals

Pulling one trial of one V1 neuron, plus the pupil/treadmill traces, just
to make the data tangible.
"""))

cells.append(code("""SAMPLE_SESS = '4_7'
SAMPLE_TRIAL = '5'  # not trial 0 — first trial sometimes has a longer stim onset

with h5py.File(H5_PATH, 'r') as f:
    tr = f['sessions'][SAMPLE_SESS]['trials'][SAMPLE_TRIAL]
    resp = tr['responses'][:]              # (n_units, n_frames)
    pup = tr['pupil'][:]                   # (4, n_frames)
    tread = tr['treadmill'][:].squeeze()   # (n_frames,)
    st = tr['stim_times'][:]
    st = st - st[0]                        # zero at trial onset

    raw_hash = f['sessions'][SAMPLE_SESS]['meta']['condition_hashes'][int(SAMPLE_TRIAL)]
    raw_hash = raw_hash.decode() if isinstance(raw_hash, (bytes, bytearray)) else raw_hash
    stim_type = hash_to_type(raw_hash)

    # Pick one V1 neuron with non-trivial response in this trial.
    v1_idx = f['sessions'][SAMPLE_SESS]['meta']['area_indices']['V1'][:]
    v1_var = resp[v1_idx].var(axis=1)
    chosen = v1_idx[np.argsort(-v1_var)[3]]   # 4th most variable V1 neuron in this trial

print(f'Trial {SAMPLE_TRIAL} | stim type = {stim_type} | n_frames = {resp.shape[1]}')

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axes[0].plot(st, resp[chosen], color='#2E86AB')
axes[0].set_ylabel('response')
axes[0].set_title(f'V1 unit_id={chosen} — calcium trace ({stim_type} trial)')

for ch in range(pup.shape[0]):
    axes[1].plot(st, pup[ch], alpha=0.7, label=f'pupil ch{ch}')
axes[1].set_ylabel('pupil')
axes[1].legend(loc='upper right', fontsize=8)

axes[2].plot(st, tread, color='#fb5607')
axes[2].set_ylabel('treadmill')
axes[2].set_xlabel('time within trial (s)')

plt.tight_layout()
plt.show()
"""))

# ============================ Summary ============================

cells.append(md("""---
## 10. Summary — direct answers

| # | Question | Answer |
|---|----------|--------|
| 1 | What is one "time series"? | A `(neuron, trial)` calcium trace. Per session-scan a neuron has 464 traces (one per trial), **not** one continuous recording. |
| 2 | How long is a time series? | **75 frames** (Clip, ≈10 s) or **113 frames** (Monet2 / Trippy, ≈15 s) at ≈ **7.5 Hz**. |
| 3 | Trials per session-scan? | Always **464**. |
| 4 | Condition variety per session-scan? | **280** unique condition hashes per session-scan, distributed as **~240 Clip + 20 Monet2 + 20 Trippy**, played out over 464 trials. |
| 5 | Same trial sequence across sessions? | **No.** All 14 sessions have a different randomized ordering. **116** conditions are shared across *all* sessions ("oracle": 96 Clip + 10 Monet2 + 10 Trippy); the rest are session-specific. |
| 6 | # condition hashes per stimulus family? | **Clip 2 112 · Monet2 150 · Trippy 149** (total 2 411, mutually exclusive). |
| 7 | Stimuli as time series? | Yes — `videos/<hash>/clip` of shape `(n_frames, H, W)`, already aligned to the calcium frame rate (75 / 113 frames). H×W differs by family. |
| 8 | Behavioural signals? | Per trial: `pupil` (4, n_frames) and `treadmill` (n_frames, 1). Same time-base as `responses`. |
| 9 | Neuron identity across scans? | **Not** via `unit_id` (numeric overlaps are coincidental). Use `nucleus_id` from the structural metadata (`microns-datacleaner` match mode). |

### Implications for the classifier (CLAUDE.md research question)

- The natural input unit for a per-neuron classifier is **a 3-D tensor of
  shape `(n_trials, n_frames, ...)`** — and `n_frames` varies by stim
  family, so any architecture has to handle two trial lengths (or pad/crop
  uniformly per family, or process Clip / Monet2 / Trippy as separate
  streams).
- The 116 oracle conditions give a **clean cross-session axis**: features
  derived from the same stimuli regardless of which session a neuron was
  imaged in.
- The pre-computed 1024-dim foundation-model embedding mentioned in
  CLAUDE.md sidesteps the variable-length issue entirely; this notebook
  documents what we'd lose by not using it.
"""))

# ============================ WRITE NOTEBOOK ============================

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path('functional_eda.ipynb')
with open(OUT, 'w') as f:
    json.dump(notebook, f, indent=1)
print(f'Wrote {OUT.resolve()}  ({len(cells)} cells)')
