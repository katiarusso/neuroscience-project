# Phase 2 — Revised planning document: staged CNN analysis on raw calcium traces

*This document updates the Phase-2 plan to reflect the practical constraint that full CNN training is computationally expensive. The plan keeps the original scientific question — whether raw temporal calcium traces contain layer-discriminative information beyond Phase 1 engineered features — but introduces a staged computational strategy. The first stage uses one carefully selected balanced session to debug and validate the CNN pipeline before scaling to more sessions and, if feasible, LOSO.*

---

## 0. Core framing

Phase 2 tests whether raw stimulus-locked calcium traces contain layer-discriminative information that was not captured by Phase 1 engineered features.

The final scientific benchmark remains Tier G HGB, because Tier G was the strongest Phase-1 representation: one row per neuron, 116 oracle-hash response amplitudes, and session-invariant stimulus axes. However, because CNN training on raw traces is computationally expensive, the project will not start immediately with full 13-session LOSO. Instead, Phase 2 will proceed in stages:

1. **Single-session feasibility** on one balanced and label-clean session.
2. **Replication on a second balanced session**.
3. **Two-session transfer** to test whether the CNN collapses outside the training scan.
4. **Balanced-session pooled training** using the six four-class sessions.
5. **Balanced-session LOSO** if compute allows.
6. **Full 13-session LOSO** only if earlier stages justify the cost.

The one-session analysis is not the headline scientific result. It is an architecture and data-pipeline feasibility analysis. It answers: can a small 1D CNN learn layer-relevant structure from raw traces in a controlled within-session setting?

The later multi-session analyses answer the stronger scientific question: does the trace CNN generalise across scans and improve over Tier G?

---

## 1. Phase-1 facts that constrain Phase 2

### 1.1 Tier G is the benchmark

Phase 1 established that the strongest model was Tier G HGB:

- GKF balanced accuracy: **0.663 ± 0.008**
- LOSO balanced accuracy: **0.497 ± 0.105**
- Per-class recall under GKF: L2/3 = 0.74, L4 = 0.63, L5 = 0.53, L6 = 0.75

Tier G is biologically and methodologically clean because it represents each neuron by its response fingerprint across the 116 oracle stimuli. Each feature column corresponds to the same physical stimulus in every session.

Phase 2 should therefore compare CNNs to Tier G whenever possible, using the same neurons and the same oracle-hash subset.

### 1.2 Cross-session generalisation is the hard problem

Phase 1 showed a large gap between within-protocol GKF and LOSO. Tier G dropped from 0.663 to 0.497; the long-row model dropped more severely. Therefore, a CNN that works only within session is not enough to claim general biological decoding.

However, a one-session CNN is still useful as a feasibility step. If the CNN cannot learn within one balanced session, it is unlikely to succeed under LOSO.

### 1.3 Label mismatch should be handled before Phase 2

The label sensitivity analysis found that 407 / 8,895 neurons, or 4.6%, show disagreement between depth-derived `layer_label` and morphology-derived `cell_type`-implied layer. The mismatch is concentrated in deep layers and especially in four session-6 scans.

Because Phase 2 is computationally expensive and sensitive to noisy labels, the first CNN experiments should use the **consistent-only subset**:

$$
layer\_label = celltype\_layer
$$

This reduces label ambiguity and makes the first feasibility test cleaner. The inconsistent neurons should not be discarded forever: later they can be reintroduced as a sensitivity analysis, but not in the first CNN debug experiment.

### 1.4 Use balanced sessions first

The initial CNN should use one of the six four-class sessions:

$$
5\_6, 5\_7, 6\_2, 6\_4, 6\_6, 6\_7
$$

The preferred first choice is a session that is both balanced and low-mismatch. Based on Phase-1 sensitivity analyses, **5_6 or 5_7** are the safest starting sessions because they have very low mismatch and strong within-session Tier G performance.

Avoid starting with high-mismatch session-6 scans unless the purpose is explicitly to test difficult recordings. They are scientifically interesting, but not ideal for first architecture debugging.

---

## 2. Data representation for Phase 2

Phase 2 uses raw stimulus-locked calcium traces rather than engineered scalar features.

Two granularities will be tested.

### 2.1 E0 — trial-level traces

$$
E0 = (neuron, hash, trial)
$$

Each row/sample corresponds to one neuron’s response to one trial of one stimulus hash.

E0 preserves trial-to-trial variability and may contain reliability/state information. It is larger, noisier, and computationally more expensive.

E0 should be used after the E1 pipeline works.

### 2.2 E1 — within-hash trial-averaged traces

$$
E1 = (neuron, hash)
$$

For each neuron and each hash, trials of the same hash are averaged.

This is not averaging across different hashes. It only averages repeated presentations of the same physical stimulus.

E1 is the recommended first CNN representation because it is cleaner, smaller, and closest to Tier G. It asks whether the temporal shape of the mean response to each oracle stimulus adds information beyond the scalar amplitude fingerprint.

### 2.3 Never average across different hashes

Responses to different `condition_hash` values must never be averaged together before modeling.

Each hash corresponds to a distinct physical stimulus. Averaging across hashes would destroy the stimulus-specific response fingerprint, which is exactly the structure that made Tier G successful.

Allowed:

$$
\text{average trials within the same } (neuron, hash)
$$

Not allowed:

$$
\text{average responses across different hashes}
$$

The model may aggregate predictions across hashes only at the end, when producing one neuron-level prediction.

---

## 3. Handling different trace lengths

The stimulus families have different trace lengths. A 1D CNN requires a consistent tensor shape inside a batch, but the first solution should not shrink the dataset by training separate family-specific models. Within one session, splitting again by family would make the training set too small and would make the first architecture test unnecessarily unstable.

The initial Phase-2 model should therefore use **one shared temporal encoder trained on all oracle hashes from the selected session**, across Clip, Monet2, and Trippy. Family is kept as metadata for later analysis, not used as a separate training split in the first pass.

### 3.1 Primary solution: pad traces to a common length

For the first unified E1 and E0 models, traces should be padded to the maximum trace length needed in the selected oracle set. Shorter traces are padded; longer traces are kept intact. This preserves the available temporal information and avoids discarding late response dynamics.

Cropping should not be the default because Phase 2 is explicitly asking whether temporal dynamics contain additional layer information. Cutting longer traces to the shortest family length may remove precisely the late/sustained/decay dynamics that the CNN is supposed to test. Cropping is acceptable only as a later sensitivity analysis, for example an “early response only” experiment.

For raw traces, padding with literal zero may introduce an artificial value if zero is outside the natural trace range. A safer first implementation is to pad with either the last observed value or the pre-stimulus baseline median for that trace. For baseline-subtracted traces, zero-padding is more defensible because zero approximately represents the trace baseline.

### 3.2 Masking

If implementation time allows, keep a valid-frame mask for each trace:

$$
m(t)=1 \quad 	ext{for real frames}
$$

$$
m(t)=0 \quad 	ext{for padded frames}
$$

The first implementation can use a standard 1D CNN on padded traces. The safer implementation uses the mask during final temporal pooling so padded frames do not contribute to the trace embedding. The model should not be interpreted using saliency on padded regions; padded frames are technical filler, not biological signal.

### 3.3 What not to do first

Do not train separate family-specific encoders in the first one-session feasibility notebook. That design is too data-hungry for the initial setting.

Do not crop as the primary model. Cropping should be a later sensitivity analysis.

Do not average traces across different hashes to create a fixed-size neuron trace. Different hashes are different physical stimuli and must remain separate observations.

### Initial recommendation

For the first feasibility notebook, use **all oracle hashes from one balanced low-mismatch session, one shared CNN encoder, padded traces, and neuron-level aggregation of row predictions**. Family-specific analyses can be done later by evaluating the trained all-family model separately on Clip, Monet2, and Trippy rows, or by training family-specific models only after the unified pipeline is stable.

---

## 4. Normalisation plan

The first single-session experiment should not start with aggressive normalisation. Since all neurons are from the same session, a pure session-identity shortcut is absent. Therefore, raw traces are a valid first debug input.

However, raw traces can still contain within-session acquisition confounds: depth-dependent brightness, baseline offset, neuropil contamination, motion correction quality, local signal-to-noise differences, and z-plane effects. Therefore, raw performance cannot be interpreted alone as temporal biological signal.

The initial normalisation ladder should be minimal:

### 4.1 Raw traces

$$
X_{raw}(t) = X(t)
$$

Use raw trace values as stored.

This is the first feasibility condition. It asks whether the CNN can learn from all information present in the trace.

### 4.2 Baseline-subtracted traces

For each trace, estimate the baseline from the pre-stimulus window:

$$
b = median(X(t_{pre}))
$$

Then subtract it from the full trace:

$$
X_{base-sub}(t) = X(t) - b
$$

This removes the vertical offset of each trial while preserving evoked amplitude and temporal shape.

Example:

$$
[100, 101, 100, 102, 130, 145] \rightarrow [0, 1, 0, 2, 30, 45]
$$

Baseline subtraction is useful because two traces may have identical evoked responses but different absolute fluorescence offsets. The CNN should not be allowed to rely only on the offset.

### 4.3 Do not use within-neuron normalisation in the first pass

Within-neuron normalisation is too aggressive for the first CNN experiment. It may remove biologically meaningful amplitude differences, and Phase 1 showed that amplitude features were strongly informative.

It can be considered later as a strict sensitivity analysis, but it should not be part of the initial one-session pipeline.

### Initial normalisation decision

The first notebook should compare:

1. E1 raw traces
2. E1 baseline-subtracted traces
3. E1 time-shuffled raw traces
4. E1 amplitude-only baseline

Then repeat the same logic for E0 if E1 works.

---

## 5. Controls required from the beginning

Even in one session, controls are necessary because raw traces may contain non-biological shortcuts.

### 5.1 Time-shuffled control

Shuffle the temporal order of frames within each trace while preserving the marginal amplitude distribution.

If the intact CNN and time-shuffled CNN perform similarly, the CNN is probably using amplitude distribution rather than temporal order.

If intact CNN outperforms time-shuffled CNN, there is evidence that temporal structure matters.

### 5.2 Amplitude-only baseline

Collapse each trace into simple amplitude summaries, for example mean, max, min, peak-to-baseline, AUC, and quantiles, then feed these to a small MLP or simple classifier.

This tests whether the CNN’s advantage comes from temporal structure or merely from flexible amplitude readout.

### 5.3 Optional matched engineered baseline

Use the existing Tier-A scalar features on the same session, same neurons, and same hashes.

This gives a direct local comparison between CNN traces and engineered trace summaries.

---

## 6. Model and scoring contract

### 6.1 Unit of prediction

The biological label is neuron-level. Therefore the final prediction must be neuron-level. A CNN may be trained on trace rows, but the reported metrics must be computed after converting the model outputs to one prediction per neuron.

The final neuron prediction is obtained from class probabilities or logits in the standard multiclass way: aggregate to a neuron-level probability vector, then use:

$$
\hat{y}_i = \arg\max_y p_i(y)
$$

This use of `argmax` is not specific to HGB. It is the standard decision rule for multiclass probabilistic classifiers and remains valid for CNNs. What matters is that `argmax` is applied after neuron-level aggregation, not independently to each row when computing the final biological prediction.

### 6.2 Design A — row-level CNN with probability aggregation

This is the simplest first implementation and should be used to debug the trace pipeline.

For E1, each sample is:

$$
(neuron, hash)
$$

The CNN outputs row-level probabilities:

$$
p(y \mid x_{i,h})
$$

The natural E1 aggregation is **hash-uniform aggregation**:

$$
p_i(y)= \frac{1}{H_i}\sum_h p(y\mid x_{i,h})
$$

Only this aggregation should be reported for E1 in the first stage, because E1 already has one row per hash after averaging repeated trials within the same hash. There is no separate trial-repeat imbalance left to correct at evaluation.

For E0, each sample is:

$$
(neuron, hash, trial)
$$

The CNN outputs:

$$
p(y \mid x_{i,h,r})
$$

Two aggregation schemes should be reported for E0:

**Row-uniform aggregation**:

$$
p_i(y)= \frac{1}{N_i}\sum_{h,r}p(y\mid x_{i,h,r})
$$

This treats every observed trial as one piece of evidence. It allows hashes with more repeated trials to contribute more votes.

**Hash-uniform aggregation**:

$$
p_{i,h}(y)=\frac{1}{R_h}\sum_r p(y\mid x_{i,h,r})
$$

$$
p_i(y)=\frac{1}{H_i}\sum_h p_{i,h}(y)
$$

This first combines repeated presentations of the same hash, then averages across hashes. It makes the E0 evaluation more directly comparable to E1, but it should not be declared a priori as the only headline. The main interpretation should follow the observed results.

Family-balanced aggregation is not part of the initial report. Clip has more oracle hashes by design, so forcing each family to contribute equally would impose an artificial stimulus balance. It can be considered later as a sensitivity analysis if there is evidence that family composition dominates the results.

### 6.3 Design B — set-CNN / DeepSets-style trace encoder

After Design A works, implement a more principled deep-learning version analogous to the DeepSets logic from Phase 1. The model should not average raw traces across different hashes. Instead, it encodes each trace separately, pools learned embeddings, and predicts once per neuron.

For E1:

$$
e_{i,h}=Encoder(x_{i,h})
$$

$$
e_i=Pool_h(e_{i,h})
$$

$$
p_i(y)=Classifier(e_i)
$$

For E0, the analogous structure is:

$$
e_{i,h,r}=Encoder(x_{i,h,r})
$$

then pool trials within hash and hashes within neuron, or use a masked set-pooling operation over all trace embeddings. The key distinction is that pooling occurs after trace encoding, not by averaging raw traces across different stimuli.

Design B is biologically attractive because it learns the neuron-level representation directly. Design A is computationally simpler and should come first; Design B should be run as a complementary model once the data pipeline is stable.

### 6.4 Cross-validation

Within a single session, use GroupKFold by `nucleus_id`, not random KFold.

All rows from the same neuron must stay in the same fold. This prevents leakage where the CNN sees some trials/hashes from a neuron during training and then predicts the same neuron in testing.

### 6.5 Metrics

Report:

- balanced accuracy
- macro-F1
- per-class recall
- confusion matrix
- L6 recall variance across folds

Balanced accuracy remains the primary metric because L6 is rare.

### 6.6 Loss function

Use class-weighted cross-entropy as the default.

If L6 collapses or becomes extremely unstable, try focal loss as a sensitivity analysis. Do not tune the loss extensively in the first notebook.

---
## 7. Staged computational plan

The Phase-2 CNN analysis is staged for computational and interpretability reasons. The first stages are feasibility/debugging analyses; later stages test replication and cross-session generalisation.

Controls are not repeated mechanically at every stage. The full control suite is run first on the one-session E1 Design A model to identify whether performance depends on baseline offset, temporal order, or amplitude summaries. Later stages use the strongest real trace condition, usually raw traces unless evidence suggests otherwise. Controls are reintroduced only when a model becomes a candidate for a scientific claim about temporal dynamics.

### Stage 2.1 — One-session E1 feasibility with Design A

Purpose: test whether the basic row-level CNN pipeline works on the cleanest raw-trace representation.

Session choice: start with **5_6 or 5_7**, after removing label-mismatch neurons.

Input:

$$
E1 = (neuron, hash)
$$

Each sample is the trial-averaged trace for one neuron and one oracle hash. Trials are averaged only within the same hash. Responses are never averaged across different hashes.

Use the oracle hashes. Pad traces to a common length rather than cropping in the primary model.

Model:

**Design A: row-level CNN + neuron-level probability aggregation**

For each row:

$$
CNN(x_{i,h}) \rightarrow p(y \mid x_{i,h})
$$

Then aggregate probabilities across hashes for each neuron:

$$
p_i(y)=\frac{1}{H_i}\sum_h p(y \mid x_{i,h})
$$

Final prediction:

$$
\hat{y_i}=\arg\max_y p_i(y)
$$

For E1, report only **hash-uniform aggregation**, because one row already corresponds to one hash.

Run:

1. E1 raw CNN
2. E1 baseline-subtracted CNN
3. E1 time-shuffled raw CNN
4. E1 amplitude-summary baseline

The amplitude-summary baseline is not a CNN. It is a non-temporal model, for example an MLP or tabular classifier, trained on scalar amplitude summaries from the same E1 traces. It tests whether the CNN adds information beyond response magnitude.

Report neuron-level metrics:

- balanced accuracy
- macro-F1
- per-class recall
- confusion matrix
- predicted class distribution
- class precision/F1, especially for L6

Interpretation:

- If raw and baseline-subtracted both work, the signal is not only vertical baseline offset.
- If raw works but baseline-subtracted collapses, the model probably relied on baseline/offset information.
- If time-shuffled performs similarly to intact raw traces, temporal order is not contributing measurably.
- If the amplitude-summary baseline performs similarly to intact raw traces, the CNN is amplitude-equivalent.
- If intact raw traces beat both time-shuffled and amplitude-summary controls, temporal structure may contribute and further temporal interpretation becomes justified.

### Stage 2.1b — One-session E1 architecture check with Design B

Purpose: test whether a set-CNN architecture, which is closer to the neuron-level structure of the dataset, improves over row-level probability aggregation.

Run this only after Stage 2.1 confirms that Design A learns above chance.

Input:

$$
E1 = (neuron, hash)
$$

Model:

**Design B: set-CNN / DeepSets-style embedding pooling**

For each hash trace:

$$
CNN(x_{i,h}) \rightarrow e_{i,h}
$$

Pool embeddings across hashes:

$$
e_i=\frac{1}{H_i}\sum_h e_{i,h}
$$

Classify once at neuron level:

$$
p_i(y)=Classifier(e_i)
$$

Final prediction:

$$
\hat{y_i}=\arg\max_y p_i(y)
$$

Run first:

1. E1 raw Design B

If Design B raw improves meaningfully over Design A raw, optionally run:

2. E1 baseline-subtracted Design B

Do not run time-shuffled or amplitude-summary controls for Design B immediately. Those controls are needed only if Design B becomes a candidate main architecture for temporal interpretation.

Interpretation:

- If Design B improves over Design A, pooling learned embeddings before classification is beneficial and better matches the biological unit of prediction.
- If Design B is similar or worse, Design A remains the simpler and more stable architecture for scaling.
- A meaningful improvement should not be a tiny fold-noise difference. Prefer switching architecture only if the gain is approximately ≥ 0.03 balanced accuracy and is accompanied by better macro-F1 or more balanced per-class recall.

### Stage 2.1c — Optional E0 feasibility check with Design A
Run E0 Design A both with neuron-level and hash-level aggregation in probabilities prediction and and compare results.


### Stage 2.2 — Hash-aware CNNs after E1/E0 diagnostic failures

#### Purpose

Stage 2.2 is introduced after the first CNN diagnostics showed that the basic trace-based models can learn some layer-relevant signal, but still fail to predict some classes reliably, especially L2/3.

By this point, the following models have already been tested:

1. E1 Design A: row-level CNN with neuron-level probability aggregation;
2. E1 Design B: neuron-level set-CNN with embedding pooling;
3. E0 Design A: trial-level row CNN with neuron-level probability aggregation.

The common issue is that L2/3 remains poorly predicted across the tested architectures. This suggests that the model may not be using the stimulus-specific response fingerprint effectively. In the current trace-only CNNs, each input row contains the calcium trace, but the model is not explicitly told which oracle stimulus generated that trace.

This is a limitation because the biological signal may not be simply:

$$
\text{high response} \rightarrow \text{layer}
$$

but rather a stimulus-indexed pattern such as:

$$
\text{high response to hash A + low response to hash B} \rightarrow \text{layer}
$$

Therefore, Stage 2.2 tests whether explicitly providing `condition_hash` identity improves layer decoding before moving to more expensive or broader analyses, such as E0 Design B, additional engineered features, replication on another session, or multi-session LOSO.

The key question is:

$$
\text{Does stimulus identity help the CNN interpret raw calcium traces?}
$$

This stage should be understood as a targeted diagnostic step, not as a full scaling stage.

---

#### Core principle

Hash identity must be added without averaging raw responses across different hashes.

Each trace remains a separate stimulus-locked observation. The model may combine information across hashes only after each trace has been encoded separately.

Allowed:

$$
x_{i,h}(t) \rightarrow Encoder_{CNN} \rightarrow e_{i,h}
$$

then aggregate embeddings or probabilities across hashes.

Not allowed:

$$
\frac{1}{H_i}\sum_h x_{i,h}(t)
$$

because this would average calcium traces evoked by different physical stimuli and destroy the stimulus-specific response fingerprint.

---

#### Stage 2.2a — Primary model: E1 Design B with hash embeddings

The primary hash-aware model should be built on E1 Design B, because Design B already matches the biological unit of prediction: one set of hash responses corresponds to one neuron, and the model predicts the layer once per neuron.

The input remains:

$$
E1 = (neuron, hash)
$$

Each sample is the trial-averaged calcium trace of neuron $i$ in response to oracle hash $h$:

$$
x_{i,h}(t)
$$

Trials are averaged only within the same `(neuron, hash)` pair. Responses from different hashes are not averaged.

The trace is passed through the shared CNN encoder:

$$
e_{i,h}^{trace} = Encoder_{CNN}(x_{i,h})
$$

The hash identity is passed through a learned embedding layer:

$$
e_h^{hash} = Embedding(condition\_hash_h)
$$

The trace embedding and hash embedding are concatenated:

$$
z_{i,h} = [e_{i,h}^{trace}, e_h^{hash}]
$$

A small MLP maps the concatenated vector to a stimulus-aware row embedding:

$$
u_{i,h} = MLP(z_{i,h})
$$

The model then aggregates all hash-specific embeddings belonging to the same neuron:

$$
u_i = Pool_h(u_{i,h})
$$

Finally, the neuron-level classifier predicts the layer:

$$
p_i(y) = Classifier(u_i)
$$

$$
\hat{y}_i = \arg\max_y p_i(y)
$$

This model should be reported as:

$$
\text{E1 Design B + hash identity}
$$

not as a pure trace-only CNN, because it uses stimulus metadata in addition to the calcium trace.

---

#### Stage 2.2b — Pooling choice: mean pooling first, attention pooling second

The first E1 hash-aware Design B model should use mean pooling:

$$
u_i = \frac{1}{H_i}\sum_h u_{i,h}
$$

This is the cleanest test of whether hash identity helps. It keeps the aggregation rule simple and directly comparable to the previous trace-only Design B model.

The primary comparison is therefore:

$$
\text{E1 Design B trace-only mean pooling}
$$

versus:

$$
\text{E1 Design B + hash embedding + mean pooling}
$$

If hash embeddings improve balanced accuracy, macro-F1, or L2/3 recall under mean pooling, then stimulus identity is likely useful.

If the mean-pooling hash-aware model is stable, the next variant is attention pooling.

Attention pooling allows the model to learn that not all oracle hashes are equally informative for layer prediction. For each stimulus-aware embedding $u_{i,h}$, compute an attention score:

$$
a_{i,h} = w^\top \tanh(Wu_{i,h})
$$

Normalize scores across hashes belonging to the same neuron:

$$
\alpha_{i,h} =
\frac{\exp(a_{i,h})}
{\sum_{h'} \exp(a_{i,h'})}
$$

Then compute the neuron-level embedding as:

$$
u_i = \sum_h \alpha_{i,h}u_{i,h}
$$

The classifier then predicts:

$$
p_i(y) = Classifier(u_i)
$$

Attention pooling is appropriate for the scientific objective because the layer signal may depend on a non-uniform stimulus-response fingerprint. In other words, some oracle hashes may be more informative than others for distinguishing L2/3 from the remaining layers.

However, attention pooling is also more flexible and therefore more prone to overfitting. It should only be preferred over mean pooling if it improves validation metrics, not only training metrics.

The main comparison within Stage 2.2 is:

$$
\text{trace-only Design B}
$$

versus:

$$
\text{hash-aware Design B + mean pooling}
$$

versus:

$$
\text{hash-aware Design B + attention pooling}
$$

The attention model should be used as the candidate model for Stage 2.3 only if it improves validation balanced accuracy and macro-F1, and especially if it improves L2/3 recall without collapsing another class.

---

#### Stage 2.2c — Optional sanity check: E1 Design A with hash embeddings

Hash embeddings can also be added to E1 Design A, but only as a sanity check.

This is useful because Design A is cheaper and simpler than Design B. It can test whether hash identity improves row-level evidence before or alongside the more structured set model.

For each E1 row:

$$
x_{i,h}(t)
$$

the CNN encoder produces:

$$
e_{i,h}^{trace} = Encoder_{CNN}(x_{i,h})
$$

The hash embedding layer produces:

$$
e_h^{hash} = Embedding(condition\_hash_h)
$$

The two vectors are concatenated:

$$
z_{i,h} = [e_{i,h}^{trace}, e_h^{hash}]
$$

Then a classifier predicts row-level probabilities:

$$
p(y \mid x_{i,h}, h) = Classifier(MLP(z_{i,h}))
$$

Neuron-level probabilities are obtained through hash-uniform aggregation:

$$
p_i(y) = \frac{1}{H_i}\sum_h p(y \mid x_{i,h}, h)
$$

Final prediction:

$$
\hat{y}_i = \arg\max_y p_i(y)
$$

This model should not be the main hash-aware architecture unless Design B is unstable. It remains a row-level model because it classifies each `(neuron, hash)` row independently before aggregation.

Its role is diagnostic:

- if E1 Design A improves with hash embeddings, then hash identity carries useful row-level information;
- if E1 Design A improves but E1 Design B does not, the set aggregation mechanism may be the problem;
- if neither improves, then poor L2/3 prediction is unlikely to be fixed by stimulus identity alone.

---

#### Stage 2.2d — E0 with hash embeddings only if E1 hash identity is useful

E0 hash-aware models should not be the first hash-embedding experiment.

E0 is computationally more expensive because each sample is:

$$
E0 = (neuron, hash, trial)
$$

and the number of traces is much larger than in E1. Therefore, hash embeddings should be extended to E0 only if E1 hash-aware models give favorable results.

A favorable result means at least one of the following:

1. improved L2/3 recall;
2. improved macro-F1;
3. improved balanced accuracy;
4. more balanced predicted class distribution;
5. better stability across folds.

If E1 hash-aware Design B improves the model, then E0 can be tested to ask whether trial-to-trial variability adds information beyond the trial-averaged E1 representation.

The first E0 hash-aware model should be Design A, because it is cheaper and has already been implemented in trace-only form.

For each E0 row:

$$
x_{i,h,r}(t)
$$

where $r$ indexes repeated trials of the same hash, the model computes:

$$
e_{i,h,r}^{trace} = Encoder_{CNN}(x_{i,h,r})
$$

$$
e_h^{hash} = Embedding(condition\_hash_h)
$$

$$
z_{i,h,r} = [e_{i,h,r}^{trace}, e_h^{hash}]
$$

$$
p(y \mid x_{i,h,r}, h) = Classifier(MLP(z_{i,h,r}))
$$

Two neuron-level aggregation schemes should be reported.

Row-uniform aggregation:

$$
p_i(y)=\frac{1}{N_i}\sum_{h,r}p(y \mid x_{i,h,r}, h)
$$

Hash-uniform aggregation:

$$
p_{i,h}(y)=\frac{1}{R_{i,h}}\sum_r p(y \mid x_{i,h,r}, h)
$$

$$
p_i(y)=\frac{1}{H_i}\sum_h p_{i,h}(y)
$$

Hash-uniform aggregation is more directly comparable to E1, because each hash contributes equally after averaging over repeated trials.

E0 Design B with hash embeddings should be treated as a later, higher-cost model. It should only be attempted if:

1. E1 hash-aware Design B improves clearly;
2. E0 Design A with hash embeddings suggests that trial-level traces are useful;
3. compute resources allow a hierarchical set model.

The E0 Design B architecture would be:

$$
e_{i,h,r}^{trace}=Encoder_{CNN}(x_{i,h,r})
$$

$$
e_h^{hash}=Embedding(condition\_hash_h)
$$

$$
u_{i,h,r}=MLP([e_{i,h,r}^{trace}, e_h^{hash}])
$$

First pool repeated trials within the same hash:

$$
u_{i,h}=Pool_r(u_{i,h,r})
$$

Then pool hashes within the same neuron:

$$
u_i=Pool_h(u_{i,h})
$$

Finally:

$$
p_i(y)=Classifier(u_i)
$$

$$
\hat{y}_i=\arg\max_y p_i(y)
$$

The first E0 Design B version should use mean pooling over trials and hashes. Attention pooling over hashes can be considered only if the E1 attention model was beneficial.

---

#### Cross-validation and leakage control

All Stage 2.2 models must use neuron-grouped cross-validation.

Within a single session, use GroupKFold by `nucleus_id`.

All rows belonging to the same neuron must stay in the same fold. This is mandatory because both E1 and E0 contain multiple rows per neuron.

If rows from the same neuron appear in both training and validation, validation performance will be inflated by leakage.

For E1 Design B, the model receives a neuron-level set:

$$
\{(x_{i,h}, condition\_hash_h)\}_{h=1}^{H_i}
$$

For E0 Design B, the model receives a hierarchical neuron-level set:

$$
\{(x_{i,h,r}, condition\_hash_h)\}_{h,r}
$$

Therefore, the train/validation split must be performed at neuron level before constructing batches.

---

#### Recommended run order

The recommended Stage 2.2 order is:

1. E1 Design B + hash embedding + mean pooling.
2. E1 Design B + hash embedding + attention pooling, only if the mean-pooling model is stable.
3. Optional E1 Design A + hash embedding as a sanity check.
4. E0 Design A + hash embedding only if E1 hash identity improves performance.
5. E0 Design B + hash embedding only if E1 and E0 Design A results justify the additional computational cost.

This stage should not expand immediately to another session. The purpose is first to determine whether the poor L2/3 prediction is partly due to missing stimulus identity.

Only the best stable model from Stage 2.2 should be carried forward to Stage 2.3 for replication on a second clean session.

---

#### Decision criteria before moving to Stage 2.3

Move to Stage 2.3 with the best hash-aware model only if it improves validation performance in a meaningful and interpretable way.

Useful evidence includes:

1. improved L2/3 recall;
2. improved macro-F1;
3. improved balanced accuracy;
4. no collapse of L4, L5, or L6 recall;
5. predicted class distribution closer to the true class distribution;
6. stable results across folds.

If hash-aware Design B improves L2/3 recall but damages another class severely, it should be reported as a partial improvement rather than as the new main model.

If hash-aware Design B does not improve over trace-only Design B, then the next step should not be more hash-aware complexity. Instead, consider:

1. replication of the best existing model on a second session;
2. adding engineered amplitude features;
3. comparing directly against Tier G on the same neurons;
4. testing whether label noise or class imbalance explains the L2/3 failure.

---

#### Interpretation

If E1 Design B with hash embeddings improves L2/3 recall, this suggests that the missing information was not only temporal response shape, but the association between response shape and stimulus identity.

If attention pooling improves over mean pooling, this suggests that only a subset of oracle hashes is strongly informative for layer decoding.

If hash embeddings improve balanced accuracy but not L2/3 recall, then the model benefits from stimulus identity but still does not solve the main class-specific failure.

If hash embeddings improve E1 but not E0, trial-level noise may obscure the stimulus-indexed signal.

If hash embeddings do not improve either E1 or E0, then poor L2/3 prediction is unlikely to be solved by giving the CNN stimulus identity alone. In that case, the limitation may come from insufficient data, weak temporal signal, class imbalance, label noise, or the fact that Tier G-style scalar fingerprints are better suited to this classification task.

The final report should clearly distinguish between:

$$
\text{trace-only CNN}
$$

and:

$$
\text{trace + hash identity CNN}
$$

because the second model uses additional stimulus metadata.


### Stage 2.3 — Balanced-session pooled GKF and LOSO

Purpose: test the best candidate CNN from the one-session diagnostics on multiple balanced sessions, using both within-pooled validation and cross-session validation.

This stage replaces the earlier second-session replication and two-session transfer steps. Since CNN training is computationally expensive, the analysis should not spend compute on intermediate checks that do not directly answer the main biological question. Instead, after Stage 2.2 selects the best stable hash-aware or trace-only candidate, the next step is to move directly to a pooled balanced-session analysis.

Use the balanced four-class sessions:

$$
5\_6,\ 5\_7,\ 6\_2,\ 6\_4,\ 6\_6,\ 6\_7
$$

If compute is limited, start with a smaller subset of 4 or 5 balanced sessions, preferably prioritising cleaner and lower-mismatch sessions. The exact subset should be reported explicitly.

Continue to use the consistent-only subset first:

$$
layer\_label = celltype\_layer
$$

Use the best stable candidate from Stage 2.2. The priority order is:

1. E1 Design B + hash embedding + mean pooling, if hash identity improved L2/3 recall or macro-F1.
2. E1 Design B + hash embedding + attention pooling, only if attention improved validation metrics without clear overfitting.
3. E1 Design A + hash embedding, if Design B was unstable but hash identity improved row-level evidence.
4. E0 Design A + hash embedding, only if E0 showed a real advantage over E1.
5. Trace-only models only if hash-aware models did not improve performance.

The default input remains E1 raw traces unless previous controls showed that raw performance depended strongly on baseline offset. Use baseline-subtracted traces only if they were clearly more reliable or scientifically cleaner.

This stage should be implemented as one notebook with two validation modes:

1. pooled GroupKFold;
2. leave-one-session-out validation.

The pooled GroupKFold result is used for model stability and model selection.

The LOSO result is used for biological interpretation.

---

#### Stage 2.3a — Pooled GroupKFold

Run pooled GroupKFold by `nucleus_id` across the selected balanced sessions.

All rows belonging to the same neuron must stay in the same fold. This is mandatory because each neuron contributes multiple hash rows in E1 and multiple trial rows in E0.

For E1 Design B, the model receives neuron-level sets:

$$
\{(x_{i,h}, condition\_hash_h)\}_{h=1}^{H_i}
$$

For E0 Design A, the model receives trial-level rows:

$$
(neuron, hash, trial)
$$

but scoring must still be done after neuron-level aggregation.

Report:

- balanced accuracy;
- macro-F1;
- per-class recall;
- confusion matrix;
- predicted class distribution;
- class precision and F1;
- L2/3 recall specifically;
- L6 recall specifically.

Use a matched Tier G benchmark for the same session subset and same consistent-only neuron set if available.

Do not compare this result to the global full-13 Tier G number unless the CNN is trained and evaluated on the full 13-session dataset.

Interpretation:

- If pooled GKF improves over the one-session result, additional sessions provide useful training variation.
- If pooled GKF worsens substantially, session heterogeneity may dominate the raw-trace representation.
- If L2/3 remains poorly predicted, hash identity and additional data did not solve the main class-specific failure.
- If the model becomes biased toward one class, inspect predicted class distributions and consider class-weighted loss or class-balanced sampling.
- If attention pooling performs better in training but not validation, keep mean pooling as the safer architecture.

Important limitation:

Pooled GKF is not a cross-session generalisation test. Since train and validation folds may contain neurons from the same sessions, the model may still exploit session-specific structure. Therefore, pooled GKF should be treated as a stability and model-selection result, not as the main biological endpoint.

---

#### Stage 2.3b — Balanced-session LOSO

Run leave-one-session-out validation across the same balanced-session subset.

For each held-out session \(s\), train on all other selected sessions and test on session \(s\).

For example, if six balanced sessions are used:

$$
5\_6,\ 5\_7,\ 6\_2,\ 6\_4,\ 6\_6,\ 6\_7
$$

then each fold holds out one full session.

Use only the best stable model from the pooled GKF analysis. Do not run every architecture under LOSO unless compute allows.

Compare against the matched Tier G benchmark on the same session subset and the same consistent-only neuron set:

$$
\Delta_s = CNN_s - TierG_s
$$

Report:

- balanced accuracy per held-out session;
- macro-F1 per held-out session;
- per-class recall per held-out session;
- predicted class distribution per held-out session;
- confusion matrix per held-out session;
- mean CNN performance across held-out sessions;
- mean Tier G performance across held-out sessions;
- paired difference:

$$
\Delta_s = CNN_s - TierG_s
$$

- mean paired difference:

$$
\bar{\Delta} = \frac{1}{S}\sum_s \Delta_s
$$

If feasible, report a bootstrap confidence interval for \(\bar{\Delta}\). If the number of held-out sessions is sufficient, also report a paired sign test or paired permutation test.

Interpretation:

- If pooled GKF is strong but LOSO collapses, the CNN is probably learning session-specific structure.
- If LOSO remains above chance with reasonable per-class recall, the trace representation contains some cross-session layer-discriminative information.
- If LOSO improves L2/3 recall relative to previous one-session models, then the additional sessions helped the main failure mode.
- If LOSO remains poor for L2/3, the hash-aware CNN did not solve the main biological classification problem.
- If the CNN approaches or beats the matched Tier G benchmark, the raw trace representation is competitive with the Phase-1 scalar oracle-hash fingerprint.
- If the CNN falls clearly below Tier G, then raw temporal traces do not currently justify further expensive CNN scaling.

Controls are not repeated immediately for every architecture. They are reintroduced only if the LOSO CNN becomes scientifically competitive.

If the CNN is competitive with Tier G, rerun the relevant controls under the same LOSO protocol:

1. time-shuffled traces;
2. amplitude-summary baseline;
3. baseline-subtracted traces, if baseline offset was a concern.

A temporal-dynamics claim is allowed only if the intact CNN beats both the time-shuffled control and the amplitude-summary baseline.

---

#### Decision after Stage 2.3

Stage 2.3 is the final planned Phase-2 CNN generalisation test.

Do not run full 13-session LOSO by default.

Full 13-session LOSO should be considered only if the balanced-session LOSO result is unexpectedly strong and clearly worth scaling. Otherwise, the analysis should move to Stage 3.

Move to Stage 3 if one of the following is true:

1. the CNN performs below Tier G under balanced-session LOSO;
2. the CNN is unstable across held-out sessions;
3. L2/3 remains poorly predicted;
4. performance is amplitude-equivalent rather than temporally specific;
5. the model needs additional structured information to improve biological decoding.

Stage 3 should then test whether adding engineered or anatomical/functional features improves layer decoding beyond raw traces alone, for example through a multichannel or multimodal architecture.

---

#### Final interpretation of Phase 2

If the balanced-session LOSO CNN is strong, Phase 2 supports the idea that raw calcium traces contain cross-session layer-discriminative information.

If pooled GKF is strong but LOSO is weak, Phase 2 shows that CNNs can learn within-session or within-distribution trace structure, but this structure does not generalise reliably across scans.

If hash embeddings improve pooled GKF but not LOSO, stimulus identity helps within-distribution decoding but does not solve cross-session generalisation.

If L2/3 remains poorly predicted, the main limitation is class-specific and should be carried forward explicitly into Stage 3.

If the CNN does not beat the matched Tier G benchmark, the conclusion should be framed as:

Raw temporal traces, with the tested CNN architectures, do not provide robust cross-session information beyond the scalar oracle-hash response fingerprint.

This is still a useful result because it motivates the next stage: adding structured features or multichannel inputs rather than further scaling trace-only CNNs.

---

## 8. When to add engineered tiers or multimodal fusion

Do not add engineered tiers in the first CNN experiments.

The first question is whether the raw trace encoder itself learns useful information. Adding Tier D, behaviour, reliability, or Tier G-like amplitude features too early would make interpretation unclear: a weak CNN could be rescued by tabular features, and we would not know whether temporal traces helped.

The correct order is:

1. CNN-only on traces.
2. CNN-only controls: time-shuffled and amplitude-only.
3. Multi-session CNN-only generalisation.
4. Only then: multimodal fusion.

Possible later fusion model:

$$
CNN(trace) + TierD(stimulus\ content) + behaviour/reliability \rightarrow layer
$$

This should be reported as a follow-up, not as the primary Phase-2 result.

---

## 9. Family analyses

Family-specific CNNs are scientifically useful but should not be the first step.

Phase 1 already showed that layer signatures differ by stimulus family: Monet2 was strongest, Clip and Trippy behaved differently, and cross-family transfer was incomplete.

After the all-oracle E1 pipeline works, first evaluate the trained all-family model separately on Clip, Monet2, and Trippy rows. This does not shrink the training set and gives an initial readout of which family contributes the strongest evidence.

Only later, if enough data and compute remain, train family-specific CNNs:

1. Clip-only
2. Monet2-only
3. Trippy-only

These family-specific models should be interpreted as stimulus-family analyses, not as the main Phase-2 headline. They should not be used for first-stage architecture debugging within one session because the training sets would be too small.

---

## 10. Reportable interpretation of early stages

The single-session experiments should be reported, but with restricted language.

Correct framing:

> Because full raw-trace CNN training is computationally expensive, we first performed a single-session feasibility analysis on a balanced four-class session after excluding layer/cell-type mismatch neurons. This analysis tested whether the proposed 1D CNN architecture could learn layer-relevant structure from raw calcium traces under a low-confound within-session setting. It was not intended as the final cross-session generalisation result.

If the one-session CNN succeeds:

> The one-session result shows that raw traces contain sufficient information for within-session layer decoding and justifies scaling the architecture to additional sessions.

If it fails:

> The failure suggests that either the architecture, preprocessing, trace window, or raw trace representation is insufficient; full LOSO should not be attempted before debugging these issues.

If raw succeeds but controls fail:

> The CNN decodes layer from raw traces, but the source of the signal is not isolated. Performance may reflect amplitude, baseline, or within-session acquisition structure rather than temporal response dynamics.

If intact CNN beats time-shuffled and amplitude-only controls:

> There is preliminary evidence that temporal structure contributes to layer decoding within session. Cross-session validation remains required.

---

## 11. Final scientific success criteria

The final Phase-2 claim depends on the strongest completed stage.

### Weak but useful claim

If only one-session CNN works:

> Raw calcium traces contain layer-discriminative information within at least one balanced recording session, justifying further CNN scaling.

### Intermediate claim

If one-session and two-session transfer work:

> The CNN learns trace-based layer signatures that are not purely idiosyncratic to one session.

### Strong claim

If balanced-session LOSO works and beats Tier G on paired held-out sessions:

> Raw temporal traces improve cross-session layer decoding beyond the Phase-1 oracle fingerprint on balanced sessions.

### Full Phase-2 claim

If full 13-session LOSO works and beats Tier G in paired comparison, and intact CNN beats time-shuffled and amplitude-only controls:

> Raw temporal calcium traces contain layer-discriminative temporal structure beyond the engineered Phase-1 response fingerprint.

### Kinetics claim

A kinetics claim is allowed only if all of the following hold:

1. CNN beats Tier G under paired LOSO.
2. CNN beats time-shuffled trace control.
3. CNN beats amplitude-only baseline.
4. Saliency/occlusion localises the signal to stimulus-evoked windows, not baseline or padded regions.

Without these controls, the claim must remain: “the CNN extracts additional trace-based information,” not “kinetics explain the improvement.”

---

## 12. Bottom line

The revised Phase-2 plan is computationally staged and scientifically conservative.

Start with one clean balanced session, remove label-mismatch neurons, and test E1 raw traces first using one shared CNN encoder and padded traces. Then compare against baseline-subtracted, time-shuffled, and amplitude-only controls. Only after the trace CNN works within one session should E0, Design B, a second session, two-session transfer, balanced-session pooling, and LOSO be attempted.

Do not average across different hashes. Do not add engineered tiers before proving the trace encoder itself is meaningful. Do not use within-neuron normalisation in the first pass. Raw traces are acceptable for one-session debugging, but biological interpretation requires baseline and temporal-order controls.

This strategy preserves the scientific logic of Phase 2 while making the computation feasible.
