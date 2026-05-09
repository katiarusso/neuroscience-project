# Label-sensitivity analysis: layer label vs cell-type-derived layer

## Purpose of the analysis

The goal of this sensitivity analysis was to test whether the Phase 1 layer-decoding result depends on a small disagreement between two anatomical label sources:

1. `layer_label`, the depth-derived layer label used as the Phase 1 target.
2. `celltype_layer`, the layer implied by the AIBS excitatory cell-type label, using the mapping:
   - `23P` -> L2/3
   - `4P` -> L4
   - `5P-IT`, `5P-ET`, `5P-NP` -> L5
   - `6P-IT`, `6P-CT` -> L6

This matters because the project uses `layer_label` as the supervised target for Phase 1. If a non-negligible number of neurons have a layer label that disagrees with their morphology-based cell type, the model may be trained on partly ambiguous labels. The purpose of the analysis was not to decide which label source is definitively correct, but to ask whether this ambiguity changes the modelling conclusion and whether the ambiguity itself reveals useful structure in the dataset.

The main conclusion is that the Phase 1 result is robust. The mismatch is real and biologically/methodologically informative, but it does not materially change the Tier G layer-decoding result.

## Size of the mismatch

The working population contains 8,895 V1 excitatory best-only matched neurons. Of these, 8,488 neurons have agreement between `layer_label` and `celltype_layer`, while 407 neurons are mismatched. This corresponds to:

$$
\frac{407}{8895} = 4.6\%
$$

So the disagreement is small at the whole-dataset level. It is not large enough, by itself, to explain the main Phase 1 decoding result.

However, the mismatch is not uniformly distributed across the dataset. It is concentrated in deep layers: L5 has an 11.8% mismatch rate and L6 has a 16.3% mismatch rate. This means that the ambiguity is most relevant for deep-layer analyses and especially for Phase 3 subtype decoding, where L5 and L6 are the main target layers.

## Check 1: distribution of mismatches by layer and by session

The first check asked whether mismatches are uniformly distributed across layers and sessions. Uniformly distributed mismatches would suggest a minor background annotation noise. Concentrated mismatches would suggest either biological boundary structure or a session-specific labelling problem.

The results show both layer structure and session structure.

By layer, mismatches are enriched in L5 and L6. This is biologically plausible because deeper layer boundaries and deep-layer pyramidal subclasses are harder to separate cleanly. L5 and L6 also contain morphologically diverse excitatory populations, so disagreement between a depth-derived layer label and a morphology-derived cell type is not surprising.

By session, the mismatch is highly concentrated. Four sessions, `6_2`, `6_4`, `6_6`, and `6_7`, account for 385 of the 407 mismatched neurons, i.e. 94.6% of all mismatches. This is the strongest methodological warning from the analysis. A mismatch pattern this concentrated is unlikely to be only random biological ambiguity. It suggests that these scans may have specific label-assignment, field-of-view, depth-boundary, or registration characteristics that should be mentioned in the methods or discussed with the professors.

This does not invalidate the Phase 1 result, because the robustness check below shows that removing all mismatched neurons barely changes performance. But it is still a meaningful dataset insight: label ambiguity is not evenly spread across the acquisition protocol.

## Check 2: refitting Tier G after excluding mismatched neurons

The second check was the key robustness test. The model was refit after removing all 407 mismatched neurons, leaving only the 8,488 consistent neurons. The same Tier G HGB model was evaluated under both GroupKFold and LOSO.

The original Phase 1 Tier G result was:

| Evaluation | Original Tier G HGB |
|---|---:|
| GroupKFold | 0.663 +/- 0.008 |
| LOSO | 0.497 +/- 0.105 |

After excluding mismatches:

| Evaluation | Consistent-only Tier G HGB | Delta |
|---|---:|---:|
| GroupKFold | 0.660 +/- 0.015 | -0.003 |
| LOSO | 0.508 +/- 0.108 | +0.011 |

This is the most important result. Removing the ambiguous neurons does not improve the model in a meaningful way and does not reduce performance. The GroupKFold difference is only -0.003, and the LOSO result actually increases slightly by +0.011. These differences are small relative to fold/session variability.

Therefore, the Phase 1 headline result is robust to the label mismatch. The model's performance is not driven by the 407 ambiguous neurons, and the mismatch does not require changing the Phase 1 pipeline.

The interpretation is that the layer-decoding signal is present in the consistent majority of the dataset. The mismatched neurons may add biological ambiguity, but they are not responsible for the observed decoding accuracy.

## Check 3: are mismatches close to layer boundaries?

The third check examined whether mismatched neurons are located near depth boundaries. This is important because there are three possible explanations for the mismatch:

1. The cell-type label is wrong.
2. The depth-derived layer label is wrong.
3. The neuron is genuinely borderline: its soma lies near a layer interface, and morphology/depth assignment may reasonably disagree.

The plot of `pt_position_y` shows that many red mismatched neurons lie at layer transitions, especially around the L2/3-L4, L4-L5, and L5-L6 interfaces. This supports the boundary-cell interpretation for a substantial part of the mismatch.

There is one important nuance. The pooled histogram of `|pt_position_y - layer median|` reports KS p = 1.00e+00, so the global pooled distribution does not show a useful overall matched-vs-mismatched separation. This pooled KS test should not be overinterpreted because it mixes layers with different depth ranges and different mismatch mechanisms. The more informative reading is the per-layer analysis.

Per-layer tests show a strong boundary/centroid-distance effect for L2/3, L4, and L5. The notebook reports Mann-Whitney p < 1e-17 for these layers. In those layers, mismatched neurons tend to sit farther from the median depth of their assigned layer, consistent with being closer to a layer interface. Operationally, the analysis describes roughly 75% of mismatches as boundary-related.

L6 is the exception. L6 mismatches do not show a boundary effect; the L6 p-value is 0.74. These neurons sit at typical L6 depths but their morphology/cell-type mapping points to L5. This means that L6 mismatches are not well explained as simple L5-L6 boundary cells. They may reflect cell-type annotation errors, morphology-depth ambiguity, or genuine subtype admixture between deep-layer pyramidal classes. Without histological ground truth, this cannot be resolved directly.

The meaningful dataset insight is therefore not simply “some labels are wrong.” A better statement is:

Most mismatch structure is compatible with boundary ambiguity in L2/3, L4, and L5, while L6 shows a distinct mismatch pattern that may reflect deeper annotation or biological ambiguity.

## Check 4: what does the trained model predict on mismatched neurons?

The fourth check used out-of-fold Tier G HGB predictions from the full Phase 1 setting. The model was trained on the original `layer_label`, then predictions were examined only for the 407 mismatched neurons.

For each mismatched neuron, the question was whether the model predicted:

1. `layer_label`, the depth-derived label used during training;
2. `celltype_layer`, the layer implied by cell type/morphology;
3. neither.

The overall result was:

| Prediction target on mismatched neurons | Count | Percent |
|---|---:|---:|
| Predicted = `layer_label` | 218 | 53.6% |
| Predicted = `celltype_layer` | 119 | 29.2% |
| Predicted = neither | 70 | 17.2% |

The model mostly predicts the depth-derived label. This is expected because the model was trained on `layer_label`. It also supports the idea that the model is learning the depth-based labelling structure rather than simply converting cell type into layer.

However, the per-direction breakdown is more informative than the global average.

| Depth label | Cell-type-implied layer | n | Predicted depth label | Predicted cell-type layer | Interpretation |
|---|---:|---:|---:|---:|---|
| L2/3 | L4 | 87 | 58.6% | 21.8% | Model follows depth label. |
| L4 | L2/3 | 38 | 42.1% | 52.6% | Cell-type-implied layer wins. |
| L4 | L5 | 32 | 53.1% | 25.0% | Model follows depth label. |
| L5 | L4 | 158 | 51.3% | 25.9% | Model follows depth label. |
| L5 | L6 | 33 | 42.4% | 48.5% | Cell-type-implied layer wins. |
| L6 | L5 | 59 | 66.1% | 25.4% | Model follows depth label. |

Two specific mismatch directions are especially interesting.

First, among 23P cells labelled as L4, the model predicts L2/3 in 52.6% of cases. Since 23P is the canonical superficial pyramidal morphology, this suggests that some of these neurons may have a functional fingerprint closer to their morphology-implied layer than to their depth-derived label.

Second, among 6P cells labelled as L5, the model predicts L6 in 48.5% of cases. Again, this suggests that in some deep-layer mismatches, the model can detect a functional signature aligned with the morphology-implied identity.

This is meaningful, but it should be reported cautiously. The model was trained on depth-derived labels, not morphology-derived labels. Therefore, when it predicts the cell-type-implied layer, we cannot claim that it has “corrected” the label. We can only say that in a few morphologically extreme mismatch directions, the functional fingerprint sometimes aligns more with cell type/morphology than with the assigned depth label.

## Biological interpretation

The analysis suggests that layer and cell type are strongly related but not perfectly interchangeable. This is biologically plausible. Cortical layers are spatial/depth-defined compartments, while cell types are morphology/topology-defined categories. Near layer boundaries, these criteria can disagree. A neuron can sit near a depth boundary while having morphology more typical of the adjacent layer.

The results also suggest that the deepest populations deserve special caution. L6 has the highest mismatch rate and the L6 mismatches are not explained by simple boundary proximity. Since L6 subtypes are also relatively small in the dataset, any L6 or L6-subtype result should be interpreted with wider uncertainty.

For the main Phase 1 layer-decoding question, the important point is that the model result survives removal of all ambiguous neurons. Therefore, label ambiguity is a limitation to report, not a refutation of the result.

For Phase 3 subtype decoding, the mismatch is more important. Since Phase 3 focuses on L5 and L6 subtypes, the analysis indicates that Phase 3 should distinguish between strictly within-layer decoding and morphology-defined cell-type decoding.

## Consequences for Phase 1

The Phase 1 result should not be changed or discarded. The best formulation is:

> We observed a 4.6% mismatch between depth-derived layer labels and cell-type-implied layer labels. This mismatch was concentrated in deep layers and in a small number of session-6 scans. Excluding all mismatched neurons did not materially change Tier G performance under either GroupKFold or LOSO, indicating that the layer-decoding result is robust to this annotation ambiguity.

This is stronger than ignoring the issue. It shows that the model has been tested against a plausible ground-truth-label problem.

## Consequences for Phase 3

Phase 3 should be structured carefully because subtype decoding is most affected by this ambiguity.

The clean headline analysis should be strictly within-layer:

- L5: classify `5P-IT` vs `5P-ET` using only neurons with `layer_label == L5`; optionally use a stricter consistent-only version as the main robustness check.
- L6: classify `6P-IT` vs `6P-CT` using only neurons with `layer_label == L6`; again, report a consistent-only sensitivity version.

This answers the question: at fixed depth-defined layer, is subtype identity recoverable from function?

A second analysis can be morphology-defined:

- classify all `5P-IT` vs `5P-ET` neurons, regardless of `layer_label`;
- classify all `6P-IT` vs `6P-CT` neurons, regardless of `layer_label`.

This answers a different question: is morphology-defined cell type recoverable from function, even when the depth label is ambiguous?

Both analyses are valid, but they must not be conflated. The first is the clean within-layer subtype task. The second is a morphology-defined cell-type task and should be reported as a sensitivity/parallel analysis.

For Phase 3, a depth-only or `pt_position_y`-only baseline should be reported. If subtype classification works but `pt_position_y` alone also predicts subtype, then the model may be exploiting residual depth differences rather than purely functional subtype signatures.

## Recommended report wording

A concise version for the report could be:

> As a label-quality sensitivity analysis, we compared the depth-derived layer label used in Phase 1 with the layer implied by the AIBS excitatory cell-type label. The two labels disagreed for 407/8,895 neurons (4.6%). The mismatch was enriched in L5 and L6 and highly concentrated in four session-6 scans (`6_2`, `6_4`, `6_6`, `6_7`), which together accounted for 385/407 mismatches. Removing all mismatched neurons did not materially change Tier G HGB performance: GroupKFold balanced accuracy changed from 0.663 to 0.660, and LOSO balanced accuracy changed from 0.497 to 0.508. Thus, the main layer-decoding result is robust to this annotation ambiguity.

A fuller interpretation could be:

> The mismatch itself is informative. In L2/3, L4, and L5, mismatched neurons tend to lie farther from the median depth of their assigned layer, consistent with boundary cells near laminar transitions. In L6, however, mismatches do not show this boundary pattern, suggesting a different source of ambiguity such as cell-type annotation error or deep-layer subtype admixture. Out-of-fold predictions on mismatched neurons show that the model usually follows the depth-derived layer label, but in two morphologically extreme mismatch directions — 23P cells labelled L4 and 6P cells labelled L5 — predictions more often align with the cell-type-implied layer. This suggests that some functional fingerprints may reflect morphology-associated identity, although this remains suggestive rather than definitive because the model was trained on depth-derived labels.

## Final interpretation

This analysis is useful even though it does not improve the main model. It shows that the dataset contains structured annotation ambiguity, not just random error. It also shows that the main Phase 1 result is not an artefact of those ambiguous cases. Therefore, the sensitivity analysis should be reported as both a robustness check and a dataset-insight result.

The final scientific position is:

1. The main layer-decoding result is robust to the 4.6% layer/cell-type mismatch.
2. Most ambiguity is concentrated near boundaries or in specific session-6 scans.
3. L6 is a special case and should be treated cautiously.
4. Some morphologically extreme mismatches show functional predictions closer to morphology than depth.
5. Phase 3 should separate strict within-layer subtype decoding from morphology-defined cell-type decoding.
