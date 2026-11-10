# Validation report: England river health models

*Independent review of data, design, implementation and performance. Checks are reproducible with
`python scripts/validate.py` (results in `outputs/tables/validation_checks.csv`) and `pytest`.*

## Summary

| Area | Rating | Comment |
|---|---|---|
| Data integrity | 🟢 Green | Storm overflow totals reproduce EA published figures exactly for 2023-2025; chemistry linkage agrees with EA phosphate classes (Spearman -0.79) |
| River network | 🟢 Green | Acyclic; upstream areas within 0-10% of NRFA gauged areas for five English rivers; Welsh headwaters absent (documented) |
| Evaluation design | 🟢 Green | Catchment-blocked CV, nested tuning, preprocessing inside folds; flipping test labels changes no prediction |
| Discrimination | 🟢 Green | Nested spatial-CV AUC 0.80 (overall status), 0.80 (invertebrates), 0.81 (phosphate) |
| Calibration | 🟢 Green | Largest decile gap 0.04 for phosphate; well calibrated for all three targets |
| Explanations | 🟢 Green | SHAP blame agrees with the EA's independent reasons for failure (AUC 0.64-0.88); livestock is the exception |
| Causal use (scenarios) | 🟡 Amber | Direction guaranteed by monotone constraints; magnitudes rest on cross-sectional associations |
| Fish model | 🔴 Red | AUC about 0.6; not fit for use; missing drivers (barriers, habitat, flow) |

## 1. Data

| Check | Result |
|---|---|
| EDM national totals vs EA publications | 2023: 464,056 spills, 3,606,170 h; 2024: 450,398, 3,614,428 h; 2025: 292,119, 1,878,240 h (all exact) |
| Overflows linked to a river catchment | 11,952 of 14,285 (84%); the rest discharge to coastal/estuarine catchments or Wales |
| Chemistry vs EA phosphate class | Spearman ρ = -0.79 over 2,729 water bodies |
| Land-cover sample coverage | every catchment ≥ 9 points (median 371); 0.5% of points unmapped |
| Uniqueness | one row per Cycle 4 river water body (3,772) |

**Issues found and fixed during development:**
* Excel stores spill durations over 24 hours as datetimes counted from 1899-12-31, and some company sheets
  store monitoring coverage as a fraction while others use percentages. Both are handled per sheet; the duration formats have unit tests.
* The first cycle-breaking routine removed every edge in a cycle, not one per cycle. Caught by a unit test; the
  fix restored, for example, the Eden's upstream area from 1,861 to 2,270 km² (gauge: 2,287).
* Choosing a water body's outlet from its longest line part picked tributaries for multi-part water bodies,
  fragmenting the Thames. Replaced by voting over every part's end vertex.

## 2. River network

| Gauge (NRFA) | Gauged area | Inferred upstream area | Ratio |
|---|---|---|---|
| Trent at North Muskham | 8,231 | 8,240 | 1.00 |
| Tyne at Bywell | 2,176 | 2,207 | 1.01 |
| Eden at Sheepmount | 2,287 | 2,270 | 0.99 |
| Nene at Orton | 1,634 | 1,576 | 0.96 |
| Thames at Kingston | 9,948 | 9,001 | 0.90 |

The Thames shortfall comes from a few tributary links lost in the simplified geometry. Rivers with Welsh
headwaters (Severn, Wye, Dee) are under-counted by design. Upstream pressure totals for their lower English
reaches are therefore too low, which biases those reaches towards looking cleaner than they are.

## 3. Leakage and evaluation design

* **Spatial folds.** No management catchment appears in both train and test (checked on the real data).
  Random folds overstate AUC by up to about 0.025 for boosting on phosphate, against about 0.01 for the
  logistic baseline (benchmark table, `cv = random`).
* **Out-of-bag error is optimistic for the same reason.** For phosphate, random forest OOB AUC is 0.845, against
  0.819 under spatial CV (`forest_study.csv`).
* **Labels never touch test folds.** Flipping every fold-0 test label leaves fold-0 predictions unchanged (max
  change 0.0).
* **Tuning is nested.** The tuned AUCs (0.801, 0.798, 0.812) come from outer folds whose catchments were never
  used in that fold's Optuna search.
* **Time ordering.** Spill features use 2021-2024; the 2025 classification is the target.
* **Target leakage avoided by design.** Chemistry, which partly defines the phosphate class, is excluded from the
  champion models (section 4.1 of the model development document).

## 4. Implementation

* 14 unit tests (`pytest`) on synthetic data cover:
  * grid-reference conversion (checked against EA-published eastings and northings);
  * duration parsing in all four formats;
  * network inference on a toy river;
  * upstream sets and cycle breaking;
  * spatial folds;
  * recovery of a known signal by three model types;
  * monotonicity of the constrained LightGBM;
  * the spill-reduction scenario.
* Monotonicity verified on the real champions: doubling any overflow, sewage-works or urban feature for all
  3,772 water bodies never raised a predicted probability, and removing spills never lowered one (0 violations).
* The learn notebooks re-derive each algorithm from scratch and match the libraries:
  * our tree equals scikit-learn's exactly;
  * our gradient boosting matches scikit-learn's predictions (correlation 1.0, identical AUC);
  * our second-order split gain and leaf weights match XGBoost's dump;
  * brute-force Shapley values equal TreeSHAP's to four decimals.

## 5. Performance

Out-of-fold, spatial 5-fold CV, pressure features only (`champions.csv`, `nested_tuning.csv`):

| Target | Logistic | Random forest | LightGBM tuned, monotone | Nested tuned AUC | Brier (LightGBM) |
|---|---|---|---|---|---|
| Overall status good | 0.805 | 0.794 | 0.803 | 0.801 | 0.106 |
| Invertebrates good | 0.795 | 0.800 | 0.804 | 0.798 | 0.136 |
| Phosphate good | 0.806 | 0.819 | 0.817 | 0.812 | 0.172 |

**Ordinal invertebrate class** (`ordinal_inv_class.csv`): QWK 0.439 for the monotone Frank and Hall model, 0.414
for the multiclass softmax, 0 for class frequencies alone. The model separates bad and poor rivers from high
ones, but rarely predicts the extreme classes with confidence.

**Assessment.** The ensembles' advantage over a well-specified logistic model is small, 0.00-0.01 AUC
(largest for phosphate), and should not be overstated. The champion is preferred for its calibrated
probabilities, its monotone guarantees and its exact per-river explanations, not for accuracy alone.

## 6. Explanations

Test: among water bodies failing a target, does the model's SHAP blame for a pressure group rank them the way
the EA's investigators' **Reasons for Not Achieving Good** flags do? These flags were never used in training.

| Pressure group vs EA flag | Overall status | Invertebrates | Phosphate |
|---|---|---|---|
| Urban land vs "urban and transport" | 0.86 | 0.88 | 0.84 |
| Sewage works vs "waste water treatment / sewage" | 0.76 | 0.75 | 0.76 |
| Storm overflows vs "storm overflow / intermittent discharge" | 0.71 | 0.72 | 0.64 |
| Arable vs "agriculture and rural land management" | 0.69 | 0.62 | 0.66 |
| Improved grassland vs "agriculture" | 0.51 | 0.48 | 0.57 |

Agreement is strong for urban and sewage pressures and moderate for overflows and arable land. **Livestock
pressures are not captured:** improved grassland is a poor proxy for stocking density, so the model cannot
see the agricultural failures investigators attribute to livestock. RNAG records are themselves judgements of
varying certainty, so perfect agreement is not expected.

## 7. Limitations and recommendations

1. **Cross-sectional associations.** Scenario magnitudes assume that what distinguishes rivers today is what
   would change a river tomorrow. Validate against status changes as EDM records lengthen. 🟡
2. **Missing drivers.** Livestock density, soils, rainfall, river flow (true dilution), barriers and phosphorus
   permit limits. Coordinates add 0.03 AUC for phosphate, evidence of unexplained regional structure. 🟡
3. **Welsh headwaters and coastal catchments are excluded**, so cross-border rivers' upstream pressures are
   understated. 🟡
4. **EDM data quality.** Between 9% (2024) and 19% (2021) of overflows report under 90% monitoring
   coverage, so their spill totals are lower bounds. This understates spills most in the earlier years. 🟡
5. **Fish.** Do not use the fish model. 🔴
6. **Monitoring is not random.** Rivers without chemistry are more often good (60% vs 44%), and classification
   certainty varies (`eco_certainty`). Both are recorded but not modelled. 🟡

**Recommendation:** fit for national screening and prioritisation of where pressures are most likely to be
holding status down, used alongside local investigation. Not fit for permit setting or attributing liability.
