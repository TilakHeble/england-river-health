# Model development document: what drives the ecological health of England's rivers?

*Tree-based classification models (decision tree, random forest, XGBoost, LightGBM, CatBoost) on Environment Agency
and Natural England open data. Figures and tables are produced by `src/river_health/experiments.py` and
`scripts/make_figures.py`; every number below is in `outputs/tables/`.*

## 1. Purpose and scope

**Question.** Only 15% of England's 3,772 river water bodies reached good ecological status in the 2025
Water Framework Directive (WFD) classification. Which pressures explain why rivers fail? In particular, how
much do storm overflow spills matter compared with treated sewage effluent, farming and urban land? And
where would reducing spills most likely change a river's status?

**Intended use.** Screening and prioritisation by a regulator, water company or catchment partnership:
* rank water bodies where a pressure is most likely to be holding status down;
* check investigators' "reasons for not achieving good" against a consistent national model;
* give scenario-level expectations for policy (for example, the storm overflow reduction targets).

**Not intended for:** setting permits for individual discharges, attributing legal responsibility, or
estimating causal effects of specific schemes. The models learn associations across rivers (section 8).

**Targets** (2025 Cycle 4 classification, binary *good or better* vs *moderate or worse*):

| Target | Water bodies | Share good | Why it matters |
|---|---|---|---|
| Overall ecological status | 3,770 | 15.0% | the headline WFD outcome; "one out, all out" across elements |
| Invertebrates | 3,582 | 75.9% | the biological element most sensitive to organic pollution |
| Phosphate | 3,706 | 48.4% | the most common reason for failing; linked to sewage and farming |
| Fish (supplementary) | 2,154 | 42.2% | sensitive to barriers and habitat, which these data lack |

The invertebrate class is also modelled as an ordered five-level outcome (bad to high).

## 2. Data

| Source | Content | Licence |
|---|---|---|
| EA, WFD River Water Body Catchments and centrelines, Cycle 4 Classification 2025 | 3,772 catchment polygons, typology, every element class | OGL v3 |
| EA, Event Duration Monitoring (EDM) storm overflow annual returns 2020-2025 | spill counts (12/24-hour counting) and durations for about 14,300 overflows a year | OGL v3 |
| EA, Consented discharges to controlled waters with conditions | active water-company final-effluent permits with dry-weather flow; private sewage, trade and fish-farm discharges | EA Conditional Licence |
| Natural England, Living England Habitat Map Phase 4 | 16 habitat classes, 6.9 million polygons | OGL v3 |
| EA, Water Quality Archive (new API, 2023-2025) | orthophosphate, ammonia, BOD, nitrate, dissolved oxygen: about 650,000 river samples | OGL v3 |
| EA, Catchment Data Explorer: Reasons for Not Achieving Good (Cycle 3) | investigators' attribution of failures to sectors and activities | OGL v3 |

### 2.1 Processing

* **Storm overflows.** Six annual workbooks with five different layouts are standardised
  (`river_health.edm`). Durations arrive as decimal hours, `hh:mm:ss` text, Excel times and Excel datetimes
  for durations over 24 hours. Each overflow is placed in a Cycle 4 catchment by its outlet grid reference,
  falling back on the reported WFD ID. Spill features average **2021-2024**, the four years before the 2025
  classification; 2025 spills are kept aside. National totals reproduce the EA's published figures exactly
  (2023: 464,056 spills, 3.61 million hours; 2025: 292,119 spills, 1,878,240 hours).
* **Sewage works.** Active permits for water-company final effluent (type SA) discharging to fresh water,
  joined to their permitted dry-weather flow (DWF, m³/day) and placed by effluent grid reference.
* **Land cover.** The habitat map is sampled on a 250 m grid (about 2 million points) and collapsed to arable,
  improved grassland, urban, woodland, semi-natural, wetland/bog, water and other. Every catchment receives
  at least 8 points (median 370), and under 1% fall outside mapped habitat. Across river catchments the
  sample is 29% arable, 17% improved grassland and 6% built-up. (London and other estuary-side towns drain
  to coastal and transitional water bodies, outside these catchments.)
* **Chemistry.** Site medians of 2023-2025 samples (at least six per site; pollution-incident samples
  excluded; below-detection results set to half the limit), then the median across sites in a water body.
  Available for 72% of water bodies. Used only as a diagnostic (section 4.1).

### 2.2 The river network

A pressure upstream affects every water body below it, but the EA publishes no downstream link.
`river_health.network` infers one:

1. **Line rule.** EA centrelines are digitised in the direction of flow, so the last vertex of each
   line part is a potential outlet. If it touches another water body's line within 25 m, at a point that is
   not also one of that line's ends, that water body is downstream (3,171 links).
2. **Polygon rule.** Where simplified geometry leaves a gap, the nearest neighbouring river catchment within
   250 m of an outlet (71 links). Otherwise the water body drains to the sea or an estuary (509).
3. **Cycles** caused by reversed digitisation are broken by removing one edge per cycle (21).

Upstream areas match National River Flow Archive gauged areas: Trent 8,240 vs 8,231 km², Eden 2,270 vs
2,287, Tyne 2,207 vs 2,176, Nene 1,576 vs 1,634, Thames 9,001 vs 9,948. Rivers rising in Wales lack their
Welsh headwaters, which are outside the EA data.

## 3. Features

Every pressure is measured inside the water body's own catchment (`local_`) and summed over everything
upstream (`up_`). Intensities divide by upstream area, a proxy for dilution, since flow data are not
available nationally.

| Group | Features |
|---|---|
| Storm overflows | spill hours and counts per km² upstream; local spill hours, counts and overflows; overflows averaging ≥ 60 spills a year (local and upstream); spill hours at sewage works storm outlets |
| Sewage works | DWF per km² upstream; local DWF; number of local works |
| Private sewage | septic tank and package plant permits per km² upstream; local count |
| Land cover | arable, improved grassland and urban shares (local and upstream) |
| Trade and fish farms | upstream counts of trade effluent and fish-farm permits |
| Natural setting | local and upstream area; links to the sea; number of upstream water bodies; drains directly to sea; upstream semi-natural, wetland, woodland and water shares; WFD typology (altitude, size, geology); heavily modified or artificial designation |
| *Diagnostic only* | measured orthophosphate, ammonia, BOD, nitrate, dissolved oxygen; easting and northing |

Excluded on purpose: the management catchment (it defines the CV groups), and the reasons-for-failure flags
(they are the validation reference).

## 4. Model design

### 4.1 Pressures, not chemistry

Measured chemistry predicts status better than any pressure (AUC for phosphate rises from 0.82 to 0.90), but
partly by construction: the phosphate class *is* a concentration compared with a type-specific standard.
Chemistry is also the route by which pressures act, so a model given it explains *that* a river has too
much phosphate, not *why*. The champion models therefore use **pressures and natural setting only**. Chemistry
and coordinates appear only in a diagnostic comparison (section 6.3).

### 4.2 Spatial cross-validation

Neighbouring rivers share pressures and outcomes: 72% of a water body's five nearest neighbours have the same
phosphate outcome, against 50% expected by chance. All scores use **5-fold cross-validation with whole
management catchments held out** (90 catchments; `StratifiedGroupKFold`). Every preprocessing step (imputation,
encoding, log transforms, scaling) is fitted inside the training folds. Random-fold results are reported only to
show the optimism they introduce.

### 4.3 Models compared

| Model | Settings |
|---|---|
| Logistic regression (baseline) | signed log1p transform, standardised, one-hot typology, L2 (C = 0.3) |
| Decision tree | Gini; cost-complexity pruning path and max depth, both chosen by spatial CV |
| Random forest | 500 trees, `max_features = sqrt`, min leaf 3 |
| Extra trees | 500 trees, random thresholds |
| Histogram gradient boosting (scikit-learn) | native categoricals; with and without monotone constraints |
| XGBoost | depth 4, 400 trees, learning rate 0.03; with and without monotone constraints |
| LightGBM | 15 leaves, 400 trees, learning rate 0.03; with and without monotone constraints |
| CatBoost | depth 5, 600 iterations, ordered boosting, native categoricals, monotone |
| **Champion: LightGBM, tuned, monotone** | Optuna TPE search (log loss) inside nested spatial CV |

**Monotone constraints.** Every overflow, sewage works, private sewage, trade-effluent and urban-land feature is
constrained so that more of it can never raise the predicted probability of good status. Arable and grassland
shares are left free, because their relationship with status is confounded with geology and region (section 8).

### 4.4 Tuning

Nested: for each outer fold, 25 Optuna trials scored by inner 4-fold spatial CV on that fold's training
catchments. Search space: trees 100-800, learning rate 0.01-0.1, leaves 4-48, minimum leaf 10-120, row
subsampling 0.5-1, column subsampling 0.3-1, L2 0.001-30. Final hyperparameters come from a 50-trial search
on all data (`outputs/tables/best_params.json`). Every search settled on slow learning (about 0.01) and
heavy row subsampling (0.5-0.66).

### 4.5 Ordinal invertebrate class

Two approaches: a five-class softmax (LightGBM multiclass), and the Frank and Hall (2001) decomposition. The latter
fits four monotone binary models for P(class > k), forces the cumulative probabilities to be non-increasing,
and differences them into class probabilities.

### 4.6 Explanation and validation of explanations

SHAP values from `TreeExplainer` (exact for trees), in log-odds. Because correlated features share credit
arbitrarily, contributions are summed into eight pressure groups: storm overflows, sewage works, private sewage,
arable, grassland, urban land, trade and fish farms, and natural setting. Explanations are then tested against
an independent reference: the Environment Agency's own **Reasons for Not Achieving Good** (Cycle 3), which
investigators assign to sectors and activities and which the model never saw.

## 5. Evaluation metrics

* **ROC AUC**: ranking of water bodies by risk; the headline metric.
* **Log loss and Brier score**: quality of the probabilities (used for tuning).
* **PR AUC**: reported because good overall status is rare (15%).
* **Calibration**: observed vs predicted share good by decile of out-of-fold prediction.
* **Ordinal**: quadratic weighted kappa (QWK), which penalises a bad-for-high error more than a good-for-high error;
  mean absolute error of the expected class.

## 6. Results

### 6.1 Accuracy

ROC AUC, out of fold, spatial 5-fold CV, pressure features (`outputs/tables/benchmark.csv`):

| Model | Overall status | Invertebrates | Phosphate | Fish |
|---|---|---|---|---|
| Decision tree (min leaf 20, α = 0.002) | 0.752 | 0.745 | 0.773 | 0.561 |
| Logistic (log pressures) | **0.805** | 0.795 | 0.806 | **0.626** |
| Random forest | 0.794 | **0.800** | 0.818 | 0.617 |
| Extra trees | 0.799 | 0.790 | 0.801 | 0.611 |
| Histogram GB (scikit-learn) | 0.786 | 0.790 | 0.816 | 0.591 |
| XGBoost | 0.794 | 0.798 | **0.822** | 0.599 |
| XGBoost, monotone | 0.796 | 0.796 | 0.817 | 0.611 |
| LightGBM | 0.793 | 0.798 | 0.821 | 0.594 |
| LightGBM, monotone | 0.792 | 0.795 | 0.817 | 0.605 |
| CatBoost, monotone | 0.790 | 0.794 | 0.818 | 0.610 |
| **LightGBM, tuned, monotone (champion)** | 0.803 | 0.804 | 0.817 | n/a |
| Champion, nested tuning | 0.801 | 0.798 | 0.812 | n/a |

* **Ensembles beat a single tree by 0.04-0.05.** The best spatially validated single tree has only 10-16 leaves
  (`tree_pruning.csv`). Unpruned, a tree's held-out AUC falls towards 0.65.
* **A good logistic model is nearly as accurate.** Within 0.01 for overall status and invertebrates. The
  relationships are close to monotone and additive on the log scale. Boosting gains most for **phosphate**
  (+0.015), where the effect of effluent depends on dilution, geology and land use together.
* **Monotone constraints cost 0.001-0.004 AUC.**
* **Tuning gains little** (+0.01 for overall status, otherwise within noise). The searches preferred slow
  learning and heavy subsampling.
* **Fish status is not predictable** from these pressures. Its drivers (weirs and barriers, habitat, flow,
  stocking) are not in the data.

### 6.2 Optimism of random folds and out-of-bag error

| | Overall status | Invertebrates | Phosphate |
|---|---|---|---|
| LightGBM: random CV − spatial CV | +0.027 | +0.006 | +0.025 |
| Logistic: random CV − spatial CV | +0.011 | +0.004 | +0.010 |
| Random forest (400 trees): OOB − spatial CV | +0.023 | +0.008 | +0.026 |

Flexible models gain most from seeing neighbouring rivers. Out-of-bag error, often presented as a free test set,
is as optimistic as random CV here.

### 6.3 What the pressures miss

LightGBM, spatial CV:

| Feature set | Overall status | Invertebrates | Phosphate | Fish |
|---|---|---|---|---|
| pressures | 0.793 | 0.798 | 0.821 | 0.594 |
| + coordinates | 0.805 | 0.799 | 0.858 | 0.616 |
| + measured chemistry | 0.824 | 0.821 | 0.900 | 0.612 |

Chemistry's large gain for phosphate is partly circular, since the phosphate class is a concentration standard.
The coordinates' gain for phosphate (+0.037) points to regional drivers absent from the features: soils,
rainfall, livestock and works-level phosphorus limits.

### 6.4 Calibration and the ordinal model

Out-of-fold calibration is good for all champions (largest decile gap 0.04 for phosphate; `calibration.png`).
For the five-level invertebrate class, the monotone Frank and Hall model reaches QWK 0.439 (multiclass 0.414; class
frequencies 0). Its most likely class is right 48% of the time, against 43% for always predicting the commonest class.

### 6.5 Drivers

Share of mean |SHAP| by pressure group (champion models):

| Group | Overall status | Invertebrates | Phosphate |
|---|---|---|---|
| Urban land | **29%** | **38%** | **25%** |
| Natural setting | 19% | 32% | 22% |
| Arable | 17% | 10% | 16% |
| Sewage works | 16% | 4% | **23%** |
| Storm overflows | 10% | 7% | 5% |
| Grassland | 5% | 4% | 5% |
| Private sewage | 5% | 5% | 4% |
| Trade and fish farms | 0% | 0% | 1% |

Effects are thresholds rather than straight lines (`shap_dependence.png`):
* sewage works effluent harms phosphate status from about 5-10 m³/day per km² upstream, and harms invertebrates
  only above about 60;
* overflows start to harm invertebrates above about 30 spill hours per km² a year;
* the first few per cent of built-up land do the most damage.

Agreement with the EA's reasons for failure is in the validation report (section 6).

### 6.6 Scenarios

The champions are re-scored with spill features scaled down (frequent-spiller counts fall with them), everything
else fixed. Expected number of water bodies at good status:

| Scenario | Overall status | Invertebrates | Phosphate |
|---|---|---|---|
| Today (sum of predicted probabilities) | 568.5 | 2,856.5 | 1,822.2 |
| Spills -25% | +4.6 | +14.2 | +8.5 |
| Spills -50% | +12.0 | +30.9 | +22.7 |
| Spills -75% | +20.9 | +44.4 | +42.2 |
| **No spills** | **+52.5** | **+59.9** | **+107.8** |
| *Yardstick: sewage works effluent -50%* | +11.8 | +9.7 | +63.7 |
| *Yardstick: urban land -50%* | +72.3 | +173.0 | +186.7 |

Gains accelerate as spills approach zero, because the frequent-spiller features step down only at large cuts
and many rivers sit just past the thresholds. Gains are concentrated: half the invertebrate gain falls in 200
water bodies (5%), led by the Calder, Aire, Irwell, Mersey and Don (`notebooks/07_spill_scenarios.ipynb`).

## 7. Model choice

The champion is **LightGBM with monotone constraints, tuned by nested Optuna search**. It is no more accurate
than a random forest or a well-specified logistic regression on these data, and this document does not claim it is.
It is chosen because it combines:
* calibrated probabilities;
* guaranteed physically coherent responses (more pressure never helps), which scenarios require;
* exact per-river SHAP explanations;
* non-linear threshold effects that a logistic model would need hand-crafted terms to capture.

The logistic model is kept as a challenger, and its near-equal accuracy is itself a finding.

## 8. Limitations

* **Associations, not causes.** Pressures are correlated (urban land, sewage works and overflows cluster
  together), and scenario sizes assume the cross-sectional associations transfer to change within a river.
  Grouped SHAP limits, but does not remove, credit-sharing between correlated pressures.
* **Arable and grassland are left unconstrained** and show non-monotone effects (for example, phosphate's
  arable curve rises again around 50% arable). This reflects geology and region (chalk catchments in the east),
  not a benefit of farming.
* **Missing drivers:** livestock density, soils, rainfall, flow, barriers, works-level phosphorus permits.
* **Dilution is approximated by upstream area**, not measured flow.
* **England only.** Welsh headwaters and coastal catchments are excluded.
* **EDM coverage.** 9-19% of overflows had under 90% monitoring coverage in 2021-2024, so spills are understated.
* **One classification year.** Status changes between cycles are not modelled.

## 9. References

* Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32.
* Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD*.
* Frank, E. and Hall, M. (2001). A simple approach to ordinal classification. *ECML*.
* Friedman, J. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5).
* Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*.
* Lundberg, S. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2.
* Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
* Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40.
* Environment Agency (2026). Event Duration Monitoring: storm overflows annual returns; WFD Cycle 4 classification 2025.
