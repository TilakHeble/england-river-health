# What drives the health of England's rivers?

*Executive summary. Details in the [model development document](model_development_document.md) and
[validation report](validation_report.md).*

## The question

Only **15%** of England's 3,772 river water bodies reached good ecological status in the Environment Agency's
2025 classification. Storm overflows dominate the public debate, but rivers also receive treated sewage,
runoff from farms and towns, and septic tank effluent. **Which pressures explain why rivers fail, how much do
overflows matter by comparison, and where would cutting spills most likely change a river's status?**

## How it was tested

* **Data:** five Environment Agency and Natural England open datasets joined on one river network:
  * storm overflow spills 2021-24 (about 14,000 overflows a year; national totals match the EA's published figures exactly);
  * every sewage works with its permitted flow;
  * habitat mapped at about 2 million points;
  * 650,000 chemistry samples;
  * the 2025 WFD classification.
* **River network:** the EA publishes no downstream links, so they were inferred from river centrelines.
  Upstream areas match gauged catchments (Trent 8,240 vs 8,231 km²; Eden 2,270 vs 2,287 km²).
* **Models:** a decision tree, random forests, XGBoost, LightGBM and CatBoost, against a logistic baseline.
  The champion is LightGBM, tuned with Optuna and constrained so that more pollution can never raise the
  predicted chance of good status.
* **Validation:** whole catchments held out (spatial cross-validation), nested tuning, calibration checks, and a
  test of the model's explanations against the EA investigators' own reasons for failure.

## The answer

| Element | Good today | Predictable from pressures? (AUC, nested spatial CV) | Main drivers in the model |
|---|---|---|---|
| Overall ecological status | 15% | 0.80 | urban land, natural setting, arable, sewage works |
| Invertebrates | 76% | 0.80 | urban land, natural setting, arable, overflows |
| Phosphate | 48% | 0.81 | urban land, **sewage works effluent**, natural setting, arable |
| Fish | 42% | ≈ 0.6 | barely predictable: barriers and habitat are not in the data |

**Five findings:**

1. **Urban land is the largest single pressure on every element** (25-38% of the explained variation). It
   stands for road runoff, misconnected drains, urban streams and the densest sewer networks together.
2. **For phosphate, treated sewage matters more than overflows.** Sewage works effluent per km² of catchment
   carries 23% of the explanation, storm overflows 5%. The first questions a decision tree asks are "is there
   any built-up land upstream?" and "does the catchment receive more than about 6 m³/day of treated sewage
   per km²?"
3. **Overflows are a real but secondary national pressure.** Stopping every spill is expected to lift about
   **50 more water bodies to good ecological status** (from 569 expected to 621), 60 more for invertebrates and
   about 110 more for phosphate. Halving spills gives about 12 more at good
   ecological status. Halving urban land's effect would give about 70.
4. **The gains are concentrated.** Half the predicted invertebrate gain from stopping spills sits in 200 water
   bodies (5%). The top of the list is the post-industrial rivers of West Yorkshire, Greater Manchester and South
   Yorkshire: the Calder, Aire, Irwell, Mersey and Don, each with hundreds of overflows upstream. There the
   chance of good invertebrates roughly doubles, from about 30% to 60%.
5. **The explanations agree with the EA's own investigators.** Among failing rivers, the model's blame
   predicts the EA's recorded reason for failure: urban AUC 0.84-0.88, sewage 0.75-0.76, overflows 0.64-0.72,
   arable farming 0.62-0.69. The one disagreement is livestock: improved grassland carries no signal (AUC ≈ 0.5)
   because stocking density is not in the data.

## How much to trust it

* **Accuracy:** AUC 0.80-0.82 on catchments the model never saw. Probabilities are well calibrated (the largest
  decile gap for phosphate is 0.04).
* **Trees vs regression:** a logistic regression on log-transformed pressures is almost as accurate (within
  0.01 AUC) for overall status. Trees add most for phosphate, where effluent's effect depends on dilution and
  geology. Their main value here is honest non-linear shapes, monotone guarantees and per-river explanations.
* **Random cross-validation would have overstated accuracy**, most for the flexible models, because
  neighbouring rivers share outcomes (72% of nearest neighbours agree on phosphate).
* **Scenarios are model expectations, not causal estimates.** They assume the associations learned across
  rivers would hold if one pressure changed. The monotone constraints guarantee only the direction.

## Recommendations

1. **Target overflow investment by river, not by spill count.** Use the ranked list of water bodies where
   removing spills most raises the chance of good status, alongside the spill-frequency targets.
2. **Do not treat overflows as the whole sewage problem.** For phosphate, continuous sewage works effluent
   relative to river size matters more. Phosphorus removal at works serving small rivers is the larger lever.
3. **Add the missing data before the next cycle:** livestock density, soils and rainfall, flows, barriers to
   fish, and phosphorus permit limits at each works. For phosphate, location alone still carries information
   the pressures miss (+0.03 AUC from coordinates).
4. **Validate with change over time.** As EDM records lengthen, test whether rivers whose spills fall improve
   in the next classification. That is the evidence a causal claim needs.

*Data: Environment Agency (EDM storm overflow annual returns, WFD Cycle 4 classification and catchments,
Catchment Data Explorer, Water Quality Archive, consented discharges register) and Natural England (Living England
Habitat Map Phase 4). © Environment Agency and Natural England copyright and/or database right. Open Government
Licence v3.0; consented discharges under the EA Conditional Licence.*
