"""Build and execute the analysis notebooks in notebooks/.

    python scripts/build_notebooks.py            # all
    python scripts/build_notebooks.py 03 05      # selected
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import code, md, write_and_run  # noqa: E402

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = code("""
import sys, warnings, json
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.figsize": (9, 4.2), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
T = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
from river_health import modeling as M
t = M.prepare(pd.read_parquet(ROOT / "data" / "processed" / "river_features.parquet"))
""")

# ----------------------------------------------------------------------------- 01
N01 = [
    md("""
# 01. Data: five public sources joined on one river network

**Question for the project:** what drives the ecological health of England's rivers? How much is
storm overflow spills, how much sewage works, farming and towns? And where would cutting spills do most good?

The unit of analysis is the **WFD river water body**: 3,772 stretches of river in England, each with its
own land catchment and its own 2025 classification (Cycle 4).

| Source (all Environment Agency or Natural England) | Used for |
|---|---|
| WFD Cycle 4 2025 classification, catchment polygons and centrelines | targets, river typology, the river network |
| Event Duration Monitoring storm overflow annual returns 2020-2025 | spill counts and hours for ~14,000 overflows |
| Consented discharges to controlled waters (permit register) | sewage works and their dry-weather flow; private sewage; trade effluent |
| Living England Habitat Map Phase 4 | land cover of every catchment |
| Water Quality Archive (new API, 2023-2025) | measured phosphate, ammonia, BOD, nitrate, oxygen |
| Reasons for Not Achieving Good status (Cycle 3) | **validation only**: the EA's own investigation of why each water body fails |
"""),
    SETUP,
    md("## Storm overflow returns: six workbooks, five layouts, one tidy table"),
    code("""
edm = pd.read_parquet(ROOT / "data/processed/edm_2020_2025.parquet")
summary = edm.groupby("year").agg(overflows=("company", "size"), spills=("spills", "sum"),
                                  spill_hours=("duration_hrs", "sum"),
                                  median_monitoring_pct=("edm_pct", "median"))
summary.round(0)
"""),
    md("""
Each year's workbook differs: the header row moves, durations switch from decimal hours (2020-23) to
`hh:mm:ss` text and Excel datetimes (2024-25), water body IDs move from Cycle 2 to Cycle 3, and 2020 has no
grid references. `river_health.edm` standardises them. The 2023-2025 totals reproduce the Environment
Agency's published national figures to the hour (see `scripts/validate.py`).
"""),
    code("""
v = pd.read_csv(T / "validation_checks.csv")
v[v.check.str.startswith("EDM")]
"""),
    md("""
## Building the river network

Pressures upstream travel downstream, so a water body's health depends on everything above it. The EA
publishes centrelines and catchments but **no downstream link**. We infer one from geometry: EA centrelines
are digitised in the direction of flow, so each line part's last vertex is a potential outlet; where it
touches another water body's line (not at that line's own end), that water body is downstream.
Gaps in the simplified geometry fall back to the nearest neighbouring catchment polygon.
"""),
    code("""
e = pd.read_csv(ROOT / "data/processed/network_edges.csv")
print(e.rule.value_counts().to_string())
v[v.check.str.contains("network")][["check", "passed", "detail"]].style.hide(axis="index")
"""),
    md("""
Upstream areas reproduce the National River Flow Archive's gauged areas: Trent 8,240 vs 8,231 km²,
Eden 2,270 vs 2,287 km², Tyne 2,207 vs 2,176 km². The Thames reaches 9,001 of 9,948 km² because a few
links are lost in the simplified geometry. Rivers rising in Wales (Severn, Wye, Dee) are missing their
Welsh headwaters, which are outside the EA data.
"""),
    code("""
t.nlargest(8, "up_area_km2")[["name", "up_area_km2", "n_upstream_wbs", "up_n_overflows", "up_n_stw"]].round(0)
"""),
    md("## The targets and the pressures on a map"),
    code("""display(Image(FIG / "map_status.png"))"""),
    md("""
Only **15%** of river water bodies reach good ecological status. WFD status is *one out, all out*:
the worst element sets the overall class. Invertebrates mostly pass (76% good or high);
phosphate fails in half the water bodies, concentrated in the Midlands and the south-east.
"""),
    code("""display(Image(FIG / "map_pressures.png"))"""),
    code("""
cov = pd.DataFrame({
    "linked overflows (2024)": [t.local_n_overflows.sum()],
    "water bodies with an overflow upstream": [(t.up_n_overflows > 0).mean()],
    "water bodies with a sewage works upstream": [(t.up_n_stw > 0).mean()],
    "water bodies with chemistry (>= 6 samples)": [t.chem_orthophosphate_mgP.notna().mean()],
}).T.rename(columns={0: "value"})
cov
"""),
]

# ----------------------------------------------------------------------------- 02
N02 = [
    md("""
# 02. Pressures and status: what the raw data already says, and why it cannot be read directly

Before any model: how does status vary with each pressure, and which pressures travel together?
"""),
    SETUP,
    code("""
for tgt, lab in [("y_eco", "ecological"), ("y_inv", "invertebrates"), ("y_phos", "phosphate"), ("y_fish", "fish")]:
    print(f"{lab:14s}", t[tgt].value_counts(normalize=True).reindex(["High", "Good", "Moderate", "Poor", "Bad"]).round(3).to_dict())
"""),
    md("## Status by fifth of upstream spill intensity"),
    code("""
q = pd.qcut(t.up_spill_hours_per_km2.rank(method="first"), 5, labels=["lowest", "2", "3", "4", "highest"])
tab = pd.DataFrame({
    "spill h/km2 (median)": t.groupby(q).up_spill_hours_per_km2.median(),
    "eco good": t.groupby(q).y_eco_good.mean(), "invertebrates good": t.groupby(q).y_inv_good.mean(),
    "phosphate good": t.groupby(q).phos_good.mean(), "urban share": t.groupby(q).up_lc_urban.mean(),
    "sewage works DWF/km2": t.groupby(q).up_stw_dwf_m3d_per_km2.median()})
tab.round(3)
"""),
    md("""
Status worsens as spills rise, but so does everything else: the catchments with the most spills are
also the most urban and receive the most treated sewage. Read alone, this table would credit overflows
with the effect of towns and sewage works. That confounding is the central difficulty of the project.
"""),
    code("""
cols = ["up_spill_hours_per_km2", "up_stw_dwf_m3d_per_km2", "up_lc_urban", "up_lc_arable",
        "up_lc_improved_grass", "up_lc_seminatural", "up_n_private_sewage_per_km2", "up_area_km2"]
c = t[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=70, fontsize=8)
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8); ax.grid(False)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=0.7); plt.title("Spearman correlation between pressures"); plt.show()
"""),
    md("## Rank correlations with each target"),
    code("""
num = [f for f in M.PRESSURE_SET if f not in M.CATEGORICAL] + M.CHEMISTRY
r = t[num + ["y_eco_code", "y_inv_code", "y_phos_code", "y_fish_code"]].corr(method="spearman").loc[
    num, ["y_eco_code", "y_inv_code", "y_phos_code", "y_fish_code"]]
r.sort_values("y_phos_code").round(2)
"""),
    md("""
* **Phosphate** follows sewage works effluent (per km² upstream), urban land and arable land.
* **Invertebrates** follow urban land and measured ammonia and BOD: classic organic pollution.
* **Fish** correlate weakly with everything here. Barriers, flow and habitat are not in the data.
* Measured **chemistry** correlates more strongly than any pressure, as it should: it is the route by which
  pressures reach the biology. The models therefore use pressures only, and keep chemistry as a diagnostic
  (notebook 05), so that their explanations are about *what to manage*.
"""),
    code("""
pd.crosstab(t.hmwb, t.y_eco, normalize="index").reindex(columns=["High", "Good", "Moderate", "Poor", "Bad"]).round(3)
"""),
]

# ----------------------------------------------------------------------------- 03
N03 = [
    md("""
# 03. One decision tree

A single classification tree splits the data on one feature at a time, choosing the split that most
reduces impurity (Gini). It is the most readable model there is, and the building block of every model
that follows. Here: **is the phosphate class good or better?**
"""),
    SETUP,
    code("""
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
feats = M.FEATURE_SETS["pressures"]
X, y, g, d = M.xy(t, "phos_good", feats)
pre = M._onehot_pre(feats).fit(X)
Xp = pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out())
small = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=0).fit(Xp, y)
fig, ax = plt.subplots(figsize=(17, 7))
plot_tree(small, feature_names=Xp.columns, class_names=["fails", "good"], filled=True, rounded=True,
          impurity=False, proportion=True, fontsize=8, ax=ax)
plt.show()
"""),
    md("""
The first split asks whether there is **any built-up land upstream** (more than about 1% of the catchment).
Rivers without it are then split by arable land. Rivers with it are split by **sewage works dry-weather flow
per km² upstream**: how much treated sewage the river receives relative to the catchment that dilutes it.
Above about 6 m³/day per km², most rivers fail. That is the textbook phosphorus story (sewage first, then
farming), rediscovered from data.
"""),
    md("## How big should the tree be? Cost-complexity pruning, judged on unseen catchments"),
    code("""
display(Image(FIG / "tree_pruning.png"))
p = pd.read_csv(T / "tree_pruning.csv")
best = p[p.ccp_alpha.notna()].sort_values("roc_auc", ascending=False).groupby("target").head(1)
best[["target", "ccp_alpha", "leaves", "depth", "roc_auc", "log_loss"]].round(4)
"""),
    md("""
A fully grown tree memorises the training catchments, and its AUC on unseen catchments collapses.
Pruning trades leaves against fit: the cost-complexity criterion minimises
$R(T) + \\alpha |T|$, where $|T|$ is the number of leaves. The best spatially validated trees keep only
10-16 leaves (AUC 0.75-0.77) and still trail the ensembles in notebooks 04 and 05 by 0.05 AUC.
"""),
    md("## The weakness ensembles fix: instability"),
    code("""
rng = np.random.default_rng(0)
shapes, thresholds = [], []
for b in range(30):
    i = rng.choice(len(y), len(y), replace=True)
    m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=b).fit(Xp.iloc[i], y[i])
    tr = m.tree_
    used = sorted({Xp.columns[f] for f in tr.feature if f >= 0})
    shapes.append(" | ".join(used))
    thresholds.append(tr.threshold[0])
print("root threshold on", Xp.columns[m.tree_.feature[0]], "across 30 resamples:",
      np.round(np.quantile(thresholds, [0.05, 0.5, 0.95]), 4))
print("distinct sets of split features in 30 depth-4 trees:", len(set(shapes)))
pd.Series(shapes).value_counts().head(5)
"""),
    md("""
The root split is stable, but the rest of the tree is not: almost every resample uses a different set of
features below the root. Correlated pressures (urban land, sewage flow, overflows) substitute for each other.
Averaging many trees (random forest) or adding small corrective trees (boosting) removes most of that variance.
"""),
    code("""print(export_text(small, feature_names=list(Xp.columns), decimals=2))"""),
]

# ----------------------------------------------------------------------------- 04
N04 = [
    md("""
# 04. Random forests: bagging, out-of-bag error, and two kinds of importance

A random forest grows hundreds of deep trees, each on a bootstrap sample of water bodies and choosing
each split from a random subset of features (`max_features`), then averages their votes.
"""),
    SETUP,
    md("## Out-of-bag error looks reassuring and is wrong here"),
    code("""
display(Image(FIG / "forest_oob_vs_spatial.png"))
f = pd.read_csv(T / "forest_study.csv")
f[f.max_features.isna() & (f.n_trees == 400)][["target", "oob_auc", "random_cv_auc", "spatial_cv_auc"]].round(3)
"""),
    md("""
Each tree's out-of-bag rows are a free test set, but they are **neighbours** of the training rows:
a water body left out of one bootstrap sample usually has its upstream and downstream neighbours in it.
OOB AUC therefore tracks random-fold CV, and both overstate how well the forest predicts a catchment
it has never seen. Every reported score in this project uses spatial folds (whole management catchments held out).
"""),
    code("""
f[f.max_features.notna()][["target", "max_features", "spatial_cv_auc"]].round(3)
"""),
    md("""
`max_features="sqrt"` (about 6 of 32 features per split) beats using every feature: decorrelating
the trees matters more than making each tree strong.
"""),
    md("## Impurity importance vs permutation importance"),
    code("""
display(Image(FIG / "forest_importance_phos.png"))
imp = pd.read_csv(T / "forest_importance.csv")
w = imp[imp.target == "phos_good"].assign(
    impurity_rank=lambda d: d.impurity.rank(ascending=False),
    permutation_rank=lambda d: d.permutation_auc_drop.rank(ascending=False))
w.sort_values("permutation_rank").head(12).round(4)
"""),
    md("""
Impurity (MDI) importance rewards continuous features with many possible split points and is computed on
the training data. Permutation importance asks how much the held-out AUC falls when one feature's values
are shuffled. With correlated features it under-states each one (the others stand in), which is why
the project groups features by pressure in the SHAP analysis (notebook 06).
"""),
]

# ----------------------------------------------------------------------------- 05
N05 = [
    md("""
# 05. Gradient boosting, tuning, monotonic constraints and validation

Boosting fits trees in sequence, each one to the gradient of the loss left by the ensemble so far,
shrunk by a learning rate. XGBoost, LightGBM and CatBoost differ in how they grow trees and handle
categorical features; all are compared here against the forest, a single tree and a logistic baseline,
under the same spatial folds.
"""),
    SETUP,
    code("""display(Image(FIG / "benchmark_auc.png"))"""),
    code("""
b = pd.read_csv(T / "benchmark.csv")
sp = b[(b.features == "pressures") & (b.cv == "spatial")]
sp.assign(model=sp.model + np.where(sp.monotone, " (monotone)", "")).pivot(
    index="model", columns="target", values="roc_auc").round(3).sort_values("phos_good", ascending=False)
"""),
    md("""
* The ensembles beat one tree by 0.04-0.05 AUC on every target.
* **A logistic regression on log-transformed pressures is nearly as good** for overall status and for
  invertebrates. These relationships are close to monotone and additive on the log scale, so trees add
  little. Boosting earns its keep on **phosphate**, where dilution makes the effect of effluent depend on
  catchment size and geology.
* Fish status is barely predictable (AUC about 0.6) from these pressures.
"""),
    md("## Random folds flatter flexible models most"),
    code("""display(Image(FIG / "cv_optimism.png"))"""),
    md("""
Neighbouring water bodies share geology, land use and often the same sewage works. With random folds a
flexible model can recognise a place rather than learn a relationship; the gap between random and
spatial CV is largest for the most flexible models.
"""),
    md("## Nested tuning with Optuna"),
    code("""
n = pd.read_csv(T / "nested_tuning.csv")
print(n[n.fold == "all"][["target", "test_auc", "log_loss"]].round(3).to_string(index=False))
n[n.fold != "all"].groupby("target")[[c for c in n.columns if c.startswith("p_")]].median().round(3)
"""),
    md("""
Each outer fold is tuned on its own training catchments (inner 4-fold spatial CV, 30 Optuna trials,
minimising log loss), then scored on its held-out catchments, so the tuned score is honest. The per-fold
searches mostly choose **few leaves (median 6-10), large minimum leaves and strong row subsampling**. The final
all-data searches allow more leaves for overall status and phosphate (42-46), but pair them with a learning rate
of about 0.01 and subsampling. That is heavy regularisation either way, as expected with 3,700 rows. Nested
tuned AUCs (0.80, 0.80, 0.81) match the untuned defaults: tuning gains are small relative to the choice of
features and folds.
"""),
    code("""
json.loads((T / "best_params.json").read_text())
"""),
    md("## Monotonic constraints"),
    code("""
mono = b[(b.features == "pressures") & (b.cv == "spatial") & b.model.isin(["lightgbm", "xgboost"])]
mono.pivot_table(index=["model", "monotone"], columns="target", values="roc_auc").round(3)
"""),
    md("""
Constraining the model so that more spills, more sewage effluent or more urban land can **never raise**
the predicted chance of good status costs about 0.003 AUC. In exchange every scenario in notebook 07
is guaranteed to move in the physically sensible direction, and odd wiggles in the fitted curves
(noise fitted as signal) disappear.
"""),
    md("## Calibration"),
    code("""
display(Image(FIG / "calibration.png"))
pd.read_csv(T / "champions.csv").round(3)
"""),
    md("## Chemistry: what is missing from the pressures"),
    code("""display(Image(FIG / "feature_sets.png"))"""),
    md("""
Adding measured chemistry lifts phosphate AUC from about 0.82 to 0.90, partly by construction (the
phosphate class *is* a concentration threshold). Adding only the coordinates lifts it by about 0.03:
there is spatial structure the pressures do not explain (soils, rainfall, livestock density, phosphorus
removal at individual works). Those are the next data to add.
"""),
    md("## The ordinal target: invertebrate class from bad (0) to high (4)"),
    code("""
display(pd.read_csv(T / "ordinal_inv_class.csv").round(3))
display(Image(FIG / "ordinal_confusion.png"))
"""),
    md("""
The Frank & Hall decomposition fits four monotone binary models, P(class > k) for k = 0..3, and turns
them into class probabilities. It respects the ordering that a plain multiclass model ignores, and it scores
better on quadratic weighted kappa (0.44 vs 0.41), which penalises a bad-as-high mistake more than a
good-as-high one. Its log loss is slightly worse (1.23 vs 1.21): ordering helps rank the classes, not sharpen
the probabilities.
"""),
]

# ----------------------------------------------------------------------------- 06
N06 = [
    md("""
# 06. What drives the models? SHAP, grouped by pressure, checked against the EA's investigators

SHAP values split each prediction (in log-odds of good status) into additive contributions from each
feature. For trees, `TreeExplainer` computes them exactly. Correlated features share credit
arbitrarily, so contributions are summed into **pressure groups**: storm overflows, sewage works,
private sewage, arable, grassland, urban land, trade and fish farms, and the natural setting.
"""),
    SETUP,
    code("""display(Image(FIG / "shap_groups.png"))"""),
    code("""
rows = {}
for tg in ["eco_good", "inv_good", "phos_good"]:
    g = pd.read_parquet(T / f"shap_groups_{tg}.parquet").drop(columns="water_body_id")
    rows[tg] = (g.abs().mean() / g.abs().mean().sum())
pd.DataFrame(rows).round(3).sort_values("phos_good", ascending=False)
"""),
    code("""display(Image(FIG / "shap_beeswarm_phos_good.png"))"""),
    code("""display(Image(FIG / "shap_beeswarm_inv_good.png"))"""),
    md("## Shape of the main effects"),
    code("""display(Image(FIG / "shap_dependence.png"))"""),
    md("""
Points are coloured by upstream urban share. The monotone constraints make every curve one-directional;
what the data decides is where each effect bites and how hard.
"""),
    md("""
## Does the model blame the same pressures as the Environment Agency?

For every failing water body the EA records **Reasons for Not Achieving Good** (RNAG): the sector and
activity its staff judged responsible. These were never shown to the model. If the model's SHAP blame for
a pressure group ranks water bodies the way the EA's flags do (AUC > 0.5), the explanation is pointing at
something real.
"""),
    code("""
display(Image(FIG / "rnag_agreement.png"))
pd.read_csv(T / "rnag_agreement.csv").round(3)
"""),
]

# ----------------------------------------------------------------------------- 07
N07 = [
    md("""
# 07. Scenarios: what would cutting storm overflow spills achieve?

The fitted champion models are re-scored with spill features scaled down: spill hours and counts,
and the number of frequently spilling overflows. Everything else is held fixed. For comparison, the same is
done for sewage works effluent and urban land.

**Read these as model-based expectations, not causal effects.** The models learned associations across
rivers. Monotone constraints guarantee the direction, but the size of each effect may still borrow
from correlated, unmeasured pressures.
"""),
    SETUP,
    code("""
s = pd.read_csv(T / "scenarios.csv")
s[["target", "scenario", "expected_good_base", "expected_good", "extra_good_wbs"]].round(1)
"""),
    code("""display(Image(FIG / "scenarios.png"))"""),
    md("## Where would stopping spills matter most? (invertebrates)"),
    code("""display(Image(FIG / "map_spill_gain_inv.png"))"""),
    code("""
p = pd.read_parquet(T / "scenario_per_wb.parquet")
top = (p[p.target == "inv_good"].merge(t[["water_body_id", "name", "management_catchment", "local_n_overflows",
                                         "up_n_overflows", "up_spill_hours", "y_inv"]], on="water_body_id")
       .sort_values("gain", ascending=False).head(20))
top[["name", "management_catchment", "y_inv", "up_n_overflows", "up_spill_hours", "p_base", "p_no_spills", "gain"]].round(2)
"""),
    md("""
The top of the priority list is local: the Calder, Aire, Irwell, Mersey and Don, the old industrial river
systems of West Yorkshire, Greater Manchester and South Yorkshire. Each has hundreds of overflows upstream, and
the model puts its chance of good invertebrates near 30%, rising to about 60% without spills. The gain is concentrated: the top 200 water bodies (5% of the total) carry about half of the predicted
invertebrate gain. These are places where spills are heavy relative to the river's size, so removing them
could tip the balance. Elsewhere, removing spills leaves the river failing for other reasons: sewage works
effluent, urban runoff, farming.
"""),
]

ALL = {"01": ("01_data_and_network.ipynb", N01), "02": ("02_pressures_and_status.ipynb", N02),
       "03": ("03_decision_tree.ipynb", N03), "04": ("04_random_forest.ipynb", N04),
       "05": ("05_boosting_and_validation.ipynb", N05), "06": ("06_explaining_the_models.ipynb", N06),
       "07": ("07_spill_scenarios.ipynb", N07)}

if __name__ == "__main__":
    pick = sys.argv[1:] or list(ALL)
    for k in pick:
        name, cells = ALL[k]
        write_and_run(cells, NB / name)
