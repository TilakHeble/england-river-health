"""England river health dashboard.

Reads the precomputed tables in outputs/tables (no model fitting at runtime).
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
ORDER = ["High", "Good", "Moderate", "Poor", "Bad"]
COLOURS = {"High": "#1f78b4", "Good": "#33a02c", "Moderate": "#ffd92f", "Poor": "#ff7f00", "Bad": "#e31a1c",
           "Not assessed": "#bbbbbb"}
TARGETS = {"eco_good": "Ecological status", "inv_good": "Invertebrates", "phos_good": "Phosphate"}
ALL_TARGETS = {**TARGETS, "fish_good": "Fish"}
STATUS_COL = {"eco_good": "y_eco", "inv_good": "y_inv", "phos_good": "y_phos"}
GROUPS = ["Urban land", "Sewage works", "Natural setting", "Arable", "Storm overflows", "Grassland",
          "Private sewage", "Trade & fish farms"]
MODEL_NAMES = {"logistic": "Logistic (log pressures)", "tree": "Decision tree", "random_forest": "Random forest",
               "extra_trees": "Extra trees", "hist_gb": "Hist. gradient boosting", "xgboost": "XGBoost",
               "lightgbm": "LightGBM", "catboost": "CatBoost", "lightgbm_tuned_mono": "LightGBM tuned, monotone (champion)"}
LABELS = {"name": "Water body", "management_catchment": "Catchment", "y_eco": "Ecological", "y_inv": "Invertebrates",
          "y_phos": "Phosphate", "up_n_overflows": "Overflows upstream", "up_spill_hours": "Spill hours/yr upstream",
          "group": "Pressure group", "rnag_flag": "EA reason-for-failure flag", "n_failing": "Failing water bodies",
          "flag_rate": "Share flagged", "auc": "AUC (agreement)", "target": "Element"}
PRESSURE_X = {"Storm overflows": ("up_spill_hours_per_km2", "Spill hours per km² upstream (+1, log scale)", True),
              "Sewage works": ("up_stw_dwf_m3d_per_km2", "Sewage works dry-weather flow, m³/day per km² upstream (+1, log scale)", True),
              "Urban land": ("up_lc_urban", "Urban share of upstream catchment", False),
              "Arable": ("up_lc_arable", "Arable share of upstream catchment", False),
              "Grassland": ("up_lc_improved_grass", "Improved grassland share of upstream catchment", False),
              "Natural setting": ("up_area_km2", "Upstream catchment area, km² (log scale)", True),
              "Private sewage": ("up_area_km2", "Upstream catchment area, km² (log scale)", True),
              "Trade & fish farms": ("up_area_km2", "Upstream catchment area, km² (log scale)", True)}

st.set_page_config(page_title="England river health", page_icon="🌊", layout="wide")


@st.cache_data
def load(name, **kw):
    f = T / name
    if not f.exists():
        return None
    return pd.read_parquet(f) if name.endswith(".parquet") else pd.read_csv(f, **kw)


def pick_target(label, key, options=TARGETS):
    v = st.radio(label, list(options.values()), horizontal=True, key=key)
    return [k for k, x in options.items() if x == v][0]


rv = load("app_rivers.parquet")
bench = load("benchmark.csv")
champ = load("champions.csv")
scen = load("scenarios.csv")
rnag = load("rnag_agreement.csv")
checks = load("validation_checks.csv")
nested = load("nested_tuning.csv")
nested_auc = {} if nested is None else nested[nested.fold == "all"].set_index("target").test_auc.to_dict()

st.title("What drives the health of England's rivers?")
st.caption("3,772 river water bodies, Water Framework Directive 2025 classification (Environment Agency). "
           "Pressures: storm overflow spills 2021-24, sewage works, private sewage, land cover and a river "
           "network inferred from EA centrelines. Tree ensembles validated on whole catchments held out.")

tabs = st.tabs(["Overview", "Explore a river", "What drives status", "Spill scenarios", "Models", "Validation", "About"])

# ------------------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Good ecological status", f"{(rv.y_eco.isin(['Good', 'High'])).mean():.0%}")
    c[1].metric("Invertebrates good or high", f"{(rv.y_inv.isin(['Good', 'High'])).sum() / rv.y_inv.notna().sum():.0%}")
    c[2].metric("Phosphate good or high", f"{(rv.y_phos.isin(['Good', 'High'])).sum() / rv.y_phos.notna().sum():.0%}")
    c[3].metric("Storm overflows linked to rivers", f"{int(rv.local_n_overflows.sum()):,}")
    left, right = st.columns([1.1, 1])
    with left:
        tg = pick_target("Map", "ovmap")
        m = rv.assign(status=rv[STATUS_COL[tg]].fillna("Not assessed"))
        fig = px.scatter_map(m, lat="lat", lon="lon", color="status", color_discrete_map=COLOURS,
                             category_orders={"status": ORDER + ["Not assessed"]}, hover_name="name",
                             hover_data={"lat": False, "lon": False, "management_catchment": True},
                             zoom=5.1, center={"lat": 52.8, "lon": -1.6}, height=620, map_style="carto-positron")
        fig.update_traces(marker_size=5)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.98, x=0.01, title="2025 class"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each dot is the centre of one river water body's catchment.")
    with right:
        st.subheader("Key findings")
        st.markdown(f"""
* **Urban land is the largest pressure** on every element (25-38% of the models' explanation).
* **For phosphate, treated sewage matters more than overflows**: sewage works effluent per km² carries 23%
  of the explanation, storm overflows 5%.
* **Overflows are a real but secondary national pressure.** Stopping every spill is expected to lift about
  50 more rivers to good ecological status; half the invertebrate gain sits in 200 rivers, led by the
  Calder, Aire, Irwell, Mersey and Don.
* **The model blames what the EA's investigators blame**: urban AUC 0.84-0.88, sewage works 0.75, overflows 0.64-0.72.
* **Accuracy on unseen catchments** (nested spatial CV, AUC): ecological status {nested_auc.get('eco_good', np.nan):.2f},
  invertebrates {nested_auc.get('inv_good', np.nan):.2f}, phosphate {nested_auc.get('phos_good', np.nan):.2f}.
  Fish status is barely predictable (≈ 0.6): barriers and habitat are not in the data.
""")
        if scen is not None:
            s = scen[scen.scenario == "spills -100%"].set_index("target")
            st.markdown("**If every storm overflow stopped spilling** (model expectation, not a causal estimate):")
            st.dataframe(pd.DataFrame({"Element": [TARGETS[k] for k in s.index],
                                       "Rivers at good now (expected)": s.expected_good_base.round(0).astype(int),
                                       "With no spills": s.expected_good.round(0).astype(int),
                                       "Extra": s.extra_good_wbs.round(0).astype(int)}),
                         hide_index=True, width="stretch")

# ------------------------------------------------------------------------------ explore
with tabs[1]:
    labels = (rv.name + " (" + rv.water_body_id + ")").tolist()
    order = sorted(labels)
    default = next((l for l in order if l.startswith("Thames (Egham")), order[0])
    pick = st.selectbox("River water body (type to search)", order, index=order.index(default))
    r = rv.iloc[labels.index(pick)]
    st.caption(f"{r.management_catchment} management catchment · {r.rbd} river basin district · "
               f"{r.hmwb.replace('_', ' ')} · {r.altitude} altitude, {r.geology} geology")
    c = st.columns(4)
    for col, (lab, k) in zip(c, [("Ecological", "y_eco"), ("Invertebrates", "y_inv"), ("Phosphate", "y_phos"), ("Fish", "y_fish")]):
        col.metric(lab, r[k] if isinstance(r[k], str) else "not assessed")
    c = st.columns(4)
    c[0].metric("Upstream catchment", f"{r.up_area_km2:,.0f} km²")
    c[1].metric("Overflows upstream", f"{int(r.up_n_overflows):,}", f"{r.up_spill_hours:,.0f} spill hours/yr", delta_color="off")
    c[2].metric("Sewage works upstream", f"{int(r.up_n_stw):,}", f"{r.up_stw_dwf_m3d_per_km2:,.1f} m³/day per km²", delta_color="off")
    c[3].metric("Urban / arable upstream", f"{r.up_lc_urban:.0%} / {r.up_lc_arable:.0%}")
    st.markdown("**Model view.** Probability of good status from pressures alone, predicted while this river's whole "
                "catchment was held out of training, and which pressures push it down (red) or up (green).")
    cols = st.columns(3)
    for col, tg in zip(cols, TARGETS):
        with col:
            p = r[f"p_{tg}"]
            if pd.isna(p):
                st.markdown(f"**{TARGETS[tg]}**: not assessed")
                continue
            st.markdown(f"**{TARGETS[tg]}**: P(good) = {p:.0%}")
            vals = pd.Series({g: r.get(f"shap_{tg}__{g}", np.nan) for g in GROUPS}).dropna().sort_values()
            fig = go.Figure(go.Bar(x=vals.values, y=vals.index, orientation="h",
                                   marker_color=np.where(vals.values < 0, "#d62728", "#2ca02c")))
            fig.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="SHAP, log-odds of good status")
            st.plotly_chart(fig, width="stretch", key=f"shap_{tg}")
            a, b = r.get(f"pfit_{tg}"), r.get(f"pnospill_{tg}")
            if pd.notna(a) and pd.notna(b):
                st.caption(f"Stopping every spill upstream: {a:.0%} → {b:.0%} (final model fitted on all rivers).")
    flags = [n for k, n in [("rnag_overflow", "storm overflows"), ("rnag_sewage", "sewage"),
                            ("rnag_agriculture", "agriculture"), ("rnag_urban", "urban and transport")] if r[k] == 1]
    st.caption("EA reasons for not achieving good recorded here (Cycle 3): " + (", ".join(flags) if flags else "none of these four sectors"))

# ------------------------------------------------------------------------------ drivers
with tabs[2]:
    rows = []
    for tg in TARGETS:
        v = rv[[f"shap_{tg}__{g}" for g in GROUPS]].abs().mean()
        tot = v.sum()
        rows += [dict(Element=TARGETS[tg], group=g, share=v[f"shap_{tg}__{g}"] / tot) for g in GROUPS]
    imp = pd.DataFrame(rows)
    order_g = imp.groupby("group").share.mean().sort_values(ascending=False).index.tolist()
    fig = px.bar(imp, x="share", y="group", color="Element", barmode="group", orientation="h", height=470,
                 category_orders={"group": order_g}, labels={"share": "Share of the model's explanation (mean |SHAP|)", "group": ""})
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption("Correlated features share SHAP credit arbitrarily, so contributions are summed over pressure groups.")

    st.subheader("How each pressure acts")
    tg = pick_target("Element", "dep")
    grp = st.selectbox("Pressure group", [g for g in GROUPS if g in ("Urban land", "Sewage works", "Storm overflows", "Arable", "Grassland")])
    xvar, xlab, logx = PRESSURE_X[grp]
    d = rv.dropna(subset=[f"shap_{tg}__{grp}"]).copy()
    d["x"] = d[xvar] + (1 if logx and "area" not in xvar else 0)
    d["status"] = d[STATUS_COL[tg]].fillna("Not assessed")
    fig = px.scatter(d, x="x", y=f"shap_{tg}__{grp}", color="status", color_discrete_map=COLOURS,
                     category_orders={"status": ORDER + ["Not assessed"]}, hover_name="name", opacity=0.6,
                     height=440, log_x=logx)
    fig.update_traces(marker_size=5)
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(xaxis_title=xlab, yaxis_title=f"SHAP of {grp.lower()} (log-odds)", legend_title="2025 class")
    st.plotly_chart(fig, width="stretch")
    st.caption("Below zero, the pressure lowers the chance of good status. Monotone constraints mean more overflow, "
               "sewage-works or urban pressure can never raise it; arable and grassland are left free.")

    if rnag is not None:
        st.subheader("Does the model blame what the EA's investigators blamed?")
        tab = rnag.assign(target=rnag.target.map(TARGETS), rnag_flag=rnag.rnag_flag.str.replace("rnag_", ""))
        tab = tab.pivot(index=["group", "rnag_flag"], columns="target", values="auc").reset_index()
        st.dataframe(tab.rename(columns=LABELS).round(2), hide_index=True, width="stretch")
        st.caption("Among failing water bodies: AUC of the model's SHAP blame for a pressure group against the EA's "
                   "Reasons for Not Achieving Good flag for that sector (0.5 = no agreement). The flags were never used in training.")

# ------------------------------------------------------------------------------ scenarios
with tabs[3]:
    if scen is not None:
        s = scen.assign(Element=scen.target.map(TARGETS))
        sc_order = ["spills -25%", "spills -50%", "spills -75%", "spills -100%",
                    "sewage works effluent -50% (yardstick)", "urban land -50% (yardstick)"]
        fig = px.bar(s, x="extra_good_wbs", y="scenario", color="Element", barmode="group", orientation="h", height=430,
                     category_orders={"scenario": sc_order},
                     labels={"extra_good_wbs": "Expected extra rivers at good status", "scenario": ""})
        st.plotly_chart(fig, width="stretch")
        st.caption("Each scenario rescales one pressure and re-scores the fitted monotone models, holding everything else "
                   "fixed. These are model-based expectations that assume the associations learned across rivers would hold "
                   "within a river; they are not estimates from interventions.")
    st.subheader("Where stopping spills is predicted to matter most")
    tg = pick_target("Element", "scen")
    g = rv[f"gain_nospill_{tg}"]
    m = rv.assign(gain=g.clip(lower=0)).dropna(subset=["gain"]).sort_values("gain")
    fig = px.scatter_map(m, lat="lat", lon="lon", color="gain", color_continuous_scale="YlGn",
                         range_color=(0, max(0.05, float(m.gain.quantile(0.995)))), hover_name="name",
                         hover_data={"lat": False, "lon": False, "gain": ":.1%"}, zoom=5.1,
                         center={"lat": 52.8, "lon": -1.6}, height=580, map_style="carto-positron")
    fig.update_traces(marker_size=6)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title="Rise in<br>P(good)", tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")
    top = rv.assign(gain=g).sort_values("gain", ascending=False).head(25)
    show = top[["name", "management_catchment", STATUS_COL[tg], "up_n_overflows", "up_spill_hours",
                f"pfit_{tg}", f"pnospill_{tg}", "gain"]].rename(columns={**LABELS, f"pfit_{tg}": "P(good) now",
                                                                       f"pnospill_{tg}": "P(good), no spills", "gain": "Rise"})
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "P(good) now": st.column_config.NumberColumn(format="percent"),
        "P(good), no spills": st.column_config.NumberColumn(format="percent"),
        "Rise": st.column_config.NumberColumn(format="percent"),
        "Spill hours/yr upstream": st.column_config.NumberColumn(format="%.0f"),
        "Overflows upstream": st.column_config.NumberColumn(format="%.0f")})

# ------------------------------------------------------------------------------ models
with tabs[4]:
    if bench is not None:
        b = bench.assign(Model=bench.model.map(MODEL_NAMES) + np.where(bench.monotone, " (monotone)", ""))
        c1, c2 = st.columns(2)
        cv = c1.radio("Cross-validation", ["spatial", "random"], horizontal=True,
                      help="Spatial: whole management catchments held out. Random: neighbouring rivers mixed between train and test.")
        fs = c2.radio("Features", ["pressures", "pressures+location", "pressures+chemistry"], horizontal=True)
        sel = b[(b.cv == cv) & (b.features == fs)]
        if sel.empty:
            st.info("Random folds were run for the pressure features only (to measure their optimism). "
                    "Choose spatial cross-validation to compare feature sets.")
        else:
            w = sel.pivot(index="Model", columns="target", values="roc_auc").rename(columns=ALL_TARGETS)
            w = w[[v for v in ALL_TARGETS.values() if v in w.columns]]
            st.dataframe(w.sort_values(w.columns[0], ascending=False).round(3), width="stretch")
        st.caption("ROC AUC out of fold. Random CV and the extra feature sets are shown to measure optimism and "
                   "what the pressures miss; the champion uses pressures only, scored with spatial CV.")
    if champ is not None:
        st.subheader("Champion vs baselines (spatial CV)")
        ch = champ.assign(Element=champ.target.map(TARGETS), Model=champ.model.map(MODEL_NAMES))
        st.dataframe(ch[["Element", "Model", "roc_auc", "pr_auc", "log_loss", "brier", "base_rate"]].rename(
            columns={"roc_auc": "ROC AUC", "pr_auc": "PR AUC", "log_loss": "Log loss", "brier": "Brier",
                     "base_rate": "Share good"}).round(3), hide_index=True, width="stretch")
    for f, cap in [("benchmark_auc.png", "Every model, spatial CV (bars) and random CV (ticks)"),
                   ("calibration.png", "Calibration of out-of-fold probabilities"),
                   ("cv_optimism.png", "How much random folds flatter each model"),
                   ("tree_pruning.png", "A single tree: pruning judged on unseen catchments"),
                   ("forest_oob_vs_spatial.png", "Random forest: out-of-bag vs spatial CV")]:
        if (FIG / f).exists():
            st.image(str(FIG / f), caption=cap)

# ------------------------------------------------------------------------------ validation
with tabs[5]:
    if checks is not None:
        st.markdown(f"**{int(checks.passed.sum())} of {len(checks)} independent checks passed** (`scripts/validate.py`).")
        for row in checks.itertuples():
            st.markdown(f"{'✅' if row.passed else '❌'} **{row.check}**  \n<small>{row.detail}</small>",
                        unsafe_allow_html=True)

# ------------------------------------------------------------------------------ about
with tabs[6]:
    st.markdown("""
### About this project

Only 15% of England's rivers reached good ecological status in the Environment Agency's 2025 classification.
This dashboard explores **which pressures explain why rivers fail**: storm overflow spills, treated sewage from
sewage works, septic tanks, farming and towns. It also asks **where reducing spills would most likely change a
river's status**.

**How it works**
* **Data:** five Environment Agency and Natural England open datasets joined on one river network:
  * storm overflow spills 2021-24 (national totals reproduce the EA's published figures exactly);
  * every sewage works with its permitted flow;
  * habitat sampled at about 2 million points;
  * 650,000 river chemistry samples;
  * the 2025 Water Framework Directive classification.
* **River network:** the EA publishes no downstream links, so they were inferred from river centrelines.
  Upstream areas match gauged catchments (Trent 8,240 vs 8,231 km²).
* **Models:** decision tree, random forest, XGBoost, LightGBM and CatBoost, against a logistic baseline. The champion
  is LightGBM, tuned with Optuna inside nested cross-validation and constrained so that more pollution can never
  raise the predicted chance of good status.
* **Validation:** whole management catchments held out (spatial cross-validation), calibration checks, and a test of
  the model's explanations against the EA investigators' own reasons for failure.

**Read with care**
* The models learn associations across rivers. Scenarios are model expectations, not causal estimates.
* Livestock density, soils, rainfall, river flow and fish barriers are not in the data.
* Welsh headwaters and coastal catchments are outside the EA data used.

**Project:** part of a portfolio with one data science project per model type, by Tilak Heble.
Code, notebooks, documentation and a from-scratch guide to tree-based models:
[github.com/TilakHeble/england-river-health](https://github.com/TilakHeble/england-river-health).

**Data:** Environment Agency (EDM storm overflow annual returns 2020-2025; WFD Cycle 4 classification and
catchments; Catchment Data Explorer; Water Quality Archive; consented discharges register) and Natural England
(Living England Habitat Map Phase 4). © Environment Agency and Natural England copyright and/or database right.
Open Government Licence v3.0; the consented discharges register is under the EA Conditional Licence.
""")
