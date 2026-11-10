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
STATUS_COL = {"eco_good": "y_eco", "inv_good": "y_inv", "phos_good": "y_phos"}
GROUPS = ["Storm overflows", "Sewage works", "Private sewage", "Arable", "Grassland", "Urban land",
          "Trade & fish farms", "Natural setting"]

st.set_page_config(page_title="England river health", layout="wide")


@st.cache_data
def load(name, **kw):
    f = T / name
    if not f.exists():
        return None
    return pd.read_parquet(f) if name.endswith(".parquet") else pd.read_csv(f, **kw)


rv = load("app_rivers.parquet")
bench = load("benchmark.csv")
champ = load("champions.csv")
scen = load("scenarios.csv")
rnag = load("rnag_agreement.csv")
checks = load("validation_checks.csv")
nested = load("nested_tuning.csv")

st.title("What drives the health of England's rivers?")
st.caption("3,772 river water bodies, Water Framework Directive 2025 classification (Environment Agency). "
           "Pressures: storm overflow spills 2021-24, sewage works, private sewage, land cover and the river "
           "network inferred from EA centrelines. Tree ensembles validated on whole catchments held out.")

tabs = st.tabs(["Overview", "Explore a river", "What drives status", "Spill scenarios", "Models", "Validation"])

# ------------------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Good ecological status", f"{(rv.y_eco.isin(['Good', 'High'])).mean():.0%}")
    c[1].metric("Invertebrates good or high", f"{(rv.y_inv.isin(['Good', 'High'])).sum() / rv.y_inv.notna().sum():.0%}")
    c[2].metric("Phosphate good or high", f"{(rv.y_phos.isin(['Good', 'High'])).sum() / rv.y_phos.notna().sum():.0%}")
    c[3].metric("Overflows linked to rivers", f"{int(rv.local_n_overflows.sum()):,}")
    left, right = st.columns([1.1, 1])
    with left:
        which = st.radio("Map", list(TARGETS.values()), horizontal=True, key="ovmap")
        tg = [k for k, v in TARGETS.items() if v == which][0]
        m = rv.assign(status=rv[STATUS_COL[tg]].fillna("Not assessed"))
        fig = px.scatter_map(m, lat="lat", lon="lon", color="status", color_discrete_map=COLOURS,
                             category_orders={"status": ORDER + ["Not assessed"]}, hover_name="name",
                             hover_data={"lat": False, "lon": False, "management_catchment": True},
                             zoom=5.2, center={"lat": 52.8, "lon": -1.6}, height=620, map_style="carto-positron")
        fig.update_traces(marker_size=5)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.98, x=0.01))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Key findings")
        if champ is not None:
            a = champ[champ.model == "lightgbm_tuned_mono"].set_index("target").roc_auc
            st.markdown(f"""
* **Phosphate** is the most predictable element from pressures (spatial-CV AUC {a.get('phos_good', np.nan):.2f}).
  It is driven by **sewage works effluent** relative to catchment size, urban land and arable farming.
* **Invertebrates** (AUC {a.get('inv_good', np.nan):.2f}) respond most to **urban land**, overflows and organic pollution.
* **Overall status** (AUC {a.get('eco_good', np.nan):.2f}) is one-out-all-out: only 15% of rivers pass every element.
* Fish status is barely predictable from these pressures (AUC ≈ 0.6): barriers and habitat are missing from the data.
* A logistic baseline on log-pressures is almost as accurate as boosting except for phosphate; random folds
  overstate every model, the flexible ones most.
""")
        if scen is not None:
            s = scen[scen.scenario == "spills -100%"].set_index("target")
            st.markdown("**If every storm overflow stopped spilling** (model expectation, not a causal estimate):")
            st.dataframe(pd.DataFrame({"element": [TARGETS[k] for k in s.index],
                                       "water bodies good now (expected)": s.expected_good_base.round(0),
                                       "with no spills": s.expected_good.round(0),
                                       "extra": s.extra_good_wbs.round(0)}).reset_index(drop=True),
                         hide_index=True, width="stretch")

# ------------------------------------------------------------------------------ explore
with tabs[1]:
    rv["label"] = rv.name + " (" + rv.water_body_id + ")"
    default = rv.label[rv.name.str.startswith("Thames (Egham")].iloc[0] if rv.name.str.startswith("Thames (Egham").any() else rv.label.iloc[0]
    pick = st.selectbox("River water body", sorted(rv.label), index=sorted(rv.label).index(default))
    r = rv[rv.label == pick].iloc[0]
    c = st.columns(4)
    for col, (lab, k) in zip(c, [("Ecological", "y_eco"), ("Invertebrates", "y_inv"), ("Phosphate", "y_phos"), ("Fish", "y_fish")]):
        col.metric(lab, r[k] if isinstance(r[k], str) else "not assessed")
    c = st.columns(4)
    c[0].metric("Upstream catchment", f"{r.up_area_km2:,.0f} km²")
    c[1].metric("Overflows upstream", f"{int(r.up_n_overflows):,}", f"{r.up_spill_hours:,.0f} spill h/yr", delta_color="off")
    c[2].metric("Sewage works upstream", f"{int(r.up_n_stw):,}", f"{r.up_stw_dwf_m3d_per_km2:,.1f} m³/d per km²", delta_color="off")
    c[3].metric("Urban / arable upstream", f"{r.up_lc_urban:.0%} / {r.up_lc_arable:.0%}")
    st.markdown("**Model view.** Probability of good status from pressures alone (out-of-fold, i.e. predicted "
                "while this river's whole catchment was held out), and which pressures push it down.")
    cols = st.columns(3)
    for col, tg in zip(cols, TARGETS):
        with col:
            p = r[f"p_{tg}"]
            st.markdown(f"**{TARGETS[tg]}**: P(good) = {p:.0%}" if pd.notna(p) else f"**{TARGETS[tg]}**: not assessed")
            vals = pd.Series({g: r.get(f"shap_{tg}__{g}", np.nan) for g in GROUPS}).dropna()
            if len(vals):
                vals = vals.sort_values()
                fig = go.Figure(go.Bar(x=vals.values, y=vals.index, orientation="h",
                                       marker_color=np.where(vals.values < 0, "#d62728", "#2ca02c")))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="SHAP (log-odds of good)")
                st.plotly_chart(fig, width="stretch", key=f"shap_{tg}")
                g = r.get(f"gain_nospill_{tg}")
                if pd.notna(g):
                    st.caption(f"Stopping all spills upstream: P(good) {p:.0%} → {p + g:.0%} (fitted model on all data).")
    flags = [k.replace("rnag_", "") for k in ["rnag_overflow", "rnag_sewage", "rnag_agriculture", "rnag_urban"] if r[k] == 1]
    st.caption("EA reasons for not achieving good (Cycle 3) recorded here: " + (", ".join(flags) if flags else "none of overflow/sewage/agriculture/urban"))

# ------------------------------------------------------------------------------ drivers
with tabs[2]:
    rows = []
    for tg in TARGETS:
        cols = [f"shap_{tg}__{g}" for g in GROUPS]
        v = rv[cols].abs().mean()
        rows += [dict(target=TARGETS[tg], group=g, importance=v[f"shap_{tg}__{g}"]) for g in GROUPS]
    imp = pd.DataFrame(rows)
    fig = px.bar(imp, x="importance", y="group", color="target", barmode="group", orientation="h", height=460,
                 labels={"importance": "mean |SHAP| (log-odds), features grouped by pressure", "group": ""})
    st.plotly_chart(fig, width="stretch")
    st.caption("Correlated features share SHAP credit arbitrarily, so contributions are summed over pressure groups.")
    tg = st.radio("Element", list(TARGETS.values()), horizontal=True, key="dep")
    tg = [k for k, v in TARGETS.items() if v == tg][0]
    grp = st.selectbox("Pressure group", GROUPS, index=1)
    xvar = {"Storm overflows": "up_spill_hours_per_km2", "Sewage works": "up_stw_dwf_m3d_per_km2",
            "Urban land": "up_lc_urban", "Arable": "up_lc_arable", "Grassland": "up_lc_improved_grass"}.get(grp, "up_area_km2")
    fig = px.scatter(rv, x=xvar, y=f"shap_{tg}__{grp}", color=STATUS_COL[tg], color_discrete_map=COLOURS,
                     category_orders={STATUS_COL[tg]: ORDER}, hover_name="name", opacity=0.6, height=430,
                     log_x=xvar not in ("up_lc_urban", "up_lc_arable", "up_lc_improved_grass"))
    fig.update_layout(yaxis_title=f"SHAP of {grp}", legend_title="observed")
    st.plotly_chart(fig, width="stretch")
    if rnag is not None:
        st.subheader("Does the model blame what the EA's investigators blamed?")
        st.dataframe(rnag.assign(target=rnag.target.map(TARGETS)).round(3), hide_index=True, width="stretch")
        st.caption("Among failing water bodies: AUC of the model's SHAP blame for a pressure group against the EA's "
                   "Reasons for Not Achieving Good flag for that sector. 0.5 = no agreement. The flags were never used in training.")

# ------------------------------------------------------------------------------ scenarios
with tabs[3]:
    if scen is not None:
        s = scen.assign(element=scen.target.map(TARGETS))
        fig = px.bar(s, x="extra_good_wbs", y="scenario", color="element", barmode="group", orientation="h", height=420,
                     labels={"extra_good_wbs": "expected extra water bodies at good status", "scenario": ""})
        st.plotly_chart(fig, width="stretch")
        st.caption("Each scenario rescales one pressure and re-scores the fitted monotone models, holding everything else fixed. "
                   "These are model-based expectations under the assumption that the learned associations are causal; "
                   "they are not estimates from intervention data.")
    tg = st.radio("Element", list(TARGETS.values()), horizontal=True, key="scen")
    tg = [k for k, v in TARGETS.items() if v == tg][0]
    top = rv.sort_values(f"gain_nospill_{tg}", ascending=False).head(25)
    fig = px.scatter_map(rv.assign(gain=rv[f"gain_nospill_{tg}"].clip(lower=0)), lat="lat", lon="lon",
                         color="gain", size=rv[f"gain_nospill_{tg}"].clip(lower=0) + 0.005,
                         color_continuous_scale="Greens", hover_name="name", zoom=5.2,
                         center={"lat": 52.8, "lon": -1.6}, height=560, map_style="carto-positron")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar_title="rise in P(good)")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Where stopping spills is predicted to matter most**")
    st.dataframe(top[["name", "management_catchment", STATUS_COL[tg], "up_n_overflows", "up_spill_hours",
                      f"p_{tg}", f"gain_nospill_{tg}"]].round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------ models
with tabs[4]:
    if bench is not None:
        b = bench.assign(model=bench.model + np.where(bench.monotone, " (monotone)", ""))
        cv = st.radio("Cross-validation", ["spatial", "random"], horizontal=True)
        fs = st.radio("Features", sorted(b.features.unique()), horizontal=True)
        w = b[(b.cv == cv) & (b.features == fs)].pivot(index="model", columns="target", values="roc_auc")
        st.dataframe(w.round(3).sort_values(w.columns[-1], ascending=False), width="stretch")
        st.caption("ROC AUC out of fold. Spatial CV holds out whole management catchments (90 groups, 5 folds); "
                   "random CV mixes neighbouring rivers between train and test.")
    if champ is not None:
        st.subheader("Champions: tuned monotone LightGBM vs baselines (spatial CV)")
        st.dataframe(champ.round(3), hide_index=True, width="stretch")
    if nested is not None:
        st.caption("Nested tuning (Optuna inside each outer fold): " + "; ".join(
            f"{TARGETS.get(r.target, r.target)} AUC {r.test_auc:.3f}" for r in nested[nested.fold == "all"].itertuples()))
    for f in ["calibration.png", "cv_optimism.png", "tree_pruning.png", "forest_oob_vs_spatial.png"]:
        if (FIG / f).exists():
            st.image(str(FIG / f))

# ------------------------------------------------------------------------------ validation
with tabs[5]:
    if checks is not None:
        st.dataframe(checks, hide_index=True, width="stretch")
        st.caption(f"{int(checks.passed.sum())}/{len(checks)} independent checks passed (scripts/validate.py).")
    st.markdown("""
**Data:** Environment Agency EDM storm overflow annual returns 2020-2025; WFD Cycle 4 2025 classification and river
water body catchments; Catchment Data Explorer reasons for not achieving good; Water Quality Archive; consented
discharges register (EA Conditional Licence); Natural England Living England Habitat Map Phase 4. © Environment Agency
and Natural England copyright and/or database right. Open Government Licence v3.0 unless stated.
""")
