"""Draw every figure in outputs/figures from outputs/tables and data/processed.

    python scripts/make_figures.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from river_health import modeling as M  # noqa: E402

T, FIG, PROC = ROOT / "outputs/tables", ROOT / "outputs/figures", ROOT / "data/processed"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10})

WFD_COLOURS = {"High": "#1f78b4", "Good": "#33a02c", "Moderate": "#ffd92f", "Poor": "#ff7f00",
               "Bad": "#e31a1c"}
TARGET_LABEL = {"eco_good": "Ecological status", "inv_good": "Invertebrates",
                "phos_good": "Phosphate", "fish_good": "Fish"}
MODEL_LABEL = {"logistic": "Logistic (log features)", "tree": "Decision tree",
               "random_forest": "Random forest", "extra_trees": "Extra trees",
               "hist_gb": "Hist. gradient boosting", "xgboost": "XGBoost", "lightgbm": "LightGBM",
               "catboost": "CatBoost"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def polys():
    g = gpd.read_file(glob.glob(str(ROOT / "data/raw/wfd/c4_2025/*.gpkg"))[0])
    return g[g["water_body_type"] == "River"][["water_body_id", "geometry"]]


# ------------------------------------------------------------------ maps
def maps(t, g):
    gg = g.merge(t, on="water_body_id")
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.5))
    for a, col, title in [(ax[0], "y_eco", "Ecological status 2025"),
                          (ax[1], "y_inv", "Invertebrates 2025"), (ax[2], "y_phos", "Phosphate 2025")]:
        for cls, c in WFD_COLOURS.items():
            sub = gg[gg[col] == cls]
            if len(sub):
                sub.plot(ax=a, color=c, linewidth=0, label=f"{cls} ({len(sub)})")
        gg[gg[col].isna()].plot(ax=a, color="#dddddd", linewidth=0)
        a.set_title(title); a.set_axis_off(); a.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "map_status.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 6.5))
    for a, col, title, cmap in [
            (ax[0], "up_spill_hours_per_km2", "Storm overflow spill hours per km² upstream\n(mean 2021-24)", "Reds"),
            (ax[1], "up_stw_dwf_m3d_per_km2", "Sewage works dry-weather flow\nm³/day per km² upstream", "Purples"),
            (ax[2], "up_lc_arable", "Arable share of upstream catchment", "YlOrBr")]:
        v = gg[col].clip(upper=gg[col].quantile(0.97))
        gg.assign(v=v).plot(ax=a, column="v", cmap=cmap, linewidth=0, legend=True,
                            legend_kwds={"shrink": 0.6})
        a.set_title(title); a.set_axis_off()
    save(fig, "map_pressures.png")


# ------------------------------------------------------------------ benchmark
def benchmark():
    b = pd.read_csv(T / "benchmark.csv")
    b["label"] = b["model"].map(MODEL_LABEL) + np.where(b["monotone"], " (monotone)", "")
    sp = b[(b["features"] == "pressures") & (b["cv"] == "spatial")]
    order = sp[sp["target"] == "phos_good"].sort_values("roc_auc")["label"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    for a, tg in zip(axes, ["eco_good", "inv_good", "phos_good", "fish_good"]):
        s = sp[sp["target"] == tg].set_index("label").reindex(order)
        a.barh(s.index, s["roc_auc"], color=["#1f77b4" if "LightGBM (monotone)" == i else "#9ecae1" for i in s.index])
        r = b[(b["target"] == tg) & (b["cv"] == "random")].set_index("label").reindex(order)
        a.scatter(r["roc_auc"], r.index, color="k", marker="|", s=120, label="random CV", zorder=3)
        a.set_xlim(0.5, 0.9); a.set_title(f"{TARGET_LABEL[tg]} (good: {s['base_rate'].iloc[0]:.0%})")
        a.set_xlabel("ROC AUC, spatial CV")
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, "benchmark_auc.png")

    rnd = b[b["cv"] == "random"].set_index(["target", "label"])["roc_auc"]
    spa = sp.set_index(["target", "label"])["roc_auc"]
    gap = (rnd - spa.reindex(rnd.index)).unstack(0).dropna(how="all")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    gap.reindex(order).dropna(how="all").plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("AUC gained by (wrongly) using random folds")
    ax.legend([TARGET_LABEL[c] for c in gap.columns], fontsize=8)
    save(fig, "cv_optimism.png")

    fs = b[(b["cv"] == "spatial") & (b["model"] == "lightgbm") & (~b["monotone"])]
    w = fs.pivot(index="target", columns="features", values="roc_auc").reindex(
        ["eco_good", "inv_good", "phos_good", "fish_good"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w.rename(index=TARGET_LABEL).plot.bar(ax=ax, rot=0, width=0.8)
    ax.set_ylim(0.5, 0.95); ax.set_ylabel("ROC AUC, spatial CV (LightGBM)")
    save(fig, "feature_sets.png")


# ------------------------------------------------------------------ single tree and forest
def tree_and_forest():
    p = pd.read_csv(T / "tree_pruning.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for tg, d in p[p["ccp_alpha"].notna()].groupby("target"):
        d = d.sort_values("leaves")
        ax[0].plot(d["leaves"], d["roc_auc"], marker="o", ms=3, label=TARGET_LABEL[tg])
    ax[0].set_xscale("log"); ax[0].set_xlabel("leaves after cost-complexity pruning")
    ax[0].set_ylabel("ROC AUC, spatial CV"); ax[0].legend()
    for tg, d in p[p["max_depth"].notna()].groupby("target"):
        ax[1].plot(d["max_depth"], d["roc_auc"], marker="o", label=TARGET_LABEL[tg])
    ax[1].set_xlabel("max depth (unpruned otherwise)"); ax[1].set_ylabel("ROC AUC, spatial CV")
    save(fig, "tree_pruning.png")


def forest_figs():
    f = pd.read_csv(T / "forest_study.csv")
    f = f[f["max_features"].isna()] if "max_features" in f else f
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    for a, (tg, d) in zip(axes, f.groupby("target")):
        for col, lab in [("oob_auc", "out-of-bag"), ("random_cv_auc", "random 5-fold CV"),
                         ("spatial_cv_auc", "spatial 5-fold CV")]:
            a.plot(d["n_trees"], d[col], marker="o", label=lab)
        a.set_xscale("log"); a.set_title(TARGET_LABEL[tg]); a.set_xlabel("trees")
    axes[0].set_ylabel("ROC AUC"); axes[0].legend(fontsize=8)
    save(fig, "forest_oob_vs_spatial.png")

    imp = pd.read_csv(T / "forest_importance.csv")
    d = imp[imp["target"] == "phos_good"].copy()
    d["impurity_rank"] = d["impurity"].rank(ascending=False)
    d["perm_rank"] = d["permutation_auc_drop"].rank(ascending=False)
    top = d.nsmallest(15, "perm_rank").sort_values("permutation_auc_drop")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax[0].barh(top["feature"], top["impurity"], color="#9ecae1"); ax[0].set_title("Impurity (MDI) importance")
    ax[1].barh(top["feature"], top["permutation_auc_drop"], color="#3182bd")
    ax[1].set_title("Permutation importance on held-out catchments\n(drop in AUC)")
    save(fig, "forest_importance_phos.png")


# ------------------------------------------------------------------ calibration
def calibration():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for a, tg in zip(axes, ["eco_good", "inv_good", "phos_good"]):
        o = pd.read_parquet(T / f"oof_{tg}.parquet")
        for col, lab in [("p_logistic", "logistic"), ("p_random_forest", "random forest"),
                         ("p_lightgbm", "LightGBM tuned, monotone")]:
            q = pd.qcut(o[col].rank(method="first"), 10, labels=False)
            c = o.groupby(q).agg(p=(col, "mean"), y=("y", "mean"))
            a.plot(c["p"], c["y"], marker="o", label=lab)
        a.plot([0, 1], [0, 1], "k--", lw=0.8); a.set_title(TARGET_LABEL[tg])
        a.set_xlabel("predicted P(good), decile mean"); a.set_ylabel("observed share good")
    axes[0].legend(fontsize=8)
    save(fig, "calibration.png")


# ------------------------------------------------------------------ SHAP
def shap_figs(t):
    import shap
    from river_health.experiments import GROUPS
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = {}
    for tg in ["eco_good", "inv_good", "phos_good"]:
        g = pd.read_parquet(T / f"shap_groups_{tg}.parquet").drop(columns="water_body_id")
        w[TARGET_LABEL[tg]] = g.abs().mean()
    pd.DataFrame(w).sort_values("Phosphate").plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("mean |SHAP| (log-odds of good status), features grouped by pressure")
    save(fig, "shap_groups.png")

    for tg in ["inv_good", "phos_good", "eco_good"]:
        S = pd.read_parquet(T / f"shap_{tg}.parquet")
        X, y, g, d = M.xy(t, tg, M.FEATURE_SETS["pressures"])
        Xn = X.copy()
        for c in M.CATEGORICAL:
            Xn[c] = Xn[c].cat.codes
        plt.figure()
        shap.summary_plot(S.drop(columns="water_body_id").values, Xn, max_display=15, show=False,
                          plot_size=(9, 6))
        plt.title(f"SHAP: {TARGET_LABEL[tg]} good or better")
        plt.tight_layout(); plt.savefig(FIG / f"shap_beeswarm_{tg}.png", bbox_inches="tight"); plt.close("all")
        print("wrote", f"shap_beeswarm_{tg}.png")

    feats = [("up_spill_hours_per_km2", "spill hours per km² upstream", True),
             ("up_stw_dwf_m3d_per_km2", "sewage works DWF m³/d per km² upstream", True),
             ("up_lc_urban", "urban share upstream", False),
             ("up_lc_arable", "arable share upstream", False)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for r, tg in enumerate(["inv_good", "phos_good"]):
        S = pd.read_parquet(T / f"shap_{tg}.parquet")
        X, *_ = M.xy(t, tg, M.FEATURE_SETS["pressures"])
        for c, (f, lab, logx) in enumerate(feats):
            a = axes[r, c]
            x = X[f].to_numpy()
            a.scatter(x + (1 if logx else 0), S[f], s=4, alpha=0.35, c=X["up_lc_urban"], cmap="viridis")
            if logx:
                a.set_xscale("log")
            a.axhline(0, color="k", lw=0.6)
            a.set_xlabel(lab + (" (+1, log)" if logx else "")); a.set_ylabel(f"SHAP, {TARGET_LABEL[tg]}")
    save(fig, "shap_dependence.png")

    ra = pd.read_csv(T / "rnag_agreement.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ra["lab"] = ra["group"] + " vs EA '" + ra["rnag_flag"].str.replace("rnag_", "") + "'"
    for i, (tg, d) in enumerate(ra.groupby("target")):
        ax.barh(np.arange(len(d)) + i * 0.27, d["auc"], height=0.27, label=TARGET_LABEL[tg])
    ax.set_yticks(np.arange(len(d)) + 0.27); ax.set_yticklabels(d["lab"])
    ax.axvline(0.5, color="k", lw=0.8, ls="--"); ax.set_xlim(0.3, 0.9)
    ax.set_xlabel("AUC: does SHAP blame predict the EA's own reason-for-failure flag?\n(water bodies failing the target only)")
    ax.legend(fontsize=8)
    save(fig, "rnag_agreement.png")


# ------------------------------------------------------------------ scenarios
def scenario_figs(t, g):
    s = pd.read_csv(T / "scenarios.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (tg, d) in enumerate(s.groupby("target", sort=False)):
        ax.barh(np.arange(len(d)) + i * 0.27, d["extra_good_wbs"], height=0.27, label=TARGET_LABEL[tg])
    ax.set_yticks(np.arange(len(d)) + 0.27); ax.set_yticklabels(d["scenario"])
    ax.set_xlabel("expected extra water bodies at good status (model, not a causal estimate)")
    ax.legend(fontsize=8)
    save(fig, "scenarios.png")

    p = pd.read_parquet(T / "scenario_per_wb.parquet")
    p = p[p["target"] == "inv_good"]
    gg = g.merge(p, on="water_body_id")
    fig, ax = plt.subplots(figsize=(7, 7))
    gg.plot(ax=ax, column="gain", cmap="Greens", linewidth=0, legend=True,
            legend_kwds={"shrink": 0.6, "label": "rise in P(invertebrates good) if all spills stopped"})
    ax.set_axis_off(); ax.set_title("Where stopping storm overflow spills is predicted to matter most")
    save(fig, "map_spill_gain_inv.png")


def ordinal_fig():
    cm = pd.read_csv(T / "ordinal_confusion.csv", index_col=0)
    cm.index = ["Bad", "Poor", "Moderate", "Good", "High"][: len(cm)]
    cm.columns = ["Bad", "Poor", "Moderate", "Good", "High"][: cm.shape[1]]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm.div(cm.sum(1), axis=0), cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm.iloc[i, j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(cm.columns, rotation=30)
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(cm.index)
    ax.set_xlabel("predicted (most likely)"); ax.set_ylabel("observed"); ax.grid(False)
    ax.set_title("Invertebrate class, ordinal model\n(row-normalised colour, counts)")
    fig.colorbar(im, shrink=0.8)
    save(fig, "ordinal_confusion.png")


def app_table(t, g):
    """One compact table per river for the dashboard (no fitting at runtime)."""
    c = g.set_index("water_body_id").geometry.centroid.to_crs(4326)
    a = t[["water_body_id", "name", "management_catchment", "rbd", "hmwb", "altitude", "geology",
           "y_eco", "y_inv", "y_phos", "y_fish", "up_area_km2", "local_n_overflows", "up_n_overflows",
           "local_spill_hours", "up_spill_hours", "up_spill_hours_per_km2", "local_n_stw", "up_n_stw",
           "up_stw_dwf_m3d_per_km2", "up_lc_urban", "up_lc_arable", "up_lc_improved_grass",
           "chem_orthophosphate_mgP", "chem_ammonia_mgN", "rnag_overflow", "rnag_sewage",
           "rnag_agriculture", "rnag_urban"]].copy()
    a["lat"], a["lon"] = a["water_body_id"].map(c.y), a["water_body_id"].map(c.x)
    for tg in ["eco_good", "inv_good", "phos_good"]:
        o = pd.read_parquet(T / f"oof_{tg}.parquet").set_index("water_body_id")
        a[f"p_{tg}"] = a["water_body_id"].map(o["p_lightgbm"])
        sg = pd.read_parquet(T / f"shap_groups_{tg}.parquet").set_index("water_body_id")
        for col in sg.columns:
            a[f"shap_{tg}__{col}"] = a["water_body_id"].map(sg[col])
        sc = pd.read_parquet(T / "scenario_per_wb.parquet")
        sc = sc[sc["target"] == tg].set_index("water_body_id")
        a[f"gain_nospill_{tg}"] = a["water_body_id"].map(sc["gain"])
    a.to_parquet(T / "app_rivers.parquet", index=False)
    print("wrote app_rivers.parquet", a.shape)


if __name__ == "__main__":
    t = M.prepare(pd.read_parquet(PROC / "river_features.parquet"))
    g = polys()
    stages = sys.argv[1:] or ["maps", "benchmark", "tree", "forest", "calibration", "shap", "scenarios", "ordinal", "app"]
    if "maps" in stages: maps(t, g)
    if "benchmark" in stages: benchmark()
    if "tree" in stages: tree_and_forest()
    if "forest" in stages: forest_figs()
    if "calibration" in stages: calibration()
    if "shap" in stages: shap_figs(t)
    if "scenarios" in stages: scenario_figs(t, g)
    if "ordinal" in stages: ordinal_fig()
    if "app" in stages or len(sys.argv) == 1: app_table(t, g)
