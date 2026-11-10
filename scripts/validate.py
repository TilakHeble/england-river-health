"""Independent validation checks on the real data (outputs/tables/validation_checks.csv).

    python scripts/validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from river_health import modeling as M  # noqa: E402
from river_health.experiments import apply_spill_cut  # noqa: E402

T = ROOT / "outputs" / "tables"
PROC = ROOT / "data" / "processed"

# Environment Agency published national totals (EDM annual return press releases / summary data)
EA_TOTALS = {2023: (464_056, 3_606_170), 2024: (450_398, 3_614_428), 2025: (292_119, 1_878_240)}
# National River Flow Archive gauged catchment areas (km2) and the water body just upstream of the gauge
NRFA = {"Trent at North Muskham (28022)": ("Trent from Soar to The Beck", 8231),
        "Tyne at Bywell (23001)": ("Tyne from Watersmeet to Tidal Limit", 2176),
        "Thames at Kingston (39001)": ("Thames (Egham to Teddington)", 9948),
        "Eden at Sheepmount (76007)": ("Eden - Eamont to tidal", 2287),
        "Nene at Orton (32001)": ("Nene - Islip to tidal", 1634)}
# (Severn and Wye gauges are omitted: their headwaters are in Wales, outside the EA data.)


def check_edm_totals(edm):
    rows = []
    for y, (spills, hours) in EA_TOTALS.items():
        d = edm[edm["year"] == y]
        s, h = d["spills"].sum(), d["duration_hrs"].sum()
        ok = abs(s / spills - 1) < 0.005 and abs(h / hours - 1) < 0.005
        rows.append(dict(check=f"EDM {y} totals match EA published figures", passed=bool(ok),
                         detail=f"spills {s:,.0f} vs {spills:,}; hours {h:,.0f} vs {hours:,}"))
    return rows


def check_network(t):
    rows, worst = [], 0
    detail = []
    for gauge, (wb, area) in NRFA.items():
        m = t[t["name"] == wb]
        if m.empty:
            detail.append(f"{gauge}: water body not found")
            continue
        r = m["up_area_km2"].iloc[0] / area
        worst = max(worst, abs(1 - r))
        detail.append(f"{gauge.split(' (')[0]} {m['up_area_km2'].iloc[0]:,.0f}/{area:,} km2")
    rows.append(dict(check="inferred network: upstream areas within 25% of NRFA gauged areas "
                           "(England-only data; Welsh headwaters missing)",
                     passed=bool(worst < 0.25), detail="; ".join(detail)))
    e = pd.read_csv(PROC / "network_edges.csv")
    d = dict(zip(e["water_body_id"], e["downstream_id"]))
    cyc = 0
    for w in d:
        seen, x = set(), w
        while isinstance(x, str) and x not in seen:
            seen.add(x)
            x = d.get(x)
        cyc += isinstance(x, str)
    rows.append(dict(check="river network is acyclic", passed=cyc == 0,
                     detail=f"{len(e)} links: {e['rule'].value_counts().to_dict()}"))
    return rows


def check_linkage(t, edm):
    e24 = edm[edm["year"] == 2024]
    linked = t["local_n_overflows"].sum()
    share = linked / len(e24)
    rows = [dict(check="storm overflows linked to a river water body (rest: coastal/estuary or Wales)",
                 passed=bool(share > 0.75), detail=f"{linked:,.0f} of {len(e24):,} 2024 overflows ({share:.0%})")]
    rho = t[["chem_orthophosphate_mgP", "y_phos_code"]].corr(method="spearman").iloc[0, 1]
    rows.append(dict(check="chemistry linkage: measured orthophosphate vs EA phosphate class (Spearman < -0.6)",
                     passed=bool(rho < -0.6), detail=f"rho = {rho:.2f} over {t['chem_orthophosphate_mgP'].notna().sum()} water bodies"))
    lc = pd.read_csv(PROC / "landcover_local.csv")
    lc = lc.set_index("water_body_id").reindex(t["water_body_id"])
    pts = lc["lc_points"]
    unm = (lc["lc_unmapped"].fillna(1) * pts).sum() / pts.sum()
    rows.append(dict(check="land cover sample: every catchment sampled (>= 8 points) and < 2% of points unmapped",
                     passed=bool(pts.min() >= 8 and unm < 0.02),
                     detail=f"points per catchment: min {pts.min():.0f}, median {pts.median():.0f}; unmapped {unm:.1%}"))
    rows.append(dict(check="one row per WFD Cycle 4 river water body", passed=bool(t["water_body_id"].is_unique and len(t) == 3772),
                     detail=f"{len(t)} rows"))
    return rows


def check_folds_and_leakage(t):
    rows = []
    feats = M.FEATURE_SETS["pressures"]
    X, y, g, _ = M.xy(t, "phos_good", feats)
    folds = M.splits(X, y, g, "spatial")
    split = sum(bool(set(g[tr]) & set(g[te])) for tr, te in folds)
    rows.append(dict(check="spatial folds: no management catchment in both train and test", passed=split == 0,
                     detail=f"{len(set(g))} catchments, 5 folds"))
    # corrupt fold-0 test labels; out-of-fold predictions for fold 0 must not change
    p1 = M.oof_predict("lightgbm", feats, X, y, folds[:1], monotone=True)
    y2 = y.copy()
    y2[folds[0][1]] = 1 - y2[folds[0][1]]
    p2 = M.oof_predict("lightgbm", feats, X, y2, folds[:1], monotone=True)
    diff = float(np.abs(p1[folds[0][1]] - p2[folds[0][1]]).max())
    rows.append(dict(check="test labels never used (flipped fold-0 labels)", passed=diff < 1e-12,
                     detail=f"max change in fold-0 predictions {diff:.1e}"))
    return rows


def check_champions(t):
    rows = []
    for target in ["eco_good", "inv_good", "phos_good"]:
        b = joblib.load(ROOT / "outputs/models" / f"lgbm_{target}.joblib")
        X = t[b["features"]]
        base = b["model"].predict_proba(X)[:, 1]
        worse = 0
        for f in ["up_spill_hours_per_km2", "local_spill_hours", "up_stw_dwf_m3d_per_km2", "up_lc_urban"]:
            Z = X.copy()
            Z[f] = Z[f] * 2 + 1
            worse += int((b["model"].predict_proba(Z)[:, 1] > base + 1e-9).sum())
        cut = b["model"].predict_proba(apply_spill_cut(X, 1.0))[:, 1]
        rows.append(dict(check=f"{target}: more pollution never raises P(good); removing spills never lowers it",
                         passed=bool(worse == 0 and (cut >= base - 1e-9).all()),
                         detail=f"violations: {worse}; min change on removing spills {np.min(cut - base):.2e}"))
    oof = pd.read_parquet(T / "oof_phos_good.parquet")
    q = pd.qcut(oof["p_lightgbm"], 10, labels=False, duplicates="drop")
    cal = oof.groupby(q).agg(p=("p_lightgbm", "mean"), y=("y", "mean"))
    gap = float((cal["p"] - cal["y"]).abs().max())
    rows.append(dict(check="phosphate champion calibrated out of fold (max decile gap < 0.10)", passed=gap < 0.10,
                     detail=f"max |predicted - observed| across deciles {gap:.3f}"))
    return rows


def main():
    t = M.prepare(pd.read_parquet(PROC / "river_features.parquet"))
    edm = pd.read_parquet(PROC / "edm_2020_2025.parquet")
    rows = (check_edm_totals(edm) + check_network(t) + check_linkage(t, edm)
            + check_folds_and_leakage(t) + check_champions(t))
    out = pd.DataFrame(rows)
    out.to_csv(T / "validation_checks.csv", index=False)
    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        print(out)
    print(f"{out['passed'].sum()}/{len(out)} checks passed")


if __name__ == "__main__":
    main()
