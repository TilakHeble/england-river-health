"""Build the modelling table: one row per English river water body (WFD Cycle 4).

Every pressure is measured twice:
* ``local_*``  inside the water body's own catchment polygon;
* ``up_*``     summed over the water body and everything upstream of it
               (the inferred river network, see ``network.py``).

Sources (all Environment Agency / Natural England, Open Government Licence
unless noted):
* WFD Cycle 4 2025 classification + catchment polygons  -> targets, typology
* EDM storm overflow annual returns 2021-2025            -> spill pressure
* Consented discharges register (EA Conditional Licence) -> sewage works size
* Living England Habitat Map Phase 4 (Natural England)   -> land cover
* Water Quality Archive 2023-2025                         -> river chemistry
* Reasons for Not Achieving Good (Cycle 3)               -> validation only

    PYTHONPATH=src python -m river_health.features
"""
from __future__ import annotations

import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .edm import ngr_to_en
from .network import infer_downstream, network_depth, upstream_sets

ROOT = Path(__file__).resolve().parents[2]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

CLASS_ORDER = ["Bad", "Poor", "Moderate", "Good", "High"]
CLASS_CODE = {c: i for i, c in enumerate(CLASS_ORDER)}
SPILL_YEARS = [2021, 2022, 2023, 2024]      # precede the 2025 classification
CHEM = ["orthophosphate_mgP", "ammonia_mgN", "bod_mg", "nitrate_mgN", "do_pct_sat"]


# ----------------------------------------------------------------- helpers
def _points(df: pd.DataFrame, ngr_col: str) -> gpd.GeoDataFrame:
    en = df[ngr_col].map(ngr_to_en)
    df = df.assign(e=[a for a, _ in en], n=[b for _, b in en]).dropna(subset=["e", "n"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["e"], df["n"]), crs=27700)


def _locate(pts: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> pd.Series:
    j = gpd.sjoin(pts, polys[["water_body_id", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    return j["water_body_id"]


def _upstream_sum(local: pd.DataFrame, ups: dict[str, set[str]]) -> pd.DataFrame:
    local = local.fillna(0)
    rows = {w: local.reindex(list(s)).sum() for w, s in ups.items()}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------- pieces
def load_catchments() -> gpd.GeoDataFrame:
    g = gpd.read_file(glob.glob(str(RAW / "wfd/c4_2025/*.gpkg"))[0])
    return g


def base_table(g: gpd.GeoDataFrame) -> pd.DataFrame:
    r = g[g["water_body_type"] == "River"].copy()
    typ = r["water_body_typology"].fillna("Unknown, Unknown, Unknown").str.split(", ", expand=True)
    c = r.geometry.centroid
    t = pd.DataFrame({
        "water_body_id": r["water_body_id"],
        "name": r["water_body_name"],
        "rbd": r["river_basin_district"],
        "management_catchment": r["management_catchment"],
        "operational_catchment": r["operational_catchment"],
        "hmwb": r["water_body_hmd"].map({"Heavily Modified": "heavily_modified",
                                         "Artificial": "artificial"}).fillna("natural"),
        "altitude": typ[0].str.lower(), "size_class": typ[1].str.lower(),
        "geology": typ[2].str.lower(),
        "local_area_km2": r.geometry.area / 1e6,
        "easting": c.x, "northing": c.y,
        "surveillance_site": r["surveillance_water_body"].eq("Yes").astype(int),
    })
    for col, name in [("ecological_class", "eco"), ("invertebrates_class", "inv"),
                      ("fish_class", "fish"), ("phosphate_class", "phos"),
                      ("macrophytes_phytobenthos_combined_class", "macro"),
                      ("ammonia_phys_chem_class", "amm"), ("dissolved_oxygen_class", "do")]:
        t[f"y_{name}"] = r[col].where(r[col].isin(CLASS_ORDER))
    t["eco_certainty"] = r["eco_certainty_less_than_good"]
    t["eco_change_vs_2022"] = r["class_change_eco"]
    return t.reset_index(drop=True)


def spill_features(polys: gpd.GeoDataFrame) -> pd.DataFrame:
    edm = pd.read_parquet(PROC / "edm_2020_2025.parquet")
    edm = edm[edm["year"].isin(SPILL_YEARS + [2025])]
    pts = _points(edm, "ngr")
    edm.loc[pts.index, "wb4"] = _locate(pts, polys)
    # fall back on the reported WFD ID where the grid reference is missing
    river_ids = set(polys["water_body_id"])
    miss = edm["wb4"].isna() & edm["wb_id"].isin(river_ids)
    edm.loc[miss, "wb4"] = edm.loc[miss, "wb_id"]
    edm = edm.dropna(subset=["wb4"])
    edm["at_stw"] = edm["asset_type"].fillna("").str.contains(
        "treatment|wwtw|stw", case=False).astype(int)

    win = edm[edm["year"].isin(SPILL_YEARS)]
    per_year = win.groupby(["wb4", "year"]).agg(
        hours=("duration_hrs", "sum"), spills=("spills", "sum"),
        hours_stw=("duration_hrs", lambda s: s[win.loc[s.index, "at_stw"] == 1].sum()))
    f = per_year.groupby("wb4").mean().rename(columns={
        "hours": "spill_hours", "spills": "spill_count", "hours_stw": "spill_hours_stw"})
    latest = edm[edm["year"] == 2024]
    f["n_overflows"] = latest.groupby("wb4").size()
    by_site = win.groupby(["wb4", "permit", "site_name"], dropna=False)["spills"].mean()
    f["n_overflows_60plus"] = (by_site >= 60).groupby("wb4").sum()
    f["edm_low_coverage_share"] = latest.assign(low=latest["edm_pct"] < 90).groupby("wb4")["low"].mean()
    f["spill_hours_2025"] = edm[edm["year"] == 2025].groupby("wb4")["duration_hrs"].sum()
    f.index.name = "water_body_id"
    return f


def sewage_works_features(polys: gpd.GeoDataFrame) -> pd.DataFrame:
    c = pd.read_csv(RAW / "discharges/consents_active.csv", low_memory=False)
    e = pd.read_csv(RAW / "discharges/Effluents.csv", low_memory=False)
    e = e.rename(columns={"PERMIT_REF": "PERMIT_NUMBER", "VERSION": "PERMIT_VERSION",
                          "DWF (m3_d)": "dwf"})
    key = ["PERMIT_NUMBER", "PERMIT_VERSION", "OUTLET_NUMBER", "EFFLUENT_NUMBER"]
    c = c.merge(e[key + ["dwf"]].drop_duplicates(key), on=key, how="left")
    c = c[c["REC_ENV_CODE_DESCRIPTION"].isin(["Freshwater river", "Canal", "Pond/Lake/Reservoir"])
          | (c["EFFLUENT_TYPE"] == "UA")]
    pts = _points(c, "EFFLUENT_GRID_REF")
    pts["wb"] = _locate(pts, polys)
    pts = pts.dropna(subset=["wb"])
    stw = pts[pts["EFFLUENT_TYPE"] == "SA"].drop_duplicates(["PERMIT_NUMBER", "OUTLET_NUMBER"])
    f = pd.DataFrame({
        "n_stw": stw.groupby("wb").size(),
        "stw_dwf_m3d": stw.groupby("wb")["dwf"].sum(),
        "n_private_sewage": pts[pts["EFFLUENT_TYPE"] == "UA"].groupby("wb").size(),
        "n_trade_effluent": pts[pts["EFFLUENT_TYPE"].isin(["TF", "TA"])].groupby("wb").size(),
        "n_fish_farms": pts[pts["EFFLUENT_TYPE"] == "AF"].groupby("wb").size(),
    })
    f.index.name = "water_body_id"
    return f


def chemistry_features(polys: gpd.GeoDataFrame) -> pd.DataFrame:
    out = {}
    for det in CHEM:
        frames = [pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(RAW / f"wq/{det}_*.csv")))]
        w = pd.concat(frames, ignore_index=True)
        w = w[~w["samplingPurpose"].str.contains("POLLUTION INCIDENT", na=False)]
        r = w["result"].str.strip()
        val = pd.to_numeric(r.str.lstrip("<>"), errors="coerce")
        val = np.where(r.str.startswith("<"), val / 2, val)  # below detection: half the limit
        w["v"] = val
        site = w.groupby("samplingPoint.notation").agg(
            v=("v", "median"), e=("samplingPoint.easting", "first"),
            n=("samplingPoint.northing", "first"), k=("v", "size"))
        site = site[site["k"] >= 6]  # at least six samples in three years
        g = gpd.GeoDataFrame(site, geometry=gpd.points_from_xy(site["e"].astype(float),
                                                              site["n"].astype(float)), crs=27700)
        g["wb"] = _locate(g, polys)
        out[f"chem_{det}"] = g.groupby("wb")["v"].median()
        if det == CHEM[0]:
            out["chem_n_sites"] = g.groupby("wb").size()
    f = pd.DataFrame(out)
    f.index.name = "water_body_id"
    return f


def rnag_flags() -> pd.DataFrame:
    g = pd.read_csv(RAW / "wfd/rnags.csv", low_memory=False)
    g = g[(g["Water Body Type"] == "River") & (g["Reason Type"] == "RNAG")]
    eco_elements = ["Phosphate", "Macrophytes and Phytobenthos Combined", "Fish", "Invertebrates",
                    "Dissolved oxygen", "Ammonia (Phys-Chem)", "Mitigation Measures Assessment",
                    "Hydrological Regime", "Biochemical Oxygen Demand (BOD)", "pH", "Temperature"]
    g = g[g["Classification Element"].isin(eco_elements)]
    sw = g["Business Sector"].eq("Waste water treatment") | g["Activity"].fillna("").str.contains(
        "Sewage|Storm overflow|Continuous sewage|Intermittent sewage", case=False)
    flags = pd.DataFrame({
        "rnag_sewage": sw, "rnag_overflow": g["Activity"].fillna("").str.contains(
            "storm overflow|Intermittent", case=False),
        "rnag_agriculture": g["Category"].eq("Agriculture and rural land management"),
        "rnag_urban": g["Category"].eq("Urban and transport"),
        "rnag_physical": g["National Swmi Header"].fillna("").str.contains(
            "Physical modification", case=False),
        "water_body_id": g["Water Body ID"]})
    return flags.groupby("water_body_id").max().astype(int)


# ----------------------------------------------------------------- assemble
def build() -> pd.DataFrame:
    g = load_catchments()
    polys = g[g["water_body_type"] == "River"].reset_index(drop=True)
    t = base_table(g)

    lines = gpd.read_file(glob.glob(str(RAW / "wfd/c4_lines/*.gpkg"))[0])
    edges = infer_downstream(lines, g)
    edges.to_csv(PROC / "network_edges.csv", index=False)
    ups = upstream_sets(edges)
    t["links_to_sea"] = t["water_body_id"].map(network_depth(edges))
    t["n_upstream_wbs"] = t["water_body_id"].map({k: len(v) for k, v in ups.items()})
    t["drains_to_sea"] = t["water_body_id"].map(dict(zip(edges["water_body_id"],
                                                         edges["downstream_id"].isna()))).astype(int)

    area = t.set_index("water_body_id")[["local_area_km2"]]
    t["up_area_km2"] = t["water_body_id"].map(_upstream_sum(area, ups)["local_area_km2"])

    # pressures: local and upstream totals
    press = pd.concat([spill_features(polys), sewage_works_features(polys)], axis=1)
    press = press.reindex(t["water_body_id"])
    additive = [c for c in press.columns if c not in ("edm_low_coverage_share",)]
    up = _upstream_sum(press[additive], ups)
    for c in press.columns:
        t[f"local_{c}"] = t["water_body_id"].map(press[c]).fillna(0)
    for c in additive:
        t[f"up_{c}"] = t["water_body_id"].map(up[c])
    # intensity: pressure per km2 of upstream catchment (a dilution proxy)
    for c in ["spill_hours", "spill_count", "stw_dwf_m3d", "n_private_sewage"]:
        t[f"up_{c}_per_km2"] = t[f"up_{c}"] / t["up_area_km2"]
        t[f"local_{c}_per_km2"] = t[f"local_{c}"] / t["local_area_km2"]

    # land cover: local shares and upstream area-weighted shares
    lc = pd.read_csv(PROC / "landcover_local.csv").set_index("water_body_id")
    lc_cols = [c for c in lc.columns if c.startswith("lc_") and c != "lc_points"]
    t = t.merge(lc[lc_cols].add_prefix("local_"), left_on="water_body_id",
                right_index=True, how="left")
    weighted = lc[lc_cols].mul(area["local_area_km2"].reindex(lc.index), axis=0)
    upw = _upstream_sum(weighted, ups).div(t.set_index("water_body_id")["up_area_km2"], axis=0)
    t = t.merge(upw.add_prefix("up_"), left_on="water_body_id", right_index=True, how="left")

    t = t.merge(chemistry_features(polys), left_on="water_body_id", right_index=True, how="left")
    t = t.merge(rnag_flags(), left_on="water_body_id", right_index=True, how="left")
    for c in [c for c in t.columns if c.startswith("rnag_")]:
        t[c] = t[c].fillna(0).astype(int)

    for name in ["eco", "inv", "fish", "phos", "macro", "amm", "do"]:
        t[f"y_{name}_code"] = t[f"y_{name}"].map(CLASS_CODE)
    t["y_eco_good"] = (t["y_eco_code"] >= 3).astype(float).where(t["y_eco_code"].notna())
    t["y_inv_good"] = (t["y_inv_code"] >= 3).astype(float).where(t["y_inv_code"].notna())
    return t


if __name__ == "__main__":
    t = build()
    t.to_parquet(PROC / "river_features.parquet", index=False)
    print(t.shape)
    print(t.describe().T.to_string())
