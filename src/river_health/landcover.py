"""Habitat composition of each river water body catchment from Living England (Phase 4).

The habitat map has 6.9 million polygons, too many to intersect directly. We
sample it on a regular 250 m point grid (one point per 6.25 ha, ~2 million points
for England), read the map tile by tile using the GeoPackage spatial index, and
record the predicted habitat at each point. Catchment shares are then the share
of points in each class.

    PYTHONPATH=src python -m river_health.landcover
"""
from __future__ import annotations

import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parents[2]
LE = ROOT / "data/raw/land/le/Living_England_Habitat_Map_Phase_4.gpkg"
SPACING = 250
TILE = 50_000

# collapse 16 habitat classes into groups that matter for river pressures
GROUPS = {
    "Arable and Horticultural": "arable",
    "Improved Grassland": "improved_grass",
    "Built-up Areas and Gardens": "urban",
    "Broadleaved, Mixed and Yew Woodland": "woodland",
    "Coniferous Woodland": "woodland",
    "Acid, Calcareous, Neutral Grassland": "seminatural",
    "Dwarf Shrub Heath": "seminatural",
    "Bracken": "seminatural",
    "Scrub": "seminatural",
    "Bog": "wetland_bog",
    "Fen, Marsh and Swamp": "wetland_bog",
    "Water": "water",
    "Bare Ground": "other",
    "Bare Sand": "other",
    "Coastal Saltmarsh": "other",
    "Coastal Sand Dunes": "other",
}


def catchment_landcover(catchments: gpd.GeoDataFrame) -> pd.DataFrame:
    rivers = catchments[catchments["water_body_type"] == "River"][["water_body_id", "geometry"]]
    x0, y0, x1, y1 = rivers.total_bounds
    out = []
    for tx in np.arange(np.floor(x0 / TILE) * TILE, x1, TILE):
        for ty in np.arange(np.floor(y0 / TILE) * TILE, y1, TILE):
            xs = np.arange(tx + SPACING / 2, tx + TILE, SPACING)
            ys = np.arange(ty + SPACING / 2, ty + TILE, SPACING)
            gx, gy = np.meshgrid(xs, ys)
            pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()),
                                   crs=27700)
            pts = gpd.sjoin(pts, rivers, predicate="within", how="inner").drop(columns="index_right")
            if pts.empty:
                continue
            hab = pyogrio.read_dataframe(LE, bbox=(tx, ty, tx + TILE, ty + TILE),
                                         columns=["a_pred"])
            if hab.empty:
                continue
            j = gpd.sjoin(pts, hab, predicate="within", how="left")
            j = j[~j.index.duplicated()]
            out.append(j[["water_body_id", "a_pred"]])
            print(f"tile {tx:.0f},{ty:.0f}: {len(j):,} points", flush=True)
    pts = pd.concat(out, ignore_index=True)
    pts["group"] = pts["a_pred"].map(GROUPS).fillna("unmapped")
    shares = pd.crosstab(pts["water_body_id"], pts["group"], normalize="index")
    shares.columns = [f"lc_{c}" for c in shares.columns]
    shares["lc_points"] = pts.groupby("water_body_id").size()
    return shares.reset_index()


if __name__ == "__main__":
    polys = gpd.read_file(glob.glob(str(ROOT / "data/raw/wfd/c4_2025/*.gpkg"))[0])
    lc = catchment_landcover(polys)
    lc.to_csv(ROOT / "data/processed/landcover_local.csv", index=False)
    print(lc.describe().T)
