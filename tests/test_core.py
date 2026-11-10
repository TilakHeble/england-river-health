"""Unit tests on synthetic data: grid references, durations, network inference,
spatial folds, monotonic constraints and the spill scenario."""
import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, box

from river_health import modeling as M
from river_health.edm import _duration_hours, ngr_to_en
from river_health.experiments import apply_spill_cut
from river_health.network import _break_cycles, infer_downstream, network_depth, upstream_sets


# ------------------------------------------------------------------ grid references
@pytest.mark.parametrize("ngr,expected", [
    ("SP9037074740", (490370.0, 274740.0)),
    ("TQ3829325157", (538293.0, 125157.0)),   # from the Catchment Data Explorer export
    ("SE1443635492", (414436.0, 435492.0)),
    ("NY 12345 67890", (312345.0, 567890.0)),
    ("TM06748336", (606740.0, 283360.0)),    # 8-digit (10 m) reference
])
def test_ngr_to_en(ngr, expected):
    assert ngr_to_en(ngr) == expected


def test_ngr_rejects_garbage():
    assert all(np.isnan(v) for v in ngr_to_en("not a grid ref"))
    assert all(np.isnan(v) for v in ngr_to_en(None))


# ------------------------------------------------------------------ durations
def test_duration_formats():
    assert _duration_hours(12.5) == 12.5
    assert _duration_hours("125:30:00") == pytest.approx(125.5)
    assert _duration_hours(dt.time(3, 15)) == pytest.approx(3.25)
    assert _duration_hours(dt.timedelta(hours=30)) == pytest.approx(30)
    # Excel stores durations over 24 h as a datetime counted from 1899-12-31
    assert _duration_hours(dt.datetime(1900, 1, 2, 6, 0)) == pytest.approx(54)
    assert np.isnan(_duration_hours(None))


# ------------------------------------------------------------------ network
def _toy_network():
    """Two headwaters A and B join and flow into C, which reaches the sea.

         A (0,10)->(5,5)     B (10,10)->(5,5)
                      \\       /
                    C (5,5)->(5,0)  -> sea
    Lines are digitised downstream, as in the EA data.
    """
    lines = gpd.GeoDataFrame({
        "water_body_id": ["A", "B", "C"], "water_body_type": ["River"] * 3,
        "geometry": [MultiLineString([LineString([(0, 1000), (500, 500)])]),
                     MultiLineString([LineString([(1000, 1000), (500, 500)])]),
                     MultiLineString([LineString([(500, 500), (500, 0)])])]}, crs=27700)
    polys = gpd.GeoDataFrame({
        "water_body_id": ["A", "B", "C", "COAST"],
        "water_body_type": ["River", "River", "River", "Coastal Catchment"],
        "geometry": [box(0, 500, 500, 1000), box(500, 500, 1000, 1000), box(0, 0, 1000, 500),
                     box(0, -500, 1000, -1)]}, crs=27700)
    return lines, polys


def test_infer_downstream_toy():
    lines, polys = _toy_network()
    e = infer_downstream(lines, polys).set_index("water_body_id")
    assert e.loc["A", "downstream_id"] == "C"
    assert e.loc["B", "downstream_id"] == "C"
    assert pd.isna(e.loc["C", "downstream_id"])            # drains to the coastal catchment


def test_upstream_sets_and_depth():
    lines, polys = _toy_network()
    e = infer_downstream(lines, polys)
    up = upstream_sets(e)
    assert up["C"] == {"A", "B", "C"} and up["A"] == {"A"}
    depth = network_depth(e)
    assert depth["A"] == 1 and depth["C"] == 0


def test_cycles_are_broken():
    e = pd.DataFrame({"water_body_id": ["X", "Y", "Z"], "downstream_id": ["Y", "Z", "X"],
                      "rule": ["line"] * 3})
    out = _break_cycles(e)
    assert out["downstream_id"].isna().sum() == 1
    assert (out["rule"] == "cycle_dropped").sum() == 1


# ------------------------------------------------------------------ modelling
def _synthetic(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({f: rng.gamma(1.0, 1.0, n) for f in M.PRESSURE_SET if f not in M.CATEGORICAL})
    for c, levels in [("hmwb", ["natural", "heavily_modified"]), ("altitude", ["low", "mid"]),
                      ("size_class", ["small", "medium"]), ("geology", ["calcareous", "siliceous"])]:
        t[c] = pd.Categorical(rng.choice(levels, n))
    t["management_catchment"] = rng.integers(0, 40, n).astype(str)
    logit = 1.0 - 1.2 * t["up_spill_hours_per_km2"] - 0.8 * t["up_lc_urban"]
    t["y_eco_good"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return t


def test_spatial_folds_keep_catchments_together():
    t = _synthetic()
    X, y, g, _ = M.xy(t, "eco_good", M.PRESSURE_SET)
    folds = M.splits(X, y, g, "spatial")
    assert len(folds) == 5
    for tr, te in folds:
        assert not set(g[tr]) & set(g[te])
    assert sorted(np.concatenate([te for _, te in folds])) == list(range(len(y)))


def test_monotone_lightgbm_never_rewards_more_spills():
    t = _synthetic()
    X, y, _, _ = M.xy(t, "eco_good", M.PRESSURE_SET)
    m = M.make_model("lightgbm", M.PRESSURE_SET, monotone=True).fit(X, y)
    base = m.predict_proba(X)[:, 1]
    for f in ["up_spill_hours_per_km2", "local_spill_count", "up_lc_urban"]:
        more = X.copy()
        more[f] = more[f] * 3 + 1
        assert (m.predict_proba(more)[:, 1] <= base + 1e-9).all(), f


def test_models_recover_the_signal():
    t = _synthetic(2000)
    X, y, g, _ = M.xy(t, "eco_good", M.PRESSURE_SET)
    folds = M.splits(X, y, g, "spatial")
    for name in ["logistic", "random_forest", "lightgbm"]:
        r, _ = M.evaluate(name, M.PRESSURE_SET, X, y, folds)
        assert r["roc_auc"] > 0.7, name


def test_spill_cut_scenario():
    t = _synthetic(50)
    X = t[M.PRESSURE_SET]
    zero = apply_spill_cut(X, 1.0)
    assert (zero[["up_spill_hours_per_km2", "local_spill_hours", "local_n_overflows_60plus"]] == 0).all().all()
    half = apply_spill_cut(X, 0.5)
    assert np.allclose(half["local_spill_hours"], X["local_spill_hours"] / 2)
    # non-spill pressures untouched
    assert (half["up_lc_urban"] == X["up_lc_urban"]).all()
