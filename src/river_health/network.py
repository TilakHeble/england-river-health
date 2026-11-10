"""Infer the downstream link of every WFD river water body in England.

The EA publishes river centrelines and catchment polygons but no
"downstream water body" field. We infer it in two steps:

1. Line rule. EA centrelines are digitised in the direction of flow, so the
   last vertex of each line part is a potential outlet. If one touches another
   river water body's line (within ``TOL`` m) at a point that is not also one of
   that line's ends, that water body is downstream.
2. Polygon rule. Where the simplified geometry leaves a gap, the downstream
   water body is the neighbouring catchment polygon nearest the outlet (within
   ``POLY_TOL`` m). An outlet whose nearest neighbour is a coastal catchment, or
   with no neighbour, drains to the sea/estuary (downstream = None).

The result is checked for cycles; any edge on a cycle is dropped. Upstream
aggregation (``upstream_sets``) then walks the tree.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import linemerge

TOL = 25.0
POLY_TOL = 250.0


def _parts(geom):
    m = linemerge(geom) if geom.geom_type == "MultiLineString" else geom
    return list(m.geoms) if m.geom_type == "MultiLineString" else [m]


def infer_downstream(lines: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> pd.DataFrame:
    lines = lines[lines["water_body_type"] == "River"].reset_index(drop=True)
    ids = lines["water_body_id"].to_numpy()
    # every part's last vertex is a potential outlet (lines are digitised downstream)
    ends = [[Point(p.coords[-1]) for p in _parts(g)] for g in lines.geometry]
    sidx = lines.sindex

    down, rule = [None] * len(lines), [None] * len(lines)
    for i in range(len(lines)):
        votes = Counter()
        for o in ends[i]:
            for j in sidx.query(o.buffer(TOL)):
                if j == i or lines.geometry[j].distance(o) >= TOL:
                    continue
                # a point that is also one of j's ends is a confluence of two inflows, not j's channel
                if any(e.distance(o) < TOL for e in ends[j]):
                    continue
                votes[j] += 1
        if votes:
            j = votes.most_common(1)[0][0]
            down[i], rule[i] = ids[j], "line"
    outlets = [min(e, key=lambda p: p.y) for e in ends]  # used only for the fallback

    # polygon fallback
    pol = polys.reset_index(drop=True)
    pidx = pol.sindex
    own = dict(zip(pol["water_body_id"], range(len(pol))))
    for i, o in enumerate(outlets):
        if down[i] is not None:
            continue
        near = [k for k in pidx.query(o.buffer(POLY_TOL))
                if pol["water_body_id"][k] != ids[i]]
        if not near:
            rule[i] = "sea"
            continue
        k = min(near, key=lambda k: pol.geometry[k].distance(o))
        if pol["water_body_type"][k] == "River" and pol.geometry[k].distance(o) < POLY_TOL:
            down[i], rule[i] = pol["water_body_id"][k], "polygon"
        else:
            rule[i] = "sea"

    edges = pd.DataFrame({"water_body_id": ids, "downstream_id": down, "rule": rule})
    return _break_cycles(edges)


def _break_cycles(edges: pd.DataFrame) -> pd.DataFrame:
    """Drop exactly one edge per cycle (the edge that closes it on first discovery)."""
    d = dict(zip(edges["water_body_id"], edges["downstream_id"]))
    d = {k: (v if isinstance(v, str) else None) for k, v in d.items()}
    bad = set()
    for start in list(d):
        seen, x = [], start
        while x is not None and x not in seen:
            seen.append(x)
            x = d.get(x)
        if x is not None:          # walked back into `seen`: a cycle
            closer = seen[-1]      # its outgoing edge closes the loop
            d[closer] = None       # remove now so other starts see a tree
            bad.add(closer)
    edges = edges.copy()
    edges["downstream_id"] = edges["water_body_id"].map(d)
    edges.loc[edges["water_body_id"].isin(bad), "rule"] = "cycle_dropped"
    return edges


def upstream_sets(edges: pd.DataFrame) -> dict[str, set[str]]:
    """Map each water body to the set of all water bodies upstream of it (inclusive)."""
    kids = defaultdict(list)
    for a, b in zip(edges["water_body_id"], edges["downstream_id"]):
        if isinstance(b, str):
            kids[b].append(a)
    memo: dict[str, set[str]] = {}

    def up(x):
        if x in memo:
            return memo[x]
        stack, out = [x], set()
        while stack:
            y = stack.pop()
            if y in out:
                continue
            out.add(y)
            stack.extend(kids.get(y, []))
        memo[x] = out
        return out

    return {w: up(w) for w in edges["water_body_id"]}


def network_depth(edges: pd.DataFrame) -> pd.Series:
    """Number of links from each water body to the sea (0 = drains to sea)."""
    d = dict(zip(edges["water_body_id"], edges["downstream_id"]))
    out = {}
    for w in d:
        n, x = 0, d[w]
        while isinstance(x, str) and n < 500:
            n, x = n + 1, d.get(x)
        out[w] = n
    return pd.Series(out, name="links_to_sea")
