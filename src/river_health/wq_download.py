"""Download river chemistry from the Environment Agency Water Quality API (2025 version).

Pulls all freshwater-river (type F6) observations in England for a set of
determinands, month by month, paginating 2,500 rows at a time. Output: one CSV
per determinand per year in data/raw/wq/.

    python -m river_health.wq_download 2023 2025
"""
from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://environment.data.gov.uk/water-quality/data/observation"
ENGLAND = {"type": "Polygon",
           "coordinates": [[[-6.5, 49.8], [2.1, 49.8], [2.1, 55.9], [-6.5, 55.9], [-6.5, 49.8]]]}
DETERMINANDS = {
    "0180": "orthophosphate_mgP",   # Orthophosphate, reactive as P
    "0111": "ammonia_mgN",          # Ammoniacal nitrogen as N
    "0085": "bod_mg",               # BOD 5-day ATU
    "0117": "nitrate_mgN",          # Nitrate as N
    "9901": "do_pct_sat",           # Dissolved oxygen % saturation
}
KEEP = ["samplingPoint.notation", "samplingPoint.prefLabel", "samplingPoint.easting",
        "samplingPoint.northing", "phenomenonTime", "samplingPurpose",
        "determinand.notation", "result", "unit"]
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "wq"
PAGE = 2500


def _get(params, tries=6):
    for k in range(tries):
        try:
            r = requests.post(API, params=params, json=ENGLAND, timeout=180,
                              headers={"Accept": "text/csv"})
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (k + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed after {tries} tries: {params}")


def month_frames(det: str, start: str, end: str):
    skip = 0
    while True:
        txt = _get({"determinand": det, "samplingPointType": "F6", "dateFrom": start,
                    "dateTo": end, "limit": PAGE, "skip": skip})
        if not txt.strip() or txt.count("\n") < 1:
            return
        df = pd.read_csv(io.StringIO(txt), dtype=str)
        if df.empty:
            return
        yield df[[c for c in KEEP if c in df.columns]]
        if len(df) < PAGE:
            return
        skip += PAGE


def download(det: str, year: int):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{DETERMINANDS[det]}_{year}.csv"
    if path.exists():
        return path
    frames = []
    for m in range(1, 13):
        start = f"{year}-{m:02d}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        frames += list(month_frames(det, start, end))
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(path)
    print(f"{path.name}: {len(df):,} rows", flush=True)
    return path


if __name__ == "__main__":
    y0, y1 = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (2023, 2025)
    jobs = [(d, y) for y in range(y0, y1 + 1) for d in DETERMINANDS]
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda a: download(*a), jobs))
