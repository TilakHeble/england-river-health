"""Load Environment Agency Event Duration Monitoring (EDM) storm overflow annual returns.

Each year's workbook has a slightly different layout (header row, column names,
duration format, Cycle 2 vs Cycle 3 water body IDs). This module standardises
2020-2025 into one tidy table: one row per overflow per year.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "edm"

# canonical name -> regex matched against the flattened header text
COLUMNS = {
    "uid": r"^unique id$",
    "company": r"^water company name",
    "site_name": r"^site name( \(ea consents|$)",
    "permit": r"^(ea permit reference|permit no)",
    "asset_type": r"^storm discharge asset type",
    "ngr": r"^outlet discharge ngr",
    "wb_id": r"^wfd waterbody id",
    "wb_catchment": r"^wfd waterbody catchment name",
    "receiving_water": r"^receiving water",
    "treatment": r"^treatment method",
    "duration_raw": r"^total duration",
    "spills": r"^counted spills",
    "lta_spills": r"^long-term average spill count",
    "edm_pct": r"^(edm operation - )?% of reporting period",
    "shellfish": r"^shellfish",
    "bathing": r"^bathing",
}


def _flat(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def _read_sheet(path: str, sheet: str) -> pd.DataFrame | None:
    head = pd.read_excel(path, sheet_name=sheet, header=None, nrows=15)
    rows = [i for i in range(len(head))
            if any(_flat(v).startswith("water company name") for v in head.iloc[i, :4])]
    if not rows:
        return None
    df = pd.read_excel(path, sheet_name=sheet, header=rows[0])
    out = {}
    for canon, pat in COLUMNS.items():
        hits = [c for c in df.columns if re.search(pat, _flat(c))]
        if hits:
            out[canon] = df[hits[0]]
            if canon == "wb_id":
                out["wb_cycle"] = 3 if "cycle 3" in _flat(hits[0]) else 2
    d = pd.DataFrame(out)
    d = d[d["company"].notna() & (d["company"].astype(str).str.strip() != "")]
    if "edm_pct" in d:
        e = _num(d["edm_pct"])
        if e.max() <= 1.5:  # this sheet stores 0-1 fractions
            e = e * 100
        d["edm_pct"] = e
    return d


def _duration_hours(v) -> float:
    """Durations are numeric hours (2020-23) or hh:mm:ss text/time (2024-25)."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if hasattr(v, "hour") and not hasattr(v, "year"):  # datetime.time
        return v.hour + v.minute / 60 + v.second / 3600
    if hasattr(v, "days"):  # timedelta
        return v.total_seconds() / 3600
    if hasattr(v, "year"):  # Excel stored >24h as datetime from 1900-01-00
        base = pd.Timestamp("1899-12-31")
        return (pd.Timestamp(v) - base).total_seconds() / 3600
    s = str(v).strip()
    m = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2})(?:\.\d+)?", s)
    if m:
        h, mi, se = map(int, m.groups())
        return h + mi / 60 + se / 3600
    try:
        return float(s)
    except ValueError:
        return np.nan


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("%", "").str.strip(), errors="coerce")


def load_year(year: int) -> pd.DataFrame:
    files = glob.glob(str(RAW / f"EDM_{year}" / "**" / "*.xlsx"), recursive=True)
    files = [f for f in files if "summary" not in f.lower()]
    frames = []
    for f in files:
        xl = pd.ExcelFile(f)
        sheets = [s for s in xl.sheet_names if s.lower() != "all wasc"]
        for s in sheets:
            d = _read_sheet(f, s)
            if d is not None and len(d):
                frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = year
    df["duration_hrs"] = df["duration_raw"].map(_duration_hours)
    df = df.drop(columns="duration_raw")
    df["spills"] = _num(df["spills"])
    if "lta_spills" in df:
        df["lta_spills"] = _num(df["lta_spills"])
    for c in ("wb_id", "ngr", "uid", "lta_spills", "wb_cycle"):
        if c not in df:
            df[c] = np.nan
    df["wb_id"] = df["wb_id"].astype(str).str.strip().str.upper().where(df["wb_id"].notna())
    df["ngr"] = df["ngr"].astype(str).str.replace(" ", "").str.upper().where(df["ngr"].notna())
    df["company"] = df["company"].astype(str).str.strip()
    for c in ("uid", "site_name", "permit", "asset_type", "wb_catchment",
              "receiving_water", "treatment", "shellfish", "bathing"):
        if c in df:
            df[c] = df[c].astype(str).str.strip().where(df[c].notna())
    return df


def load_all(years=range(2020, 2026)) -> pd.DataFrame:
    return pd.concat([load_year(y) for y in years], ignore_index=True)


# ---------------------------------------------------------------- grid refs
_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def ngr_to_en(ngr: str):
    """Convert an OS National Grid reference (e.g. SP9037074740) to easting/northing (m)."""
    if not isinstance(ngr, str):
        return (np.nan, np.nan)
    s = re.sub(r"[^A-Z0-9]", "", ngr.upper())
    m = re.fullmatch(r"([A-Z]{2})(\d+)", s)
    if not m or len(m.group(2)) % 2:
        return (np.nan, np.nan)
    l1, l2 = (_LETTERS.index(ch) for ch in m.group(1))
    e100 = ((l1 - 2) % 5) * 5 + (l2 % 5)
    n100 = (19 - (l1 // 5) * 5) - (l2 // 5)
    if not (0 <= e100 <= 12 and 0 <= n100 <= 24):
        return (np.nan, np.nan)
    digits = m.group(2)
    k = len(digits) // 2
    scale = 10 ** (5 - k)
    e = e100 * 100_000 + int(digits[:k]) * scale
    n = n100 * 100_000 + int(digits[k:]) * scale
    return (float(e), float(n))
