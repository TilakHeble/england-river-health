"""Download every raw input into data/raw/ (about 2.5 GB; ~10 minutes).

All files come from the Defra Data Services Platform (environment.data.gov.uk).

    PYTHONPATH=src python -m river_health.download
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DSP = "https://environment.data.gov.uk/api/file/download?fileDataSetId={id}&fileName={name}"

EDM_ID = "c55e170e-3c75-49a5-8026-a961ff94c8e0"
FILES = {
    # storm overflow Event Duration Monitoring annual returns
    **{f"edm/EDM_{y}.zip": DSP.format(id=EDM_ID, name=f"EDM_{y}_Storm_Overflow_Annual_Return.zip")
       for y in range(2020, 2026)},
    # WFD Cycle 4 (2025) river water body catchments and centrelines
    "wfd/c4_2025_catchments.gpkg.zip": DSP.format(
        id="7f8a213a-41f0-4491-801e-67bfd86f05ba",
        name="WFD_River_Water_Body_Catchments_Cycle_4_Classification_2025_Simplified.gpkg.zip"),
    "wfd/c4_lines.gpkg.zip": DSP.format(
        id="be54355e-0b50-480e-a700-aaf0282484a6",
        name="WFD_River_Canal_and_SWT_Water_Bodies_Cycle_4_Classification_2025_Simplified.gpkg.zip"),
    # Catchment Data Explorer: full classification history and reasons for not achieving good
    "wfd/classifications.csv": "https://environment.data.gov.uk/catchment-planning/England/classifications.csv",
    "wfd/rnags.csv": "https://environment.data.gov.uk/catchment-planning/England/rnags.csv",
    # Natural England Living England Habitat Map Phase 4
    "land/living_england_p4.gpkg.zip": DSP.format(
        id="1e6bc6dd-ee15-4bf4-b913-9ccfdf557646", name="Living_England_Habitat_Map_Phase_4.gpkg.zip"),
    # EA consented discharges register (Access database; EA Conditional Licence)
    "discharges/consents.zip": DSP.format(
        id="a54fdea1-7769-4b22-a518-10d51fed6f33",
        name="Consented%20Discharges%20to%20Controlled%20Waters%20with%20Conditions.zip"),
}
UNZIP = {"edm/EDM_{y}.zip": "edm/EDM_{y}", "wfd/c4_2025_catchments.gpkg.zip": "wfd/c4_2025",
         "wfd/c4_lines.gpkg.zip": "wfd/c4_lines", "land/living_england_p4.gpkg.zip": "land/le",
         "discharges/consents.zip": "discharges/c"}


def fetch(rel: str, url: str):
    path = RAW / rel
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(path)
    print("downloaded", rel, f"{path.stat().st_size / 1e6:.0f} MB")
    return path


def unzip_all():
    for rel, dest in UNZIP.items():
        for y in (range(2020, 2026) if "{y}" in rel else [None]):
            src = RAW / rel.format(y=y)
            out = RAW / dest.format(y=y)
            if src.exists() and not out.exists():
                with zipfile.ZipFile(src) as z:
                    z.extractall(out)


def export_access():
    """The consents register ships as .accdb; export two tables with mdbtools."""
    db = next((RAW / "discharges/c").glob("*.accdb"))
    for table in ["consents_active", "Effluents"]:
        out = RAW / "discharges" / f"{table}.csv"
        if not out.exists():
            with open(out, "w") as f:
                subprocess.run(["mdb-export", str(db), table], stdout=f, check=True)


def download_all():
    for rel, url in FILES.items():
        fetch(rel, url)
    unzip_all()
    export_access()
    from . import wq_download
    for y in (2023, 2024, 2025):
        for det in wq_download.DETERMINANDS:
            wq_download.download(det, y)


if __name__ == "__main__":
    download_all()
