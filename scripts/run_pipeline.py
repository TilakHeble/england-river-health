"""End-to-end pipeline.

    python scripts/run_pipeline.py --data          # download (~2.5 GB) and build data/processed (~15 min)
    python scripts/run_pipeline.py --experiments   # every model, tuning, SHAP, scenarios (~1 h on 2 cores)
    python scripts/run_pipeline.py --reports       # figures and validation checks
    python scripts/run_pipeline.py --notebooks     # execute notebooks/ and learn/
    python scripts/run_pipeline.py --all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, env=env)


def main():
    ap = argparse.ArgumentParser()
    for flag in ["data", "experiments", "reports", "notebooks", "all"]:
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    if a.data or a.all:
        from river_health import download, edm, features, landcover
        download.download_all()
        proc = ROOT / "data" / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        edm.load_all().to_parquet(proc / "edm_2020_2025.parquet")
        run("-m", "river_health.landcover")
        t = features.build()
        t.to_parquet(proc / "river_features.parquet", index=False)
    if a.experiments or a.all:
        run("-m", "river_health.experiments")
    if a.reports or a.all:
        run("scripts/make_figures.py")
        run("scripts/validate.py")
    if a.notebooks or a.all:
        run("scripts/build_notebooks.py")
        run("scripts/build_learn.py")


if __name__ == "__main__":
    main()
