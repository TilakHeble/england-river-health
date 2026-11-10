"""Tiny helper to write and execute notebooks from Python cell lists."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def md(s):
    return nbformat.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbformat.v4.new_code_cell(s.strip("\n"))


def write_and_run(cells, path: Path, run=True, timeout=1800):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path.parent.mkdir(parents=True, exist_ok=True)
    if run:
        ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)
    print("wrote", path)
