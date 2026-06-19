"""Helper to execute a notebook from within another notebook (report assembly).

Mirrors the reference project's `run_notebook` pattern: execute a notebook in
place with nbconvert so the assembled report can chain the analysis notebooks.
"""
from __future__ import annotations

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(path: str, timeout: int = 1800) -> nbformat.NotebookNode:
    """Execute the notebook at `path` in its own directory and return it."""
    nb = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    import os

    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(path) or "."}})
    return nb
