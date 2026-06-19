"""Small display helpers for the notebooks.

Mirrors the reference project's `utils/display.py`: a thin wrapper around
IPython display so that DataFrames and saved figures render consistently and
with a short, readable API across all notebooks.
"""
from __future__ import annotations

import pandas as pd
from IPython.display import Image, display


def display_df(df: pd.DataFrame, max_rows: int = 20, caption: str | None = None) -> None:
    """Render a DataFrame nicely, optionally with a caption and a row cap."""
    obj = df.head(max_rows) if (max_rows is not None and len(df) > max_rows) else df
    styler = obj.style
    if caption:
        styler = styler.set_caption(caption)
    display(styler)
    if max_rows is not None and len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows (showing first {max_rows} of {len(df)})")


def display_image(path: str, width: int | None = None) -> None:
    """Display a saved image (e.g. a matplotlib figure written to disk)."""
    display(Image(filename=path, width=width))
