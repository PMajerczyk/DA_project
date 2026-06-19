"""Shared data-preparation helpers used across the modelling notebooks.

Keeps the grid definition and the Stan-input construction in one place so that
Model 1 and Model 2 are fed *exactly* the same data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Grid definition for Japan (2 deg x 2 deg cells).
LAT_BINS = np.arange(24, 51, 2)   # 24..50 N
LON_BINS = np.arange(122, 155, 2)  # 122..154 E

# Prior on the per-cell log-intensity alpha. Derived ONLY from external domain
# knowledge (no statistics of our own counts -> no double-dipping):
#   centre: published Japan rate ~1200 M>=4/yr  /  208 grid cells (geometry)
#           = ~5.8 events/cell/yr  ->  mu_0 = log(5.8) ~ 1.8
#   spread: wide/weakly-informative so +-3 sigma spans log[1, 3000] = [0, 8]
#           -> sigma_0 = (8 - 1.8)/3 ~ 2.07
# See notebooks/03_priors.ipynb for the full derivation and sensitivity analysis.
PRIOR_MU = 1.8        # prior mean for alpha (log-intensity); exp(1.8) ~ 5.8 events/yr
PRIOR_SIGMA_M1 = 2.07  # FIXED prior sd for Model 1 (no pooling)


def assign_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Add lat_idx, lon_idx and cell_id columns based on the 2x2 grid."""
    df = df.copy()
    df["lat_idx"] = pd.cut(df["latitude"], bins=LAT_BINS, labels=False, right=False)
    df["lon_idx"] = pd.cut(df["longitude"], bins=LON_BINS, labels=False, right=False)
    df = df.dropna(subset=["lat_idx", "lon_idx"]).copy()
    df["lat_idx"] = df["lat_idx"].astype(int)
    df["lon_idx"] = df["lon_idx"].astype(int)
    df["cell_id"] = df["lat_idx"].astype(str) + "_" + df["lon_idx"].astype(str)
    return df


def cell_center(lat_idx: int, lon_idx: int) -> tuple[float, float]:
    """Return (lat_center, lon_center) for a grid index pair."""
    return (LAT_BINS[lat_idx] + 1.0, LON_BINS[lon_idx] + 1.0)


def build_stan_data(counts: pd.DataFrame) -> dict:
    """Build the Stan data dict from the long-format grid_annual_counts table.

    counts must have columns: cell_id, year, count.
    Returns the dict plus the cell ordering so posteriors can be mapped back.
    """
    cells = sorted(counts["cell_id"].unique())
    cell_to_int = {c: i + 1 for i, c in enumerate(cells)}  # Stan is 1-based
    cell_idx = counts["cell_id"].map(cell_to_int).to_numpy()

    stan_data = {
        "N": int(len(counts)),
        "C": int(len(cells)),
        "cell_id": cell_idx.astype(int),
        "count": counts["count"].to_numpy().astype(int),
        "mu_prior_mean": PRIOR_MU,
        "sigma_fixed": PRIOR_SIGMA_M1,
    }
    return stan_data, cells, cell_to_int
