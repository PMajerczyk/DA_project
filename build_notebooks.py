"""Construct the six project notebooks with nbformat.

Each notebook is built from a list of (kind, source) cells. We write the .ipynb
files here; execution (to populate outputs) is a separate step.

Convention: notebooks live in notebooks/ and are executed with that as the
working directory, so data/model paths are prefixed with '../'.
"""
import nbformat as nbf

NB_DIR = "notebooks"


def md(text):
    return ("md", text.strip("\n"))


def code(src):
    return ("code", src.strip("\n"))


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    out = []
    for kind, src in cells:
        if kind == "md":
            out.append(nbf.v4.new_markdown_cell(src))
        else:
            out.append(nbf.v4.new_code_cell(src))
    nb["cells"] = out
    path = f"{NB_DIR}/{name}.ipynb"
    nbf.write(nb, path)
    print("wrote", path, f"({len(out)} cells)")


# Common header code injected at top of every analysis notebook.
SETUP = r"""
import sys, os, warnings
sys.path.append("..")            # make the utils package importable
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 110

from utils.display import display_df, display_image
"""

# ---------------------------------------------------------------------------
# Notebook 01 — Data acquisition
# ---------------------------------------------------------------------------
nb01 = [
    md(r"""
# 01 — Data Acquisition

**What we do:** load the raw earthquake catalogue downloaded from the USGS
Earthquake Catalog API for Japan and describe the data source and its columns.

**Why:** the whole project models seismic activity, so we first need a clean,
documented record of *where* and *when* earthquakes happened. This notebook
establishes provenance (where the data comes from), shows the variables we have,
and produces a first geographic picture of the raw events. It maps directly to
*Criterion 1 — Problem Formulation: Data Source / Description.*

## The phenomenon and use case
We model the **annual number of magnitude >= 4.0 earthquakes per spatial grid
cell** over Japan. Such intensity maps are used for **seismic hazard
assessment**, insurance/re-insurance pricing, infrastructure planning and
prioritising monitoring. A Bayesian treatment gives not just a point estimate
of activity per region but a full *uncertainty* around it — essential when some
cells have very few observations.

## Data source — USGS Earthquake Catalog
- **Provider:** U.S. Geological Survey (USGS), FDSN `event` web service.
- **Endpoint:** `https://earthquake.usgs.gov/fdsnws/event/1/query` (CSV, no
  registration required).
- **Query used:** bounding box for Japan `lat 24-50 N`, `lon 122-154 E`,
  `minmagnitude = 4.0`, period `2000-2023`.
- **Download strategy:** the API returns at most 20 000 records per request, so
  `utils/download_data.py` downloads **year by year** (24 requests) with a
  1-second pause between calls and concatenates the results.

### Columns of interest
| column | meaning |
|--------|---------|
| `time` | event origin time (UTC, ISO-8601) |
| `latitude`, `longitude` | epicentre coordinates (degrees) |
| `depth` | hypocentre depth (km) |
| `mag`, `magType` | magnitude and its scale (mostly `mb`/`mww`) |
| `place` | human-readable location |
| `id`, `net` | catalogue identifiers |
| `*Error`, `nst`, `gap`, `rms`, `dmin` | measurement-quality metadata |
"""),
    code(SETUP),
    code(r"""
RAW_PATH = "../data/raw/earthquakes_japan.csv"
df = pd.read_csv(RAW_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
df.head()
"""),
    md(r"""
### Basic statistics
We confirm the data covers the intended region, magnitude floor and time span,
and check for missing values in the variables we will actually use.
"""),
    code(r"""
df["year"] = df["time"].str[:4].astype(int)

print("Time span      :", df["year"].min(), "->", df["year"].max())
print("Magnitude range:", df["mag"].min(), "->", df["mag"].max())
print("Latitude range :", round(df["latitude"].min(),2), "->", round(df["latitude"].max(),2))
print("Longitude range:", round(df["longitude"].min(),2), "->", round(df["longitude"].max(),2))
print("\nMissing values in key columns:")
print(df[["latitude","longitude","mag","time"]].isnull().sum())

display_df(df[["depth","mag","horizontalError","depthError"]].describe(),
           caption="Numeric summary of selected columns")
"""),
    code(r"""
# Events per year — note the huge spike in 2011 (Tohoku M9.0 + aftershocks).
per_year = df.groupby("year").size()
ax = per_year.plot(kind="bar", color="steelblue", edgecolor="k")
ax.set_title("M>=4.0 earthquakes per year — Japan region (USGS)")
ax.set_xlabel("year"); ax.set_ylabel("number of events")
plt.tight_layout(); plt.show()
print("2011 events:", int(per_year.loc[2011]),
      "vs median year:", int(per_year.median()))
"""),
    md(r"""
### Map of raw events
A scatter of every epicentre, coloured by magnitude. This is our first look at
the **spatial structure** the models must capture: activity is concentrated
along the Pacific subduction zones on the **east coast** and the Nankai trough
to the south, and is far sparser inland and to the west.
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(df["longitude"], df["latitude"], c=df["mag"],
                cmap="viridis", s=6, alpha=0.4)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label("magnitude")
ax.set_xlim(122, 154); ax.set_ylim(24, 50)
ax.set_xlabel("longitude (E)"); ax.set_ylabel("latitude (N)")
ax.set_title("Raw earthquake epicentres, Japan 2000-2023 (M>=4.0)")
ax.set_aspect("equal")
os.makedirs("../report/figures", exist_ok=True)
fig.savefig("../report/figures/01_raw_map.png", bbox_inches="tight", dpi=120)
plt.show()
"""),
    code(r"""
# Demonstrating the display_image helper on the figure we just saved.
display_image("../report/figures/01_raw_map.png", width=520)
"""),
    md(r"""
## Summary
- We loaded **~33 000 events** spanning 2000-2023, magnitude 4.0-9.1, all within the
  Japan bounding box and with **no missing coordinates / magnitudes**.
- The **2011 spike** (Tohoku M9.0 and its aftershock sequence) is already
  visible and will reappear as an *outlier* in the posterior-predictive and
  model-comparison notebooks.
- Activity is strongly **spatially structured** (east >> west), which motivates
  comparing a spatially-naive model with a hierarchical one.

Next: `02_preprocessing.ipynb` aggregates these point events into a 2x2 degree
grid of annual counts — the input to the Bayesian models.
"""),
]

# ---------------------------------------------------------------------------
# Notebook 02 — Preprocessing
# ---------------------------------------------------------------------------
nb02 = [
    md(r"""
# 02 — Preprocessing: aggregation to a spatial grid

**What we do:** turn the ~33 000 individual epicentres into the modelling unit —
the **number of M>=4.0 events per 2x2 degree grid cell per year** — and run
exploratory data analysis on the result.

**Why:** the Bayesian models treat each *cell-year* as one Poisson count
observation. Aggregating to a regular grid (a) gives a well-defined count
variable, (b) defines the spatial units across which Model 2 will pool
information, and (c) makes the "rare cell vs busy cell" contrast explicit. This
maps to *Criterion 1 — Preprocessing.*

The grid and the Stan-input construction live in `utils/data_prep.py` so that
both models are fed identical data.
"""),
    code(SETUP),
    code(r"""
from utils.data_prep import assign_cells, cell_center, LAT_BINS, LON_BINS

df = pd.read_csv("../data/raw/earthquakes_japan.csv")
df["year"] = df["time"].str[:4].astype(int)
print("Grid: lat bins", LAT_BINS, "\n      lon bins", LON_BINS)
print(f"Cell size: 2 deg x 2 deg  ->  up to {(len(LAT_BINS)-1)*(len(LON_BINS)-1)} cells")
"""),
    md(r"""
### Step 1 — assign each event to a grid cell
`pd.cut` maps latitude/longitude to integer bin indices `(lat_idx, lon_idx)`,
combined into a string `cell_id`. Events outside the grid edges are dropped.
"""),
    code(r"""
df = assign_cells(df)
print("Events kept after gridding:", len(df))
print("Distinct cells with >=1 event:", df["cell_id"].nunique())
display_df(df[["time","latitude","longitude","mag","lat_idx","lon_idx","cell_id"]],
           caption="Events with assigned grid cells")
"""),
    md(r"""
### Step 2 — annual counts per cell
Each `(cell_id, year)` pair becomes one observation with its event `count`. We
attach the cell-centre coordinates (for mapping) to build
`grid_annual_counts.csv`, and write a separate `grid_metadata.csv` with one row
per cell.

**Design choice — only observed cell-years are rows.** A cell-year with zero
recorded events almost always means "this ocean/inland cell is not seismically
active at M>=4", not "an active cell happened to be silent". Including a flood of
structural zeros would distort the Poisson means. We therefore model the
**154 cells that ever produced an event**, each over the years in which it was
active. (This is discussed again as a modelling assumption in notebook 04.)
"""),
    code(r"""
annual = (df.groupby(["cell_id", "lat_idx", "lon_idx", "year"])
            .size().reset_index(name="count"))

# cell-centre coordinates
centers = annual.apply(lambda r: cell_center(int(r.lat_idx), int(r.lon_idx)),
                       axis=1, result_type="expand")
annual["lat_center"] = centers[0]
annual["lon_center"] = centers[1]
annual = annual[["cell_id","year","count","lat_idx","lon_idx","lat_center","lon_center"]]

metadata = (annual.groupby(["cell_id","lat_idx","lon_idx","lat_center","lon_center"])
                  .agg(n_years=("year","nunique"),
                       total_events=("count","sum"),
                       mean_count=("count","mean"))
                  .reset_index())

os.makedirs("../data/processed", exist_ok=True)
annual.to_csv("../data/processed/grid_annual_counts.csv", index=False)
metadata.to_csv("../data/processed/grid_metadata.csv", index=False)

print("grid_annual_counts:", annual.shape, " grid_metadata:", metadata.shape)
display_df(annual.sort_values("count", ascending=False),
           caption="grid_annual_counts.csv (sorted by count — Tohoku 2011 on top)")
"""),
    code(r"""
display_df(metadata.sort_values("total_events", ascending=False),
           caption="grid_metadata.csv (one row per cell, sorted by total events)")
"""),
    md(r"""
### EDA 1 — distribution of annual counts
The count distribution is extremely **right-skewed**: most cell-years have only
a handful of events, while a few (east-coast cells, especially in 2011) have
hundreds. The log-scale histogram makes the bulk visible. This skew, plus
counts being non-negative integers, is exactly why a **Poisson with a log link**
is the natural likelihood.
"""),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(annual["count"], bins=60, color="steelblue", edgecolor="k")
axes[0].set_title("Annual count per cell (linear)")
axes[0].set_xlabel("events / cell / year"); axes[0].set_ylabel("frequency")

axes[1].hist(np.log1p(annual["count"]), bins=40, color="darkorange", edgecolor="k")
axes[1].set_title("Annual count per cell (log1p)")
axes[1].set_xlabel("log(1 + count)"); axes[1].set_ylabel("frequency")
plt.tight_layout(); plt.show()

display_df(annual["count"].describe().to_frame("count"),
           caption="Summary of the annual-count response")
"""),
    md(r"""
### EDA 2 — map of mean activity per cell
Average annual count per cell, drawn on the grid. The east-coast subduction
band is an order of magnitude more active than inland/western cells — the
spatial signal Model 2 will try to exploit.
"""),
    code(r"""
# build a lat_idx x lon_idx grid of mean counts for a heatmap
nlat, nlon = len(LAT_BINS)-1, len(LON_BINS)-1
grid = np.full((nlat, nlon), np.nan)
for _, r in metadata.iterrows():
    grid[int(r.lat_idx), int(r.lon_idx)] = r["mean_count"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(grid, origin="lower", cmap="magma",
               extent=[LON_BINS[0], LON_BINS[-1], LAT_BINS[0], LAT_BINS[-1]],
               aspect="equal")
cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label("mean events / year")
ax.set_xlabel("longitude (E)"); ax.set_ylabel("latitude (N)")
ax.set_title("Mean annual M>=4.0 activity per 2x2 cell")
plt.show()
"""),
    md(r"""
## Summary
- Aggregation yields **{C} active cells** and **{N} cell-year observations**
  (numbers printed above).
- The response is heavily right-skewed with a long tail dominated by 2011 —
  motivating a Poisson/log-link model and foreshadowing the Tohoku outlier
  discussion.
- Strong east-west spatial gradient is the physical basis for the partial-
  pooling model.

Outputs written: `data/processed/grid_annual_counts.csv`,
`data/processed/grid_metadata.csv`. Next: `03_priors.ipynb`.
"""),
]

build("01_data_acquisition", nb01)
build("02_preprocessing", nb02)
print("done part 1")
