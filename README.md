# Bayesian Spatial Modelling of Seismic Activity — Japan

Data Analytics project (AGH, AIR ISZ). Team: Paweł Majerczyk, Jakub Gicala.

Bayesian modelling of the annual number of **M ≥ 4.0 earthquakes per spatial
grid cell** over Japan (2000–2023), comparing a spatially-naive **no-pooling**
Poisson model with a **hierarchical partial-pooling** model. The deliverable is
an uncertainty-aware posterior map of seismic intensity.

**Tooling:** CmdStanPy + ArviZ + GeoPandas + Matplotlib (Python, Jupyter).

## Results in one line
WAIC narrowly favours the hierarchical model, PSIS-LOO flips to the no-pooling
model, and **both criteria warn** — driven by the 2011 Tohoku (M9.0) outlier
(Pareto-k up to ~9). The criteria are inconclusive; we select **Model 2** on
modelling-quality grounds (shrinkage, data-supported hierarchy, better map).

## Repository layout
```
data/
  raw/earthquakes_japan.csv          # ~33k events from the USGS API
  processed/grid_annual_counts.csv   # count per cell per year (model input)
  processed/grid_metadata.csv        # one row per cell (centres, totals)
  processed/idata_model{1,2}.nc      # saved ArviZ InferenceData
models/
  model1_nopool.stan                 # no-pooling Poisson (sigma fixed)
  model2_partial.stan                # partial pooling (sigma_global estimated)
notebooks/
  01_data_acquisition.ipynb          # USGS source, columns, raw map
  02_preprocessing.ipynb             # grid aggregation + EDA
  03_priors.ipynb                    # prior rationale + prior predictive checks
  04_model1_posterior.ipynb          # fit + diagnostics + ppc (Model 1)
  05_model2_posterior.ipynb          # fit + shrinkage + posterior map (Model 2)
  06_comparison.ipynb                # WAIC + PSIS-LOO + final assessment
report/
  main_report.ipynb                  # consolidated summary
  figures/                           # exported headline figures
utils/
  display.py                         # display_df / display_image helpers
  data_prep.py                       # grid definition + Stan-input builder
  notebook.py                        # run_notebook helper
  download_data.py                   # year-by-year USGS downloader
build_notebooks*.py                  # scripts that generate the notebooks
run_notebook.py                      # executes notebooks in place
```

## Reproduce
```bash
python3 -m venv venv && source venv/bin/activate
pip install cmdstanpy arviz geopandas pandas numpy matplotlib ipykernel jupyter nbconvert h5netcdf netcdf4
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

python utils/download_data.py                       # download raw data
python build_notebooks.py && python build_notebooks2.py \
  && python build_notebooks3.py && python build_notebooks4.py   # (re)generate notebooks
python run_notebook.py notebooks/01_data_acquisition.ipynb \
  notebooks/02_preprocessing.ipynb notebooks/03_priors.ipynb \
  notebooks/04_model1_posterior.ipynb notebooks/05_model2_posterior.ipynb \
  notebooks/06_comparison.ipynb                     # execute end-to-end
```
Run the notebooks in numerical order: 04/05 read the processed grid from 02 and
write InferenceData that 06 consumes.

## Mapping to grading criteria
1. **Problem formulation** — notebooks 01, 02
2. **Model** (two structurally different models) — `models/`, notebooks 04, 05
3. **Priors** (rationale + both prior predictive checks) — notebook 03
4. **Posterior — Model 1** — notebook 04
5. **Posterior — Model 2** — notebook 05
6. **Model comparison** (WAIC + LOO + assessment) — notebook 06
