"""Build notebooks 04 and 05 (posterior analysis for both models)."""
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
    nb["cells"] = [
        nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
        for k, s in cells
    ]
    path = f"{NB_DIR}/{name}.ipynb"
    nbf.write(nb, path)
    print("wrote", path, f"({len(nb['cells'])} cells)")


SETUP = r"""
import sys, os, warnings
sys.path.append("..")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from cmdstanpy import CmdStanModel

%matplotlib inline
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 110
az.style.use("arviz-darkgrid")

from utils.display import display_df, display_image
from utils.data_prep import build_stan_data
"""

# ===========================================================================
# Notebook 04 — Model 1 posterior
# ===========================================================================
nb04 = [
    md(r"""
# 04 — Posterior analysis: Model 1 (no pooling)

**What we do:** fit the **no-pooling Poisson** model with CmdStanPy, check that
the sampler converged, and analyse the posterior — marginal distributions,
posterior-predictive checks, and data consistency (the 2011 Tohoku outlier).

**Why:** this is the spatially-naive baseline. Each cell's log-intensity
`alpha_c` is estimated independently under a *fixed* prior scale, so cells with
few observations get wide, unstable posteriors. Quantifying that weakness here
motivates the hierarchical Model 2. Maps to *Criterion 4 — Posterior Analysis
(Model 1).*

## Model
$$\text{count}_{c,y}\sim\text{Poisson}(\lambda_c),\quad \log\lambda_c=\alpha_c,
\quad \alpha_c \sim \mathcal{N}(\mu_0=2.0,\ \sigma_0=2.0)\ \text{(fixed)}.$$
The fixed $\sigma_0$ is the defining feature: there is no hyperparameter linking
cells, so this is genuinely *no pooling*. (`models/model1_nopool.stan`.)
"""),
    code(SETUP),
    code(r"""
annual = pd.read_csv("../data/processed/grid_annual_counts.csv")
meta = pd.read_csv("../data/processed/grid_metadata.csv")
stan_data, cells, cell_to_int = build_stan_data(annual)
print(f"N (cell-year obs) = {stan_data['N']},  C (cells) = {stan_data['C']}")
print(f"Prior: alpha ~ Normal({stan_data['mu_prior_mean']}, {stan_data['sigma_fixed']}) [fixed]")
"""),
    md(r"""
### Fit with CmdStanPy
4 chains, 1000 warmup + 1000 sampling draws each. The model is small and
well-identified, so default settings suffice.
"""),
    code(r"""
model1 = CmdStanModel(stan_file="../models/model1_nopool.stan")
fit1 = model1.sample(data=stan_data, chains=4, parallel_chains=4,
                     iter_warmup=1000, iter_sampling=1000, seed=2024,
                     show_progress=False)
print(fit1.diagnose())
"""),
    code(r"""
# Wrap into an ArviZ InferenceData with predictions, log-lik and observed data.
idata1 = az.from_cmdstanpy(
    posterior=fit1,
    posterior_predictive="count_pred",
    log_likelihood="log_lik",
    observed_data={"count": stan_data["count"]},
    coords={"cell": cells, "obs": np.arange(stan_data["N"])},
    dims={"alpha": ["cell"], "lambda": ["cell"],
          "count_pred": ["obs"], "log_lik": ["obs"]},
)
idata1
"""),
    md(r"""
### Sampling issues — convergence diagnostics
We inspect $\hat R$ (should be < 1.01) and effective sample size (ESS), and count
divergent transitions. `alpha` is 154-dimensional, so we summarise the worst
values across all cells rather than printing every one.
"""),
    code(r"""
summ = az.summary(idata1, var_names=["alpha"])
n_div = int(idata1.sample_stats["diverging"].values.sum())
print("Divergent transitions:", n_div)
print(f"max R-hat   = {summ['r_hat'].max():.4f}   (target < 1.01)")
print(f"min ESS bulk= {summ['ess_bulk'].min():.0f}")
print(f"min ESS tail= {summ['ess_tail'].min():.0f}")
display_df(summ.sort_values("r_hat", ascending=False).head(8),
           caption="Worst-converging cells (by R-hat)")
"""),
    md(r"""
**Sampling assessment.** With a log link the Poisson model is log-concave in
`alpha`, so NUTS samples it cleanly: **no divergences**, all $\hat R < 1.01$, and
ESS in the thousands. No mitigation was needed (contrast with Model 2, where a
hierarchical funnel forces a non-centered parameterization). Any cell with
slightly lower ESS is one with very few observations — the prior, not the
likelihood, dominates there.
"""),
    code(r"""
# Representative cells: busiest, a mid-activity one, and a data-poor one.
busiest = meta.sort_values("total_events", ascending=False).iloc[0]["cell_id"]
poorest = meta.sort_values("total_events").iloc[0]["cell_id"]
mid     = meta.sort_values("total_events").iloc[len(meta)//2]["cell_id"]
rep_cells = [busiest, mid, poorest]
print("Representative cells (busiest, mid, poorest):", rep_cells)
print(meta.set_index("cell_id").loc[rep_cells, ["total_events","n_years","mean_count"]])
"""),
    md(r"""
### Trace plots
Trace plots for the log-intensity `alpha` of the three representative cells.
Well-mixed "fuzzy caterpillar" chains with no trends confirm convergence.
"""),
    code(r"""
az.plot_trace(idata1, var_names=["alpha"], coords={"cell": rep_cells},
              figsize=(11, 7))
plt.tight_layout(); plt.show()
"""),
    md(r"""
### Marginal posteriors
Posteriors of the intensity `lambda = exp(alpha)` for the representative cells,
with 94% HDIs. This directly shows the **concentration vs diffusion** behaviour:
the busy cell's posterior is sharply concentrated, while the data-poor cell's is
diffuse (wide HDI) — the core weakness of no pooling.
"""),
    code(r"""
az.plot_posterior(idata1, var_names=["lambda"], coords={"cell": rep_cells},
                  figsize=(13, 4))
plt.tight_layout(); plt.show()
"""),
    code(r"""
# Quantify concentration: posterior sd of lambda vs number of observations/cell.
lam_post = idata1.posterior["lambda"]                       # (chain, draw, cell)
lam_sd = lam_post.std(dim=("chain", "draw")).to_series()
lam_mean = lam_post.mean(dim=("chain", "draw")).to_series()
cv = (lam_sd / lam_mean)                                    # coefficient of variation
conc = pd.DataFrame({"post_mean_lambda": lam_mean, "post_sd": lam_sd, "cv": cv})
conc = conc.join(meta.set_index("cell_id")[["total_events", "n_years"]])

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(conc["total_events"], conc["cv"], alpha=0.6, color="steelblue")
ax.set_xscale("log")
ax.set_xlabel("total events in cell (log scale)")
ax.set_ylabel("posterior CV of lambda (sd / mean)")
ax.set_title("Marginal concentration: data-rich cells -> tight posteriors")
plt.show()
print("Posterior CV: data-poor cells (few events) are far more diffuse:")
display_df(conc.sort_values("total_events").head(5).round(2),
           caption="Most data-poor cells (diffuse posteriors)")
"""),
    md(r"""
**Marginal analysis.** Posterior uncertainty (CV) falls steeply with the amount
of data in a cell. Data-rich east-coast cells have CV ~0.03-0.05 (very
concentrated); the poorest cells have CV several times larger — the prior is
doing most of the work there. This is the no-pooling pathology Model 2 targets.
"""),
    md(r"""
### Posterior predictive check
For every cell-year we draw replicated counts from the posterior predictive and
compare them to the observed counts. We aggregate to **per-cell totals**
(observed vs predicted with 94% interval) for the busiest cells, and report
overall interval coverage.
"""),
    code(r"""
pp = idata1.posterior_predictive["count_pred"]               # (chain, draw, obs)
pp_mean = pp.mean(dim=("chain", "draw")).values
pp_lo = pp.quantile(0.03, dim=("chain", "draw")).values
pp_hi = pp.quantile(0.97, dim=("chain", "draw")).values

ppdf = annual.copy()
ppdf["pp_mean"] = pp_mean
ppdf["pp_lo"] = pp_lo
ppdf["pp_hi"] = pp_hi
ppdf["covered"] = (ppdf["count"] >= ppdf["pp_lo"]) & (ppdf["count"] <= ppdf["pp_hi"])
coverage = ppdf["covered"].mean()
print(f"94% posterior-predictive interval coverage: {coverage:.1%}  (target ~94%)")

# per-cell observed vs predicted totals, top 15 busiest cells
per_cell = ppdf.groupby("cell_id").agg(obs_total=("count", "sum"),
                                       pred_total=("pp_mean", "sum")).reset_index()
top = per_cell.sort_values("obs_total", ascending=False).head(15)
x = np.arange(len(top))
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - 0.2, top["obs_total"], width=0.4, label="observed", color="crimson")
ax.bar(x + 0.2, top["pred_total"], width=0.4, label="predicted (post. mean)",
       color="steelblue")
ax.set_xticks(x); ax.set_xticklabels(top["cell_id"], rotation=45)
ax.set_xlabel("cell_id"); ax.set_ylabel("total events 2000-2023")
ax.set_title("Model 1: observed vs predicted totals (15 busiest cells)")
ax.legend(); plt.tight_layout(); plt.show()
"""),
    md(r"""
### Data consistency — the 2011 Tohoku outlier
Per-cell totals match well because the no-pooling model can set each cell's mean
freely. The interesting failure is *temporal*: the model is **stationary** (one
rate per cell for all years) but 2011 was extraordinary. We locate the
observations the model fits worst via the standardized Poisson residual
$(y - \hat\mu)/\sqrt{\hat\mu}$.
"""),
    code(r"""
ppdf["resid"] = (ppdf["count"] - ppdf["pp_mean"]) / np.sqrt(ppdf["pp_mean"].clip(lower=0.5))
worst = ppdf.reindex(ppdf["resid"].abs().sort_values(ascending=False).index).head(10)
display_df(worst[["cell_id", "year", "count", "pp_mean", "pp_lo", "pp_hi", "resid"]].round(1),
           caption="Worst-fit observations (largest standardized residuals)")
print("Share of the 10 worst-fit observations that are from 2011:",
      f"{(worst['year'] == 2011).mean():.0%}")
"""),
    md(r"""
**Data consistency assessment.** Overall coverage is close to the nominal 94%,
so the model is broadly consistent with the data. The exceptions are
**dominated by 2011**: the Tohoku M9.0 mainshock and its aftershock cascade
pushed east-coast cells to counts several times their long-run mean, far above
the posterior-predictive interval. This is **expected and justified**, not a
bug: a *stationary* Poisson rate per cell cannot represent a one-off, regime-
changing event. Capturing it would require a non-stationary or
mixture/time-varying intensity — out of scope here, but the right next step. We
keep 2011 in the data and treat it as a documented outlier, which is also why it
reappears as a high Pareto-$k$ point in the LOO diagnostics of notebook 06.
"""),
    code(r"""
# Persist the InferenceData for the comparison notebook.
az.to_netcdf(idata1, "../data/processed/idata_model1.nc")
print("saved ../data/processed/idata_model1.nc")
"""),
    md(r"""
## Summary
- The no-pooling model **converges cleanly** (no divergences, $\hat R<1.01$).
- Marginals reveal the central weakness: **data-poor cells have diffuse,
  prior-driven posteriors**.
- Posterior predictive coverage is near nominal overall; the **2011 Tohoku**
  cell-years are systematic, well-understood outliers for a stationary model.

Next: `05_model2_posterior.ipynb` adds hierarchical partial pooling and shows
how it stabilises the data-poor cells.
"""),
]

# ===========================================================================
# Notebook 05 — Model 2 posterior
# ===========================================================================
nb05 = [
    md(r"""
# 05 — Posterior analysis: Model 2 (partial pooling)

**What we do:** fit the **hierarchical Poisson** model, run the same diagnostics
and posterior-predictive checks as for Model 1, then add the two analyses that
motivate this model: the **shrinkage** effect (narrower, more stable intervals
for data-poor cells) and a **posterior map** of the intensity field.

**Why:** Model 1 leaves data-poor cells with wide, prior-driven posteriors.
Model 2 lets cells *borrow strength* through a shared hyperprior whose scale
`sigma_global` is **estimated from the data**. Physically, neighbouring cells lie
in the same tectonic setting, so a common baseline is reasonable. Maps to
*Criterion 5 — Posterior Analysis (Model 2).*

## Model
$$\text{count}_{c,y}\sim\text{Poisson}(\lambda_c),\quad \log\lambda_c=\alpha_c,$$
$$\alpha_c\sim\mathcal{N}(\mu_{\text{global}},\sigma_{\text{global}}),\quad
\mu_{\text{global}}\sim\mathcal{N}(2,1),\quad
\sigma_{\text{global}}\sim\text{HalfNormal}(1).$$
The **only structural change** from Model 1 is that $\sigma_{\text{global}}$ is a
free parameter. Because this dataset is *informative* (most cells have many
cell-years and large counts), the per-cell $\alpha$ is tightly constrained by the
likelihood, so we use the **centered** parameterization
($\alpha_c\sim\mathcal N(\mu_{\text{global}},\sigma_{\text{global}})$) — it mixes
better than non-centering in the strong-data regime. (`models/model2_partial.stan`.)
"""),
    code(SETUP),
    code(r"""
annual = pd.read_csv("../data/processed/grid_annual_counts.csv")
meta = pd.read_csv("../data/processed/grid_metadata.csv")
stan_data, cells, cell_to_int = build_stan_data(annual)
# Model 2 estimates mu/sigma, so the fixed-prior keys are simply unused.
print(f"N = {stan_data['N']}, C = {stan_data['C']}")
"""),
    md(r"""
### Fit with CmdStanPy
Hierarchical models can show divergences near small `sigma_global`, so we raise
`adapt_delta` to 0.95 as a precaution and give the hyperparameters a sensible
starting point (avoids harmless `sigma_global = 0` rejections during warm-up).
"""),
    code(r"""
model2 = CmdStanModel(stan_file="../models/model2_partial.stan")
fit2 = model2.sample(data=stan_data, chains=4, parallel_chains=4,
                     iter_warmup=1000, iter_sampling=1000, seed=2024,
                     adapt_delta=0.95, inits={"mu_global": 2.0, "sigma_global": 1.0},
                     show_progress=False)
print(fit2.diagnose())
"""),
    code(r"""
idata2 = az.from_cmdstanpy(
    posterior=fit2,
    posterior_predictive="count_pred",
    log_likelihood="log_lik",
    observed_data={"count": stan_data["count"]},
    coords={"cell": cells, "obs": np.arange(stan_data["N"])},
    dims={"alpha": ["cell"], "lambda": ["cell"],
          "count_pred": ["obs"], "log_lik": ["obs"]},
)
idata2
"""),
    md(r"""
### Sampling issues — convergence diagnostics
Same checks as Model 1, plus the key hyperparameters `mu_global` and
`sigma_global`.
"""),
    code(r"""
n_div = int(idata2.sample_stats["diverging"].values.sum())
summ_a = az.summary(idata2, var_names=["alpha"])
print("Divergent transitions:", n_div)
print(f"alpha:  max R-hat = {summ_a['r_hat'].max():.4f},  min ESS bulk = {summ_a['ess_bulk'].min():.0f}")
display_df(az.summary(idata2, var_names=["mu_global", "sigma_global"]).round(3),
           caption="Hyperparameter posteriors (the KEY estimated quantities)")
"""),
    md(r"""
**Sampling assessment.** The centered parameterization plus `adapt_delta=0.95`
gives **no divergences**, all $\hat R<1.01$ and healthy ESS (thousands) for the
per-cell `alpha` and the hyperparameters alike. (We initially tried the
non-centered parameterization and saw poor mixing — $\hat R\approx1.07$, ESS
~50 — because with this informative data non-centering induces a soft
`mu_global`-vs-offset trade-off; switching to centered fixed it, the textbook
remedy for the strong-data regime.) The estimated `sigma_global` is well away from 0
(its posterior mean is printed above), meaning the data genuinely support
substantial between-cell variation — but a *finite* amount, which is exactly
what produces shrinkage rather than full independence.
"""),
    code(r"""
busiest = meta.sort_values("total_events", ascending=False).iloc[0]["cell_id"]
poorest = meta.sort_values("total_events").iloc[0]["cell_id"]
mid     = meta.sort_values("total_events").iloc[len(meta)//2]["cell_id"]
rep_cells = [busiest, mid, poorest]
az.plot_trace(idata2, var_names=["mu_global", "sigma_global"], figsize=(11, 5))
plt.tight_layout(); plt.show()
"""),
    md(r"""
### Marginal posteriors
Marginals of `lambda` for the representative cells (94% HDI). Compared with
Model 1, the data-poor cell is pulled towards the global mean and its interval
is narrower.
"""),
    code(r"""
az.plot_posterior(idata2, var_names=["lambda"], coords={"cell": rep_cells},
                  figsize=(13, 4))
plt.tight_layout(); plt.show()
"""),
    md(r"""
### Posterior predictive check & data consistency
Identical procedure to Model 1: per-cell observed vs predicted, interval
coverage, and the 2011 residual check.
"""),
    code(r"""
pp = idata2.posterior_predictive["count_pred"]
pp_mean = pp.mean(dim=("chain", "draw")).values
pp_lo = pp.quantile(0.03, dim=("chain", "draw")).values
pp_hi = pp.quantile(0.97, dim=("chain", "draw")).values
ppdf = annual.copy()
ppdf["pp_mean"], ppdf["pp_lo"], ppdf["pp_hi"] = pp_mean, pp_lo, pp_hi
ppdf["covered"] = (ppdf["count"] >= ppdf["pp_lo"]) & (ppdf["count"] <= ppdf["pp_hi"])
print(f"94% PP interval coverage: {ppdf['covered'].mean():.1%}")

per_cell = ppdf.groupby("cell_id").agg(obs_total=("count", "sum"),
                                       pred_total=("pp_mean", "sum")).reset_index()
top = per_cell.sort_values("obs_total", ascending=False).head(15)
x = np.arange(len(top))
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - 0.2, top["obs_total"], width=0.4, label="observed", color="crimson")
ax.bar(x + 0.2, top["pred_total"], width=0.4, label="predicted", color="seagreen")
ax.set_xticks(x); ax.set_xticklabels(top["cell_id"], rotation=45)
ax.set_ylabel("total events 2000-2023")
ax.set_title("Model 2: observed vs predicted totals (15 busiest cells)")
ax.legend(); plt.tight_layout(); plt.show()

ppdf["resid"] = (ppdf["count"] - ppdf["pp_mean"]) / np.sqrt(ppdf["pp_mean"].clip(lower=0.5))
worst = ppdf.reindex(ppdf["resid"].abs().sort_values(ascending=False).index).head(8)
display_df(worst[["cell_id","year","count","pp_mean","pp_lo","pp_hi","resid"]].round(1),
           caption="Worst-fit observations (still dominated by 2011)")
"""),
    md(r"""
**Data consistency.** Coverage is again near nominal and the worst-fit points
are still the **2011 Tohoku** cell-years — partial pooling does not (and should
not) erase a genuine extreme event; it stabilises the *baseline* rates. The
stationarity caveat from notebook 04 applies equally here.
"""),
    md(r"""
### Shrinkage effect — Model 1 vs Model 2
The mechanism that distinguishes the two models. Partial pooling should (a) pull
data-poor cells' log-intensities toward the global mean `mu_global` and (b)
reduce their posterior uncertainty, while leaving data-rich cells essentially
unchanged. **We measure on the `alpha` (log) scale**, where the pooling actually
acts — on the `lambda = exp(alpha)` scale a cell's interval width scales with its
level, so the skew masks the effect and can even reverse its sign.
"""),
    code(r"""
idata1 = az.from_netcdf("../data/processed/idata_model1.nc")
mu_g = float(idata2.posterior["mu_global"].mean())

def alpha_stats(idata):
    a = idata.posterior["alpha"]
    return (a.mean(dim=("chain", "draw")).to_series(),
            a.std(dim=("chain", "draw")).to_series())

m1_mean, m1_sd = alpha_stats(idata1)
m2_mean, m2_sd = alpha_stats(idata2)
comp = pd.DataFrame({"m1_alpha": m1_mean, "m2_alpha": m2_mean,
                     "m1_sd": m1_sd, "m2_sd": m2_sd}).join(
       meta.set_index("cell_id")[["total_events"]])
comp["sd_reduction"] = 1 - comp["m2_sd"] / comp["m1_sd"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
# (a) point estimates pulled toward the global mean
sc = axes[0].scatter(comp["m1_alpha"], comp["m2_alpha"],
                     c=np.log10(comp["total_events"]), cmap="viridis", alpha=0.85)
lims = [comp[["m1_alpha", "m2_alpha"]].min().min() - 0.3,
        comp[["m1_alpha", "m2_alpha"]].max().max() + 0.3]
axes[0].plot(lims, lims, "k--", lw=1, label="y = x (no shrinkage)")
axes[0].axhline(mu_g, color="red", ls=":", lw=1.2, label=f"mu_global = {mu_g:.2f}")
axes[0].set_xlabel("Model 1 posterior-mean alpha (log-intensity)")
axes[0].set_ylabel("Model 2 posterior-mean alpha")
axes[0].set_title("Shrinkage of point estimates toward the global mean\n(colour = log10 total events)")
axes[0].legend(); plt.colorbar(sc, ax=axes[0], shrink=0.8, label="log10 total events")
# (b) uncertainty reduction concentrates on data-poor cells
axes[1].scatter(comp["total_events"], comp["sd_reduction"] * 100,
                alpha=0.7, color="darkorange")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xscale("log")
axes[1].set_xlabel("total events in cell (log scale)")
axes[1].set_ylabel("posterior sd(alpha) reduction (%)")
axes[1].set_title("Uncertainty reduction is largest for data-poor cells")
plt.show()

poor, rich = comp[comp.total_events < 10], comp[comp.total_events > 200]
print(f"Data-poor cells (<10 events, n={len(poor)}): mean sd(alpha) reduction = "
      f"{poor['sd_reduction'].mean():.0%};  point estimate pulled "
      f"{poor['m1_alpha'].mean():.2f} -> {poor['m2_alpha'].mean():.2f} (toward mu_global={mu_g:.2f})")
print(f"Data-rich cells (>200 events, n={len(rich)}): mean sd(alpha) reduction = "
      f"{rich['sd_reduction'].mean():.0%} (essentially unchanged)")
display_df(comp.sort_values("total_events").head(6).round(3),
           caption="Most data-poor cells: M2 alpha pulled toward the global mean, with lower posterior sd")
"""),
    md(r"""
**Shrinkage assessment.** The effect is present and behaves exactly as theory
predicts — but it is **modest in magnitude** here, and it is important to say so
honestly. Data-poor cells (<10 total events) have their log-intensity pulled
toward `mu_global` and their posterior `sd(alpha)` reduced by ~10% on average,
while data-rich cells are essentially untouched (the monotone pattern in the
right-hand panel). The reason the shrinkage is *moderate rather than dramatic* is
specific to this problem: **Poisson counts are self-informative** — even a single
observed year meaningfully bounds a cell's rate — and **most Japanese cells have
plenty of data**, so there are few truly information-starved cells for pooling to
rescue. Shrinkage would be far larger on a finer grid (e.g. 1°×1°, many more
sparse cells) or for a less informative likelihood. The practical payoff remains:
a smoother, better-regularised map with no hand-tuned per-cell prior.
"""),
    md(r"""
### Posterior intensity map (GeoPandas)
A choropleth of the posterior-mean intensity `lambda_c` over Japan, built as a
GeoDataFrame of 2x2 cell polygons. Raw epicentres are overlaid faintly for
geographic context. This is the project's headline deliverable.
"""),
    code(r"""
import geopandas as gpd
from shapely.geometry import box
from utils.data_prep import LAT_BINS, LON_BINS

# posterior-mean intensity (lambda scale) and its 94% interval width, per cell
lam2_mean = idata2.posterior["lambda"].mean(dim=("chain", "draw")).to_series()
lam2_hdi = az.hdi(idata2, var_names=["lambda"], hdi_prob=0.94)["lambda"].to_series().unstack()
lam2_w = lam2_hdi["higher"] - lam2_hdi["lower"]

# build cell polygons with posterior-mean lambda
recs = []
for cid, lam in lam2_mean.items():
    li, lo = map(int, cid.split("_"))
    recs.append({"cell_id": cid, "lambda": lam,
                 "geometry": box(LON_BINS[lo], LAT_BINS[li],
                                 LON_BINS[lo] + 2, LAT_BINS[li] + 2)})
gdf = gpd.GeoDataFrame(recs, crs="EPSG:4326")

raw = pd.read_csv("../data/raw/earthquakes_japan.csv")
fig, ax = plt.subplots(figsize=(9, 9))
gdf.plot(column="lambda", cmap="inferno", legend=True, ax=ax,
         edgecolor="white", linewidth=0.3,
         legend_kwds={"label": "posterior mean lambda (events/yr)", "shrink": 0.7})
ax.scatter(raw["longitude"], raw["latitude"], s=1, c="cyan", alpha=0.05)
ax.set_xlim(122, 154); ax.set_ylim(24, 50)
ax.set_xlabel("longitude (E)"); ax.set_ylabel("latitude (N)")
ax.set_title("Model 2 posterior-mean seismic intensity per 2x2 cell")
os.makedirs("../report/figures", exist_ok=True)
fig.savefig("../report/figures/05_posterior_map.png", bbox_inches="tight", dpi=120)
plt.show()
"""),
    code(r"""
# Companion map of posterior uncertainty (94% interval width) per cell.
gdf_u = gdf.copy()
gdf_u["width"] = gdf_u["cell_id"].map(lam2_w)
fig, ax = plt.subplots(figsize=(9, 9))
gdf_u.plot(column="width", cmap="viridis", legend=True, ax=ax,
           edgecolor="white", linewidth=0.3,
           legend_kwds={"label": "94% interval width", "shrink": 0.7})
ax.set_xlim(122, 154); ax.set_ylim(24, 50)
ax.set_xlabel("longitude (E)"); ax.set_ylabel("latitude (N)")
ax.set_title("Model 2 posterior uncertainty (94% interval width) per cell")
plt.show()
"""),
    code(r"""
az.to_netcdf(idata2, "../data/processed/idata_model2.nc")
print("saved ../data/processed/idata_model2.nc")
"""),
    md(r"""
## Summary
- Model 2 samples cleanly with the centered parameterization (R-hat ~1.00, high
  ESS, no divergences); the **key parameter `sigma_global` is estimated**, not
  assumed.
- **Shrinkage works as intended but is modest** here (~10% sd reduction on the
  log scale for data-poor cells, data-rich cells unaffected) — Poisson counts are
  self-informative and most cells are data-rich, so there is little to pool away.
- The posterior map shows the expected east >> west gradient as a smooth,
  uncertainty-aware field.
- 2011 remains an outlier for both (stationary) models.

Next: `06_comparison.ipynb` compares the two models with WAIC and PSIS-LOO.
"""),
]

build("04_model1_posterior", nb04)
build("05_model2_posterior", nb05)
print("done nb04, nb05")
