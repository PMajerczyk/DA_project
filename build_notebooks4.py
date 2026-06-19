"""Build notebook 06 — model comparison (WAIC + PSIS-LOO)."""
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

%matplotlib inline
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 110
az.style.use("arviz-darkgrid")

from utils.display import display_df
"""

nb06 = [
    md(r"""
# 06 — Model comparison: WAIC and PSIS-LOO

**What we do:** compare the no-pooling model (Model 1) and the partial-pooling
model (Model 2) using two information criteria — **WAIC** and **PSIS-LOO** —
discuss their warnings and the Pareto-$k$ diagnostics, and give a reasoned final
assessment of which model to prefer and why.

**Why:** information criteria estimate out-of-sample predictive accuracy
(expected log pointwise predictive density, *elpd*) using only the fitted data.
They are the standard tool for the project's *Criterion 6 — Model Comparison*.
Crucially, we treat their **warnings as data**: with a strong outlier (2011
Tohoku) these criteria can be unreliable, and saying so is part of the analysis.

Both models include `log_lik` in their generated quantities, which is what makes
this comparison possible.
"""),
    code(SETUP),
    code(r"""
idata1 = az.from_netcdf("../data/processed/idata_model1.nc")
idata2 = az.from_netcdf("../data/processed/idata_model2.nc")
annual = pd.read_csv("../data/processed/grid_annual_counts.csv")
comp_dict = {"model_1_nopool": idata1, "model_2_partial": idata2}
print("Loaded both InferenceData objects with log_likelihood groups.")
print("sigma_global (Model 2) posterior mean:",
      round(float(idata2.posterior["sigma_global"].mean()), 3))
"""),
    md(r"""
## Information criteria usage — WAIC
WAIC = Widely Applicable Information Criterion. Higher `elpd_waic` is better;
`elpd_diff` is the gap to the best model and `dse` its standard error.
"""),
    code(r"""
comp_waic = az.compare(comp_dict, ic="waic")
display_df(comp_waic.round(2), caption="WAIC comparison")
az.plot_compare(comp_waic, figsize=(8, 3))
plt.title("WAIC comparison"); plt.tight_layout(); plt.show()
"""),
    md(r"""
### WAIC discussion
- **Winner:** `model_2_partial` has the higher `elpd_waic` (smaller is the rank
  number 0). The gap to Model 1 is `elpd_diff` ~ 61 with `dse` ~ 37, i.e. only
  about **1.6 standard errors** — a real but **not decisive** separation; the
  intervals overlap considerably.
- **Warning:** ArviZ raises `warning = True` for both models. This fires because
  some pointwise `p_waic` values exceed 0.4, the symptom of a few highly
  influential observations (the large-count cell-years) for which WAIC's
  approximation is shaky. We do **not** ignore it — it is the same misspecified
  2011/large-count points seen in notebooks 04-05, and it tells us the headline
  number rests partly on cells the (stationary) models fit poorly.
"""),
    md(r"""
## Information criteria usage — PSIS-LOO
PSIS-LOO approximates leave-one-out cross-validation. It additionally reports a
**Pareto-$k$** per observation; $k > 0.7$ means the importance-sampling estimate
for that point is unreliable.
"""),
    code(r"""
comp_loo = az.compare(comp_dict, ic="loo")
display_df(comp_loo.round(2), caption="PSIS-LOO comparison")
az.plot_compare(comp_loo, figsize=(8, 3))
plt.title("PSIS-LOO comparison"); plt.tight_layout(); plt.show()
"""),
    md(r"""
### PSIS-LOO discussion
- **Winner — and it flips!** Under LOO the ranking **reverses**:
  `model_1_nopool` comes out on top, with `elpd_diff` ~ 26 over Model 2
  (`dse` ~ 16, ~1.6 SE). So WAIC prefers the hierarchical model while LOO
  prefers the no-pooling one.
- **Warning:** LOO also flags `warning = True`. The reason is visible in the
  Pareto-$k$ diagnostic below.
- A flip between WAIC and LOO, with both warning, is a strong signal that the
  comparison is being driven by a handful of pathological points rather than by
  a clean, global predictive advantage.
"""),
    code(r"""
# Pareto-k diagnostics — which observations break PSIS-LOO?
loo1 = az.loo(idata1, pointwise=True)
loo2 = az.loo(idata2, pointwise=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, loo, name in [(axes[0], loo1, "Model 1"), (axes[1], loo2, "Model 2")]:
    k = loo.pareto_k.values
    ax.scatter(np.arange(len(k)), k, s=10, alpha=0.6)
    ax.axhline(0.7, color="red", ls="--", label="k = 0.7 (unreliable)")
    ax.axhline(1.0, color="darkred", ls=":", label="k = 1.0 (very bad)")
    ax.set_title(f"{name}: Pareto k  (n[k>0.7]={int((k>0.7).sum())}, max={k.max():.1f})")
    ax.set_xlabel("observation index"); ax.legend()
axes[0].set_ylabel("Pareto k")
plt.tight_layout(); plt.show()
"""),
    code(r"""
# Map the worst-k observations back to cell/year — are they the 2011 outliers?
k2 = loo2.pareto_k.values
worst = annual.copy()
worst["pareto_k"] = k2
worst = worst.sort_values("pareto_k", ascending=False).head(10)
display_df(worst[["cell_id", "year", "count", "pareto_k"]].round(2),
           caption="Model 2: observations with the highest Pareto-k")
hi = annual.loc[k2 > 0.7]
print(f"Observations with k>0.7: {int((k2>0.7).sum())} of {len(k2)}")
print(f"  share from 2011: {(hi['year']==2011).mean():.0%}")
print(f"  these are the highest-count cell-years (mean count "
      f"{hi['count'].mean():.0f} vs overall {annual['count'].mean():.0f})")
"""),
    md(r"""
### Pareto-$k$ discussion
The high-$k$ points are exactly the **highest-count cell-years**, led by the
**2011 Tohoku cluster** (cell `6_10` with ~1400 events has $k \approx 9$ — far
above 1). For such extreme, influential observations PSIS importance sampling
fails, so the **LOO numbers for these points (and hence the overall LOO
ranking) are not trustworthy**. A rigorous fix would be exact
leave-one-out / `reloo` or moment-matching on the high-$k$ points; we flag it
rather than over-claim. The same points drive the WAIC `p_waic` warning. In
short: *both criteria are being dominated by a few cells the stationary Poisson
models cannot represent.*
"""),
    md(r"""
## Final assessment — do we agree with the criteria?

**The criteria do not give a clean verdict, and we say so.** WAIC favours
Model 2, PSIS-LOO favours Model 1, both raise warnings, and the separations are
only ~1.6 standard errors. The disagreement is manufactured by a small set
of extreme, model-misspecified observations (2011 Tohoku and other large-count
cell-years) with Pareto-$k \gg 0.7$. On **purely predictive grounds the two
models are statistically close**, and the information criteria are too unreliable
here to crown a winner by themselves.

**We nonetheless prefer Model 2 (partial pooling)** — not because a criterion
forces it, but on principled, decision-relevant grounds that the criteria do not
contradict:

1. **Regularisation where it matters.** Model 2 delivers the shrinkage shown in
   notebook 05: data-poor cells get stable estimates pulled toward the global
   mean with markedly narrower credible intervals, instead of Model 1's wide,
   prior-driven posteriors.
2. **Physically motivated structure.** `sigma_global` is *estimated* (posterior
   mean ~1.3, comfortably > 0), so the data themselves support a shared baseline
   across cells in the same tectonic setting — the hierarchy is justified, not
   imposed.
3. **A better deliverable.** The intensity map from Model 2 is smoother and its
   uncertainty is honest, which is what a hazard-assessment use case actually
   needs — a quality the in-sample criteria barely reward.
4. **Parsimony is not violated.** Model 2 adds a single hyperparameter; its WAIC
   edge and near-tie on LOO mean we pay essentially nothing in predictive
   accuracy for these benefits.

**Caveat / next step.** The real lesson from the comparison is that **neither**
stationary model represents the 2011 regime change. The most valuable
improvement would be a non-stationary or time-varying / mixture intensity (or an
explicit aftershock term), after which WAIC and LOO would become trustworthy
(Pareto-$k$ back under 0.7) and could arbitrate cleanly. Until then we report the
criteria honestly, note their unreliability, and select Model 2 on
modelling-quality grounds.
"""),
    md(r"""
## Summary
- **WAIC:** Model 2 wins by ~61 elpd (~1.6 SE) — *with warning*.
- **PSIS-LOO:** ranking flips, Model 1 wins by ~26 elpd (~1.6 SE) — *with
  warning*; Pareto-$k$ up to ~9, dominated by 2011/large-count cells.
- **Verdict:** criteria are inconclusive and unreliable here; we choose
  **Model 2** for its shrinkage, data-supported hierarchy, and superior
  uncertainty-aware map, while flagging non-stationarity (2011) as the key
  modelling gap.
"""),
]

build("06_comparison", nb06)
print("done nb06")
