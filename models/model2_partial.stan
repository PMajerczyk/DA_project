// Model 2 — Hierarchical Poisson with partial pooling.
//
// Same likelihood as Model 1, but the hyperparameters mu_global and
// sigma_global are now FREE parameters estimated from the data. Cells "borrow
// strength" from each other through the shared hyperprior: cells with little
// data are shrunk towards the global mean (shrinkage).
//
//   count[n] ~ Poisson(lambda[cell[n]])
//   log(lambda[c]) = alpha[c]
//   alpha[c] ~ Normal(mu_global, sigma_global)   // partial pooling
//   mu_global ~ Normal(2, 1)
//   sigma_global ~ HalfNormal(0, 1)              // KEY: estimated, not fixed
//
// Parameterization: this dataset is INFORMATIVE (most cells have many cell-years
// and large counts), so the per-cell log-intensity alpha is tightly constrained
// by the likelihood. In that regime the CENTERED parameterization
// (alpha ~ Normal(mu_global, sigma_global)) mixes better than the non-centered
// one; non-centering only helps when the data are weak relative to the prior.
data {
  int<lower=0> N;                 // number of observations (cell x year)
  int<lower=0> C;                 // number of cells
  array[N] int<lower=1> cell_id;  // cell index for each observation (1-based)
  array[N] int<lower=0> count;    // observed earthquake count
  real mu_prior_mean;             // externally-derived prior centre for log-intensity (~1.8)
}
parameters {
  real mu_global;                 // hyperprior mean (log-intensity)
  real<lower=0> sigma_global;     // hyperprior sd — KEY estimated parameter
  vector[C] alpha;                // per-cell log-intensity (centered)
}
model {
  mu_global ~ normal(mu_prior_mean, 1);     // centre from external Japan rate / grid geometry
  sigma_global ~ normal(0, 1);              // half-normal via <lower=0> constraint
  alpha ~ normal(mu_global, sigma_global);  // partial pooling (centered)
  count ~ poisson_log(alpha[cell_id]);
}
generated quantities {
  array[N] int count_pred;
  vector[N] log_lik;
  vector[C] lambda = exp(alpha);                // intensity per cell
  for (n in 1:N) {
    count_pred[n] = poisson_log_rng(alpha[cell_id[n]]);
    log_lik[n] = poisson_log_lpmf(count[n] | alpha[cell_id[n]]);
  }
}
