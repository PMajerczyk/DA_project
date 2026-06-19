// Model 1 — No-pooling Poisson with log link.
//
// Each grid cell c has its own log-intensity alpha[c]. The alphas share a
// common prior Normal(mu, sigma) but sigma is FIXED (data input), so there is
// NO information sharing between cells beyond the fixed prior -> "no pooling".
//
//   count[n] ~ Poisson(lambda[cell[n]])
//   log(lambda[c]) = alpha[c]
//   alpha[c] ~ Normal(mu, sigma)      // sigma fixed -> independent cells
//
// poisson_log uses the log rate directly (lambda = exp(alpha)), which keeps the
// rate positive without abs() and is numerically stable.
data {
  int<lower=0> N;                 // number of observations (cell x year)
  int<lower=0> C;                 // number of cells
  array[N] int<lower=1> cell_id;  // cell index for each observation (1-based)
  array[N] int<lower=0> count;    // observed earthquake count
  real mu_prior_mean;             // prior mean for alpha (log-intensity)
  real<lower=0> sigma_fixed;      // FIXED prior sd for alpha (no pooling)
}
parameters {
  vector[C] alpha;                // per-cell log-intensity
}
model {
  alpha ~ normal(mu_prior_mean, sigma_fixed);   // fixed-scale, independent
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
