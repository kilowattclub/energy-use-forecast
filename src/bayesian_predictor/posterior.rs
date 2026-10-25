//! Normal-Inverse-Gamma Bayesian linear regression with discounted evidence.
//!
//! The posterior is the prior plus every observation, each weighted by
//! 0.5^(age / half-life). Old evidence fades, so the model follows a changing
//! household, and weights that stop receiving evidence (a weekday not seen for
//! a while) return to their prior rather than drifting or growing uncertain
//! without bound.
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Prior beliefs. Weight variances are relative to the noise variance σ².
#[derive(Clone, Debug)]
pub(crate) struct Prior {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    /// Age at which evidence counts half; `f64::INFINITY` never discounts.
    pub half_life_days: f64,
    /// Inverse-Gamma shape and scale for σ².
    pub shape: f64,
    pub scale: f64,
}

/// Discounted sufficient statistics of the observations seen so far.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Posterior {
    /// Σ w x xᵀ, row-major.
    information: Vec<f64>,
    /// Σ w x y.
    targets: Vec<f64>,
    /// Σ w y².
    squares: f64,
    /// Σ w.
    weight: f64,
    /// Derived from the fields above and the prior; rebuilt when needed.
    #[serde(skip)]
    solved: OnceLock<Solved>,
}

#[derive(Clone, Debug)]
struct Solved {
    mean: Vec<f64>,
    /// Lower-triangular L with LLᵀ the posterior precision, row-major.
    cholesky: Vec<f64>,
    noise_variance: f64,
}

impl Posterior {
    /// No evidence yet: predictions come from the prior alone.
    pub fn empty(dim: usize) -> Self {
        Self {
            information: vec![0.0; dim * dim],
            targets: vec![0.0; dim],
            squares: 0.0,
            weight: 0.0,
            solved: OnceLock::new(),
        }
    }

    fn dim(&self) -> usize {
        self.targets.len()
    }

    /// Whether this posterior is well formed for `prior`, e.g. after loading.
    pub fn matches(&self, prior: &Prior) -> bool {
        let n = prior.mean.len();
        self.targets.len() == n
            && self.information.len() == n * n
            && self
                .information
                .iter()
                .chain(&self.targets)
                .chain([&self.squares, &self.weight])
                .all(|v| v.is_finite())
            && self.squares >= 0.0
            && self.weight >= 0.0
    }

    /// Age all evidence by `days`.
    pub fn discount(&mut self, prior: &Prior, days: f64) {
        if days <= 0.0 {
            return;
        }
        let keep = 0.5_f64.powf(days / prior.half_life_days);
        for v in self.information.iter_mut().chain(&mut self.targets) {
            *v *= keep;
        }
        self.squares *= keep;
        self.weight *= keep;
        self.solved = OnceLock::new();
    }

    /// Learn that `x · weights = y` with noise variance σ² / `weight`.
    pub fn update(&mut self, x: &[f64], y: f64, weight: f64) {
        let n = self.dim();
        for i in 0..n {
            self.targets[i] += weight * x[i] * y;
            for j in 0..n {
                self.information[i * n + j] += weight * x[i] * x[j];
            }
        }
        self.squares += weight * y * y;
        self.weight += weight;
        self.solved = OnceLock::new();
    }

    pub fn predict(&self, prior: &Prior, x: &[f64]) -> f64 {
        dot(x, &self.solved(prior).mean)
    }

    /// xᵀPx: variance of `predict(x)` in units of σ².
    pub fn spread(&self, prior: &Prior, x: &[f64]) -> f64 {
        let z = forward(&self.solved(prior).cholesky, x);
        dot(&z, &z)
    }

    /// Expected noise variance σ².
    pub fn noise_variance(&self, prior: &Prior) -> f64 {
        self.solved(prior).noise_variance
    }

    fn solved(&self, prior: &Prior) -> &Solved {
        self.solved.get_or_init(|| self.solve(prior))
    }

    fn solve(&self, prior: &Prior) -> Solved {
        let n = self.dim();
        let mut precision = self.information.clone();
        let mut targets = self.targets.clone();
        for i in 0..n {
            precision[i * n + i] += 1.0 / prior.variance[i];
            targets[i] += prior.mean[i] / prior.variance[i];
        }
        let cholesky = cholesky(precision, n);
        let mean = backward(&cholesky, &forward(&cholesky, &targets));
        // The residual sum of squares; never negative, but rounding can make it so.
        let prior_squares = prior
            .mean
            .iter()
            .zip(&prior.variance)
            .map(|(m, v)| m * m / v)
            .sum::<f64>();
        let residual = (self.squares + prior_squares - dot(&mean, &targets)).max(0.0);
        let shape = prior.shape + 0.5 * self.weight;
        let scale = prior.scale + 0.5 * residual;
        Solved {
            mean,
            cholesky,
            noise_variance: scale / (shape - 1.0),
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Factor a symmetric positive-definite matrix as LLᵀ.
fn cholesky(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    for j in 0..n {
        let diagonal = a[j * n + j] - (0..j).map(|k| a[j * n + k].powi(2)).sum::<f64>();
        // The prior keeps the precision positive definite; this guards rounding.
        let diagonal = diagonal.max(f64::MIN_POSITIVE).sqrt();
        a[j * n + j] = diagonal;
        for i in j + 1..n {
            let sum = (0..j).map(|k| a[i * n + k] * a[j * n + k]).sum::<f64>();
            a[i * n + j] = (a[i * n + j] - sum) / diagonal;
        }
        for k in j + 1..n {
            a[j * n + k] = 0.0;
        }
    }
    a
}

/// Solve Lz = b.
fn forward(l: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut z = vec![0.0; n];
    for i in 0..n {
        let sum = (0..i).map(|k| l[i * n + k] * z[k]).sum::<f64>();
        z[i] = (b[i] - sum) / l[i * n + i];
    }
    z
}

/// Solve Lᵀx = z.
fn backward(l: &[f64], z: &[f64]) -> Vec<f64> {
    let n = z.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum::<f64>();
        x[i] = (z[i] - sum) / l[i * n + i];
    }
    x
}

#[cfg(test)]
#[path = "../../tests/unit/bayesian_posterior.rs"]
mod tests;
