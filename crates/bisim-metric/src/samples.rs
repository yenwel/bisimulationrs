//! The bisimulation distance estimated from samples only, with no model:
//! the stochastic primal–dual method of Calo, Jonsson, Neu, Schwartz and
//! Segovia-Aguas, "Distances for Markov chains from sample streams"
//! (NeurIPS 2025, Algorithm 2 and eq. 11).
//!
//! Each step consumes one observed transition `(x, x′)` of the first system
//! and one `(y, y′)` of the second, for example two consecutive weekly
//! inspections of two plots, each mapped to a small set of states. The
//! method maintains
//!
//! - `μ(x, y, x′, y′)`: a joint occupancy coupling (entropic mirror descent);
//! - `λ_X(y | x)`, `λ_Y(x | y)`: how states of one system are matched to the
//!   other's (they read out as a state-matching map);
//! - dual variables `α_X`, `α_Y`, `V` (projected gradient ascent);
//!
//! and estimates the distance as `⟨μ̄, c⟩` for the average `μ̄` of the
//! iterates. With the ground cost `c(x, y) = |r(x) − r(y)|`, this is on the
//! same scale as [`distance`](crate::distance) (whose cost carries the
//! factor `1 − γ`).
//!
//! The guarantee (Theorem 1) assumes each pair is an independent draw from
//! the discounted occupancy (restart after a geometric time; see
//! [`occupancy_sample`]); consecutive pairs of one long stream are the
//! paper's practical variant, outside the guarantee.
//!
//! **It needs many samples.** The default step sizes are the analysed ones
//! ([`StepSizes::Theorem`], with the steps so far in place of `K`), and the
//! error shrinks like `1/√K` with a large constant: measured on two
//! identical 4-state rings at `γ = 0.8`, the estimate (truth 0) was 0.29
//! after 10⁴ draws, 0.20 after 10⁵, 0.067 after 10⁶ and 0.033 after
//! 4·10⁶. A constant-step schedule like the paper's experiments
//! ([`StepSizes::Practical`]) moves faster early but levelled off around
//! 0.2–0.3 in the same test, and at 3·10⁴ draws it did not reliably rank
//! nearby candidates either; use it with care. A step costs
//! `Θ(|X|²·|Y|²)`, so the state spaces must be small.

use crate::kantorovich::MetricError;

/// Largest `|X|·|Y|` accepted (the coupling has its square as entries).
pub const MAX_SAMPLE_PAIRS: usize = 400;

/// Step sizes of [`SampleDistance`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StepSizes {
    /// Theorem 1, with the number of steps so far in place of `K`: the
    /// analysed choice, and the default; it converges, slowly.
    #[default]
    Theorem,
    /// The schedule of the paper's experiments (Appendix F): primal steps
    /// `eta0 / √(1 + decay·k)` for μ and λ, a constant dual step `beta`.
    Practical { eta0: f64, decay: f64, beta: f64 },
}

/// The streaming estimator.
#[derive(Debug, Clone)]
pub struct SampleDistance {
    steps_sizes: StepSizes,
    m: usize,
    n: usize,
    gamma: f64,
    /// `c(x, y)`, in `[0, 1]`.
    cost: Vec<f64>,
    start: (usize, usize),
    /// `μ[((x·n + y)·m + x′)·n + y′]`.
    mu: Vec<f64>,
    mu_sum: Vec<f64>,
    /// `λ_X[x·n + y]`, rows over `y` sum to 1.
    lambda_x: Vec<f64>,
    lambda_x_sum: Vec<f64>,
    /// `λ_Y[y·m + x]`.
    lambda_y: Vec<f64>,
    /// `α_X[(x·m + x′)·n + y]`.
    alpha_x: Vec<f64>,
    /// `α_Y[(x·n + y)·n + y′]`.
    alpha_y: Vec<f64>,
    /// `V[x·n + y]`.
    v: Vec<f64>,
    steps: usize,
}

impl SampleDistance {
    /// An estimator for systems with `m` and `n` states, ground cost
    /// `cost(x, y) ∈ [0, 1]`, discount `gamma` and initial states `start`.
    pub fn new(
        m: usize,
        n: usize,
        cost: impl Fn(usize, usize) -> f64,
        gamma: f64,
        start: (usize, usize),
    ) -> Result<Self, MetricError> {
        if m == 0 || n == 0 || m * n > MAX_SAMPLE_PAIRS {
            return Err(MetricError::TooManyPairs(MAX_SAMPLE_PAIRS));
        }
        if !(gamma > 0.0 && gamma < 1.0) {
            return Err(MetricError::Invalid(format!(
                "discount {gamma} not in (0, 1)"
            )));
        }
        if start.0 >= m || start.1 >= n {
            return Err(MetricError::Invalid("initial state out of range".into()));
        }
        let cost: Vec<f64> = (0..m * n).map(|k| cost(k / n, k % n)).collect();
        if cost.iter().any(|c| !(0.0..=1.0).contains(c)) {
            return Err(MetricError::Invalid("costs must be in [0, 1]".into()));
        }
        let size = m * n * m * n;
        Ok(Self {
            steps_sizes: StepSizes::default(),
            m,
            n,
            gamma,
            cost,
            start,
            mu: vec![1.0 / size as f64; size],
            mu_sum: vec![0.0; size],
            lambda_x: vec![1.0 / n as f64; m * n],
            lambda_x_sum: vec![0.0; m * n],
            lambda_y: vec![1.0 / m as f64; n * m],
            alpha_x: vec![0.0; m * m * n],
            alpha_y: vec![0.0; m * n * n],
            v: vec![0.0; m * n],
            steps: 0,
        })
    }

    /// Uses `sizes` instead of the default practical schedule.
    pub fn with_step_sizes(mut self, sizes: StepSizes) -> Self {
        self.steps_sizes = sizes;
        self
    }

    /// One step with an observed transition `x → x′` of the first system
    /// and `y → y′` of the second.
    ///
    /// # Panics
    /// If a state is out of range.
    pub fn push(&mut self, (x_k, x_next): (usize, usize), (y_k, y_next): (usize, usize)) {
        let (m, n, g) = (self.m, self.n, self.gamma);
        assert!(
            x_k < m && x_next < m && y_k < n && y_next < n,
            "state out of range"
        );
        self.steps += 1;
        let k = self.steps as f64;
        let (mf, nf) = (m as f64, n as f64);
        let one_g = 1.0 - g;
        let (eta, eta_x, eta_y, beta_x, beta_y, beta) = match self.steps_sizes {
            // Theorem 1, with k in place of K.
            StepSizes::Theorem => (
                ((mf * mf * nf * nf).ln().max(1.0) * one_g * one_g / k).sqrt(),
                (mf * nf.ln().max(1.0) * one_g * one_g / k).sqrt(),
                (nf * mf.ln().max(1.0) * one_g * one_g / k).sqrt(),
                (mf * mf * nf / (one_g * one_g * k)).sqrt(),
                (mf * nf * nf / (one_g * one_g * k)).sqrt(),
                (mf * nf / (one_g * one_g * k)).sqrt(),
            ),
            StepSizes::Practical { eta0, decay, beta } => {
                let e = eta0 / (1.0 + decay * k).sqrt();
                (e, e, e, beta, beta, beta)
            }
        };
        let idx = |x: usize, y: usize, x2: usize, y2: usize| ((x * n + y) * m + x2) * n + y2;

        // Gradients at the current iterate (all from the same point).
        let mut out_flow = vec![0.0; m * n]; // Σ_{x′y′} μ(x, y, x′, y′)
        let mut in_flow = vec![0.0; m * n]; // Σ_{x̂ŷ} μ(x̂, ŷ, x, y)
        let mut sum_y2 = vec![0.0; m * m * n]; // Σ_{y′} μ(x,y,x′,y′) at [(x·m+x′)·n+y]
        let mut sum_x2 = vec![0.0; m * n * n]; // Σ_{x′} μ(x,y,x′,y′) at [(x·n+y)·n+y′]
        for x in 0..m {
            for y in 0..n {
                for x2 in 0..m {
                    for y2 in 0..n {
                        let w = self.mu[idx(x, y, x2, y2)];
                        out_flow[x * n + y] += w;
                        in_flow[x2 * n + y2] += w;
                        sum_y2[(x * m + x2) * n + y] += w;
                        sum_x2[(x * n + y) * n + y2] += w;
                    }
                }
            }
        }
        let start = self.start.0 * n + self.start.1;

        // Primal: μ by entropic mirror descent.
        let mut max_exp = f64::NEG_INFINITY;
        let mut logs = vec![0.0; self.mu.len()];
        for x in 0..m {
            for y in 0..n {
                for x2 in 0..m {
                    for y2 in 0..n {
                        let grad = self.cost[x * n + y]
                            - self.alpha_x[(x * m + x2) * n + y]
                            - self.alpha_y[(x * n + y) * n + y2]
                            + g * self.v[x2 * n + y2]
                            - self.v[x * n + y];
                        let i = idx(x, y, x2, y2);
                        logs[i] = self.mu[i].max(1e-300).ln() - eta * grad;
                        max_exp = max_exp.max(logs[i]);
                    }
                }
            }
        }
        let total: f64 = logs.iter().map(|l| (l - max_exp).exp()).sum();
        let new_mu: Vec<f64> = logs.iter().map(|l| (l - max_exp).exp() / total).collect();

        // λ_X(·|X_k) and λ_Y(·|Y_k): only the sampled row has a gradient.
        let row = &mut self.lambda_x[x_k * n..(x_k + 1) * n];
        let grads: Vec<f64> = (0..n)
            .map(|y| self.alpha_x[(x_k * m + x_next) * n + y])
            .collect();
        mirror_step(row, &grads, eta_x);
        let row = &mut self.lambda_y[y_k * m..(y_k + 1) * m];
        let grads: Vec<f64> = (0..m)
            .map(|x| self.alpha_y[(x * n + y_k) * n + y_next])
            .collect();
        mirror_step(row, &grads, eta_y);

        // Duals: projected steps against the gradients at the old μ, λ.
        let (ra, rv) = (6.0 / one_g, 2.0 / one_g);
        for x in 0..m {
            for x2 in 0..m {
                for y in 0..n {
                    let i = (x * m + x2) * n + y;
                    let sampled = if (x, x2) == (x_k, x_next) {
                        self.lambda_x[x * n + y]
                    } else {
                        0.0
                    };
                    let grad = sum_y2[i] - sampled;
                    self.alpha_x[i] = (self.alpha_x[i] - beta_x * grad).clamp(-ra, ra);
                }
            }
        }
        for x in 0..m {
            for y in 0..n {
                for y2 in 0..n {
                    let i = (x * n + y) * n + y2;
                    let sampled = if (y, y2) == (y_k, y_next) {
                        self.lambda_y[y * m + x]
                    } else {
                        0.0
                    };
                    let grad = sum_x2[i] - sampled;
                    self.alpha_y[i] = (self.alpha_y[i] - beta_y * grad).clamp(-ra, ra);
                }
            }
        }
        for xy in 0..m * n {
            let nu0 = if xy == start { 1.0 } else { 0.0 };
            let grad = out_flow[xy] - one_g * nu0 - g * in_flow[xy];
            self.v[xy] = (self.v[xy] - beta * grad).clamp(-rv, rv);
        }

        self.mu = new_mu;
        for (s, w) in self.mu_sum.iter_mut().zip(&self.mu) {
            *s += w;
        }
        for (s, w) in self.lambda_x_sum.iter_mut().zip(&self.lambda_x) {
            *s += w;
        }
    }

    /// Steps taken.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The estimate `⟨μ̄, c⟩` (0 before the first step).
    pub fn estimate(&self) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        let (m, n) = (self.m, self.n);
        let mut total = 0.0;
        for (i, &w) in self.mu_sum.iter().enumerate() {
            let xy = i / (m * n);
            total += w * self.cost[xy];
        }
        total / self.steps as f64
    }

    /// For each state `x` of the first system, its most likely match in the
    /// second (from the averaged `λ_X`).
    pub fn matching(&self) -> Vec<usize> {
        let n = self.n;
        (0..self.m)
            .map(|x| {
                (0..n)
                    .max_by(|&a, &b| {
                        self.lambda_x_sum[x * n + a].total_cmp(&self.lambda_x_sum[x * n + b])
                    })
                    .unwrap_or(0)
            })
            .collect()
    }
}

/// `p ∝ p·exp(−η·g)`, renormalised.
fn mirror_step(p: &mut [f64], g: &[f64], eta: f64) {
    let logs: Vec<f64> = p
        .iter()
        .zip(g)
        .map(|(&w, &gi)| w.max(1e-300).ln() - eta * gi)
        .collect();
    let hi = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = logs.iter().map(|l| (l - hi).exp()).sum();
    for (w, l) in p.iter_mut().zip(&logs) {
        *w = (l - hi).exp() / total;
    }
}

/// One draw `(state, next state)` from the discounted occupancy of a chain
/// with transition rows `rows`, started in `start` (Appendix B.4): run for
/// a geometric number of steps `G` with mean `γ/(1 − γ)` and return the
/// transition taken at step `G`. `uniform` supplies numbers in `[0, 1)`.
pub fn occupancy_sample(
    rows: &[Vec<(usize, f64)>],
    start: usize,
    gamma: f64,
    mut uniform: impl FnMut() -> f64,
) -> (usize, usize) {
    let step = |s: usize, u: f64| {
        let mut acc = 0.0;
        for &(t, p) in &rows[s] {
            acc += p;
            if u < acc {
                return t;
            }
        }
        rows[s].last().map_or(s, |e| e.0)
    };
    let mut s = start;
    while uniform() < gamma {
        s = step(s, uniform());
    }
    (s, step(s, uniform()))
}
