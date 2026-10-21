//! Sinkhorn Value Iteration (Calo, Jonsson, Neu, Schwartz, Segovia-Aguas,
//! NeurIPS 2024, Algorithm 1): an entropy-regularised alternative to the
//! exact solver in [`kantorovich`](crate::kantorovich), for *dense* chains
//! where an exact transport problem per state pair is expensive.
//!
//! The iteration alternates between the two sides. On an X-round it applies
//! the X-side Bellman–Sinkhorn operator `inner` times,
//!
//! ```text
//! V_X(xy, x′) ← c(xy) − (1/η)·log Σ_{y′} π(y′ | xy, x′)·exp(−η·γ·W_X(x′y′)),
//! W_X(x′y′) = Σ_{x″} P_X(x″|x′)·V_X(x′y′, x″),
//! ```
//!
//! then reweights the transition coupling,
//! `π(x′y′|xy) ∝ π(x′y′|xy)·exp(−η·γ·W_X(x′y′))`, normalised over `y′` and
//! scaled back to `P_X(x′|x)` (the `c(xy)` term cancels). Y-rounds swap the
//! roles. The last coupling is rounded to exact marginals (Altschuler, Weed
//! and Rigollet 2017, Algorithm 2) and evaluated exactly, so the reported
//! value is the cost of a genuine coupling: an **upper bound** on the
//! distance, never below it.
//!
//! The paper's accuracy guarantee assumes exact inner steps and on the
//! order of `324·log(|X||Y|) / ((1 − γ)³ ε²)` iterations for this cost scale
//! (about 3·10⁸ for `γ = 0.9`, `ε = 0.05`); the defaults use 1,000 rounds and
//! one inner step, so the bound is not guaranteed to be tight. A large `η`
//! converges in a few rounds but can stall on a suboptimal coupling
//! (multiplicative updates cannot revive mass driven to zero); a small `η`
//! improves slowly and steadily, which is the regime the theory covers.
//! Prefer the exact solver unless the chains are dense; the tests measure
//! the gap. The memory is `(|X|·|Y|)²`, so both chains must be small.

use crate::kantorovich::{Chain, MetricError};

/// Largest `|X|·|Y|` accepted (the coupling has its square as entries).
pub const MAX_SVI_PAIRS: usize = 1_024;

/// Parameters of [`distance_svi`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SviOptions {
    /// Regularisation strength, relative to the largest possible value
    /// `max c / (1 − γ)`: `η = eta / that`.
    pub eta: f64,
    /// Outer rounds `K`.
    pub iterations: usize,
    /// Bellman–Sinkhorn steps per round (`m`).
    pub inner: usize,
}

impl Default for SviOptions {
    fn default() -> Self {
        Self {
            eta: 1.0,
            iterations: 1_000,
            inner: 1,
        }
    }
}

/// An upper bound on the bisimulation distance between the initial states
/// of `a` and `b` (observations in `[0, 1]`) at discount `gamma`.
pub fn distance_svi(
    a: &Chain,
    b: &Chain,
    gamma: f64,
    options: SviOptions,
) -> Result<f64, MetricError> {
    let (m, n) = (a.rows.len(), b.rows.len());
    if m * n > MAX_SVI_PAIRS {
        return Err(MetricError::TooManyPairs(MAX_SVI_PAIRS));
    }
    if !(gamma > 0.0 && gamma < 1.0) {
        return Err(MetricError::Invalid(format!(
            "discount {gamma} not in (0, 1)"
        )));
    }
    let dense = |c: &Chain, k: usize| {
        let mut p = vec![vec![0.0; k]; k];
        for (s, row) in c.rows.iter().enumerate() {
            for &(t, w) in row {
                p[s][t] += w;
            }
        }
        p
    };
    let (px, py) = (dense(a, m), dense(b, n));
    let pair = |x: usize, y: usize| x * n + y;
    let mn = m * n;
    let cost: Vec<f64> = (0..mn)
        .map(|k| (1.0 - gamma) * (a.observation[k / n] - b.observation[k % n]).abs())
        .collect();
    let scale = cost.iter().copied().fold(0.0f64, f64::max) / (1.0 - gamma);
    let eta = options.eta / scale.max(1e-12);

    // π[xy][x′y′], starting from the independent coupling.
    let mut pi: Vec<Vec<f64>> = (0..mn)
        .map(|k| {
            let (x, y) = (k / n, k % n);
            (0..mn).map(|j| px[x][j / n] * py[y][j % n]).collect()
        })
        .collect();
    // V_X[xy][x′] and V_Y[xy][y′].
    let mut vx = vec![vec![0.0; m]; mn];
    let mut vy = vec![vec![0.0; n]; mn];

    for k in 0..options.iterations.max(1) {
        let x_side = k % 2 == 0;
        for _ in 0..options.inner.max(1) {
            // W(x′y′): expected value at the next pair, averaged over the
            // side's own successor.
            let w: Vec<f64> = (0..mn)
                .map(|j| {
                    let (x, y) = (j / n, j % n);
                    if x_side {
                        (0..m).map(|x2| px[x][x2] * vx[j][x2]).sum()
                    } else {
                        (0..n).map(|y2| py[y][y2] * vy[j][y2]).sum()
                    }
                })
                .collect();
            for xy in 0..mn {
                let (x, y) = (xy / n, xy % n);
                if x_side {
                    for x2 in 0..m {
                        let p = px[x][x2];
                        if p <= 0.0 {
                            continue;
                        }
                        let terms =
                            (0..n).map(|y2| (pi[xy][pair(x2, y2)] / p, gamma * w[pair(x2, y2)]));
                        vx[xy][x2] = (cost[xy] + soft_min(terms, eta)).min(scale);
                    }
                } else {
                    for y2 in 0..n {
                        let p = py[y][y2];
                        if p <= 0.0 {
                            continue;
                        }
                        let terms =
                            (0..m).map(|x2| (pi[xy][pair(x2, y2)] / p, gamma * w[pair(x2, y2)]));
                        vy[xy][y2] = (cost[xy] + soft_min(terms, eta)).min(scale);
                    }
                }
            }
        }
        // Coupling update against the refreshed values.
        let w: Vec<f64> = (0..mn)
            .map(|j| {
                let (x, y) = (j / n, j % n);
                if x_side {
                    (0..m).map(|x2| px[x][x2] * vx[j][x2]).sum()
                } else {
                    (0..n).map(|y2| py[y][y2] * vy[j][y2]).sum()
                }
            })
            .collect();
        for (xy, row) in pi.iter_mut().enumerate() {
            let (x, y) = (xy / n, xy % n);
            if x_side {
                for (x2, &p) in px[x].iter().enumerate() {
                    if p > 0.0 {
                        reweight(row, (0..n).map(|y2| pair(x2, y2)), &w, eta * gamma, p);
                    }
                }
            } else {
                for (y2, &p) in py[y].iter().enumerate() {
                    if p > 0.0 {
                        reweight(row, (0..m).map(|x2| pair(x2, y2)), &w, eta * gamma, p);
                    }
                }
            }
        }
    }

    // Round every conditional coupling to the exact marginals, then
    // evaluate the resulting coupling exactly.
    for (xy, row) in pi.iter_mut().enumerate() {
        round(row, &px[xy / n], &py[xy % n], m, n);
    }
    let mut v = vec![0.0; mn];
    for _ in 0..10_000_000 {
        let mut delta = 0.0f64;
        for xy in 0..mn {
            let next = cost[xy] + gamma * pi[xy].iter().zip(&v).map(|(p, w)| p * w).sum::<f64>();
            delta = delta.max((next - v[xy]).abs());
            v[xy] = next;
        }
        if delta <= 1e-12 {
            break;
        }
    }
    Ok(v[pair(a.initial, b.initial)])
}

/// `−(1/η) log Σ wᵢ exp(−η vᵢ)` for weights summing to 1, stabilised.
fn soft_min(terms: impl Iterator<Item = (f64, f64)>, eta: f64) -> f64 {
    let terms: Vec<(f64, f64)> = terms.filter(|&(w, _)| w > 0.0).collect();
    let lo = terms.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
    if lo.is_infinite() {
        return 0.0;
    }
    let s: f64 = terms
        .iter()
        .map(|&(w, v)| w * (-eta * (v - lo)).exp())
        .sum();
    lo - s.ln() / eta
}

/// `π[j] ∝ π[j]·exp(−k·w[j])` over `group`, rescaled to total `mass`.
fn reweight(
    pi: &mut [f64],
    group: impl Iterator<Item = usize> + Clone,
    w: &[f64],
    k: f64,
    mass: f64,
) {
    let lo = group.clone().map(|j| w[j]).fold(f64::INFINITY, f64::min);
    let total: f64 = group
        .clone()
        .map(|j| pi[j] * (-k * (w[j] - lo)).exp())
        .sum();
    if total <= 0.0 {
        return;
    }
    for j in group {
        pi[j] = mass * pi[j] * (-k * (w[j] - lo)).exp() / total;
    }
}

/// Rounds `f` (an `m × n` matrix, flattened) onto the couplings of `r` and
/// `c` (Altschuler, Weed, Rigollet 2017, Algorithm 2).
fn round(f: &mut [f64], r: &[f64], c: &[f64], m: usize, n: usize) {
    for x in 0..m {
        let s: f64 = f[x * n..(x + 1) * n].iter().sum();
        let k = if s > 0.0 { (r[x] / s).min(1.0) } else { 0.0 };
        f[x * n..(x + 1) * n].iter_mut().for_each(|v| *v *= k);
    }
    for y in 0..n {
        let s: f64 = (0..m).map(|x| f[x * n + y]).sum();
        let k = if s > 0.0 { (c[y] / s).min(1.0) } else { 0.0 };
        (0..m).for_each(|x| f[x * n + y] *= k);
    }
    let er: Vec<f64> = (0..m)
        .map(|x| r[x] - f[x * n..(x + 1) * n].iter().sum::<f64>())
        .collect();
    let ec: Vec<f64> = (0..n)
        .map(|y| c[y] - (0..m).map(|x| f[x * n + y]).sum::<f64>())
        .collect();
    let total: f64 = er.iter().sum();
    if total > 0.0 {
        for x in 0..m {
            for y in 0..n {
                f[x * n + y] += er[x] * ec[y] / total;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_restores_marginals() {
        let mut f = vec![0.5, 0.1, 0.0, 0.3];
        let (r, c) = ([0.4, 0.6], [0.7, 0.3]);
        round(&mut f, &r, &c, 2, 2);
        assert!((f[0] + f[1] - 0.4).abs() < 1e-12 && (f[2] + f[3] - 0.6).abs() < 1e-12);
        assert!((f[0] + f[2] - 0.7).abs() < 1e-12 && (f[1] + f[3] - 0.3).abs() < 1e-12);
        assert!(f.iter().all(|&v| v >= 0.0));
    }
}
