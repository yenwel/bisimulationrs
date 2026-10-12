/// Distribution of a count `X ∈ {0, …, n}`, e.g. the number of plants in a
/// stage on a given day.
#[derive(Debug, Clone, PartialEq)]
pub struct CountDistribution {
    pmf: Vec<f64>,
}

impl CountDistribution {
    /// From probabilities `P(X = k)` for `k = 0..pmf.len()`; renormalised.
    pub fn from_pmf(mut pmf: Vec<f64>) -> Self {
        assert!(
            !pmf.is_empty(),
            "a count distribution needs at least one value"
        );
        for p in &mut pmf {
            *p = p.max(0.0);
        }
        let total: f64 = pmf.iter().sum();
        if total > 0.0 {
            pmf.iter_mut().for_each(|p| *p /= total);
        }
        Self { pmf }
    }

    /// `Binomial(n, p)`: the number of `n` independent copies that are in a
    /// set of states each copy occupies with probability `p`.
    pub fn binomial(n: u32, p: f64) -> Self {
        let p = p.clamp(0.0, 1.0);
        let n_usize = n as usize;
        if p == 0.0 || p == 1.0 {
            let mut pmf = vec![0.0; n_usize + 1];
            pmf[if p == 0.0 { 0 } else { n_usize }] = 1.0;
            return Self { pmf };
        }
        // log C(n, k) + k log p + (n − k) log(1 − p), via cumulative log-factorials.
        let mut ln_fact = vec![0.0; n_usize + 1];
        for k in 1..=n_usize {
            ln_fact[k] = ln_fact[k - 1] + (k as f64).ln();
        }
        let (lp, lq) = (p.ln(), (1.0 - p).ln());
        let pmf = (0..=n_usize)
            .map(|k| {
                (ln_fact[n_usize] - ln_fact[k] - ln_fact[n_usize - k]
                    + k as f64 * lp
                    + (n_usize - k) as f64 * lq)
                    .exp()
            })
            .collect();
        Self::from_pmf(pmf)
    }

    /// Largest possible count.
    pub fn max(&self) -> u32 {
        (self.pmf.len() - 1) as u32
    }

    pub fn prob(&self, k: u32) -> f64 {
        self.pmf.get(k as usize).copied().unwrap_or(0.0)
    }

    pub fn mean(&self) -> f64 {
        self.pmf.iter().enumerate().map(|(k, p)| k as f64 * p).sum()
    }

    /// Smallest `k` with `P(X ≤ k) ≥ q`.
    pub fn quantile(&self, q: f64) -> u32 {
        let mut cdf = 0.0;
        for (k, p) in self.pmf.iter().enumerate() {
            cdf += p;
            if cdf >= q - 1e-12 {
                return k as u32;
            }
        }
        self.max()
    }

    /// Central range holding at least `level` of the probability, e.g.
    /// `level = 0.95` gives the 2.5 % and 97.5 % quantiles.
    pub fn central_range(&self, level: f64) -> (u32, u32) {
        let tail = (1.0 - level) / 2.0;
        (self.quantile(tail), self.quantile(1.0 - tail))
    }

    /// Two-sided p-value of observing `k`: `min(1, 2 · min(P(X ≤ k), P(X ≥ k)))`.
    pub fn p_value(&self, k: u32) -> f64 {
        let k = k as usize;
        let below: f64 = self.pmf.iter().take(k + 1).sum();
        let above: f64 = self.pmf.iter().skip(k).sum();
        (2.0 * below.min(above)).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_moments_and_ranges() {
        let d = CountDistribution::binomial(20, 0.3);
        assert!((d.mean() - 6.0).abs() < 1e-9);
        assert!((d.prob(6) - 0.191_638_982_753_442_4).abs() < 1e-12);
        assert_eq!(d.central_range(0.95), (2, 10));
        assert!(d.p_value(6) > 0.9);
        assert!(d.p_value(14) < 0.001);
        assert_eq!(
            CountDistribution::binomial(5, 0.0).central_range(0.95),
            (0, 0)
        );
        assert_eq!(CountDistribution::binomial(5, 1.0).p_value(5), 1.0);
    }
}
