use std::hash::Hash;

use bisim_core::{Real, StateId, Wlts};

/// Continuous-time Markov chain in sparse row form. Times are in seconds.
#[derive(Debug, Clone)]
pub struct Ctmc {
    /// Off-diagonal rates `q(s, t)`, summed over labels, `t ≠ s`.
    rows: Vec<Vec<(usize, f64)>>,
    /// Exit rate `−q(s, s)` of every state.
    exit: Vec<f64>,
    initial: Option<StateId>,
}

impl Ctmc {
    /// Forgets labels and sums rates. Self-loops do not change the state and
    /// are dropped.
    pub fn from_wlts<L: Eq + Hash + Clone>(sys: &Wlts<L, Real>) -> Self {
        let n = sys.num_states();
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut exit = vec![0.0; n];
        for (s, _, t, w) in sys.transitions() {
            if s == t {
                continue;
            }
            let row = &mut rows[s.index()];
            match row.iter_mut().find(|(j, _)| *j == t.index()) {
                Some((_, r)) => *r += w.value(),
                None => row.push((t.index(), w.value())),
            }
            exit[s.index()] += w.value();
        }
        Self {
            rows,
            exit,
            initial: sys.initial(),
        }
    }

    pub fn num_states(&self) -> usize {
        self.exit.len()
    }

    /// Point mass on the initial state.
    ///
    /// # Panics
    /// If the source system had no initial state.
    pub fn initial_distribution(&self) -> Vec<f64> {
        let mut p = vec![0.0; self.num_states()];
        p[self.initial.expect("CTMC has no initial state").index()] = 1.0;
        p
    }

    /// Uniformisation rate `Λ ≥ max exit rate` (strictly larger, so the
    /// uniformised chain is aperiodic).
    fn uniformisation_rate(&self) -> f64 {
        let max = self.exit.iter().copied().fold(0.0, f64::max);
        if max == 0.0 { 1.0 } else { max * 1.05 }
    }

    /// One step of the uniformised DTMC `P = I + Q / Λ`: returns `p · P`.
    fn step(&self, p: &[f64], lambda: f64) -> Vec<f64> {
        let mut next: Vec<f64> = p
            .iter()
            .zip(&self.exit)
            .map(|(pi, e)| pi * (1.0 - e / lambda))
            .collect();
        for (s, row) in self.rows.iter().enumerate() {
            if p[s] == 0.0 {
                continue;
            }
            for &(t, r) in row {
                next[t] += p[s] * r / lambda;
            }
        }
        next
    }

    /// Transient distribution at time `t` seconds from `p0`, by
    /// uniformisation with Poisson weights truncated at total mass `1 − ε`.
    ///
    /// Long horizons are split into slices with `Λ·h ≤ 20` so the Poisson
    /// weights never underflow.
    pub fn transient(&self, p0: &[f64], t: f64, epsilon: f64) -> Vec<f64> {
        assert_eq!(p0.len(), self.num_states());
        assert!(t >= 0.0, "time must be non-negative");
        let lambda = self.uniformisation_rate();
        let slices = (lambda * t / 20.0).ceil().max(1.0) as usize;
        let h = t / slices as f64;
        let mut p = p0.to_vec();
        for _ in 0..slices {
            p = self.transient_slice(&p, lambda, h, epsilon / slices as f64);
        }
        p
    }

    fn transient_slice(&self, p0: &[f64], lambda: f64, h: f64, epsilon: f64) -> Vec<f64> {
        let mean = lambda * h;
        let mut weight = (-mean).exp();
        let mut mass = weight;
        let mut term = p0.to_vec();
        let mut out: Vec<f64> = term.iter().map(|x| x * weight).collect();
        let mut k = 0u32;
        while mass < 1.0 - epsilon && k < 10_000 {
            k += 1;
            term = self.step(&term, lambda);
            weight *= mean / f64::from(k);
            mass += weight;
            for (o, x) in out.iter_mut().zip(&term) {
                *o += weight * x;
            }
        }
        out
    }

    /// Time-bounded reachability `P[F<=t target]` from `p0`: makes target
    /// states absorbing, then reports the transient mass inside them at
    /// time `t` (uniformisation, as in [`Self::transient`]).
    pub fn reach_bounded(&self, p0: &[f64], targets: &[bool], t: f64, epsilon: f64) -> f64 {
        assert_eq!(p0.len(), self.num_states());
        assert_eq!(targets.len(), self.num_states());
        let mut absorbing = self.clone();
        for (i, is_target) in targets.iter().enumerate() {
            if *is_target {
                absorbing.rows[i].clear();
                absorbing.exit[i] = 0.0;
            }
        }
        absorbing
            .transient(p0, t, epsilon)
            .into_iter()
            .zip(targets)
            .filter(|(_, is_target)| **is_target)
            .map(|(p, _)| p)
            .sum()
    }

    /// The embedded jump chain as a stochastic matrix in sparse row form:
    /// `P(s, t) = Q(s, t) / exit(s)` (0 when `exit(s) = 0`). Unbounded
    /// CTMC reachability reduces to DTMC reachability on this chain
    /// (see [`unbounded_reach`](super::unbounded_reach)).
    pub fn embedded(&self) -> Vec<Vec<(usize, f64)>> {
        self.rows
            .iter()
            .zip(&self.exit)
            .enumerate()
            .map(|(s, (row, e))| {
                if *e == 0.0 {
                    // Absorbing: DTMC self-loop.
                    vec![(s, 1.0)]
                } else {
                    row.iter().map(|&(t, r)| (t, r / e)).collect()
                }
            })
            .collect()
    }

    /// Stationary distribution by power iteration on the uniformised chain.
    /// Meaningful for irreducible chains; otherwise it depends on the start.
    pub fn steady_state(&self, tolerance: f64, max_iterations: usize) -> Vec<f64> {
        let n = self.num_states();
        let lambda = self.uniformisation_rate();
        let mut p = vec![1.0 / n as f64; n];
        for _ in 0..max_iterations {
            let next = self.step(&p, lambda);
            let diff: f64 = next.iter().zip(&p).map(|(a, b)| (a - b).abs()).sum();
            p = next;
            if diff < tolerance {
                break;
            }
        }
        let total: f64 = p.iter().sum();
        p.iter().map(|x| x / total).collect()
    }
}
