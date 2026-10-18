use std::hash::Hash;

use bisim_core::{Real, StateId, Wlts};

use crate::reach::unbounded_reach;

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
            rows[s.index()].push((t.index(), w.value()));
            exit[s.index()] += w.value();
        }
        // Merge parallel transitions (same target, different labels) by
        // sorting, instead of searching the row for every transition.
        for row in &mut rows {
            row.sort_by_key(|&(t, _)| t);
            row.dedup_by(|next, kept| {
                let same = next.0 == kept.0;
                if same {
                    kept.1 += next.1;
                }
                same
            });
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
        let mut term = p0.to_vec();
        let mut out: Vec<f64> = term.iter().map(|x| x * weight).collect();
        let mut k = 0u32;
        // Stop once past the mode with the unseen tail below `epsilon`: the
        // terms shrink at least geometrically with ratio `mean / (k + 1)`.
        // (Summing the weights up to `1 − ε` fails for tiny `ε`: the sum
        // rounds to just below 1 and never gets there.)
        while k < 10_000 {
            let ratio = mean / (f64::from(k) + 1.0);
            if f64::from(k) >= mean && weight * ratio / (1.0 - ratio) < epsilon {
                break;
            }
            k += 1;
            term = self.step(&term, lambda);
            weight *= mean / f64::from(k);
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

    /// Long-run distribution from the initial state (uniform over all states
    /// if there is none).
    ///
    /// The chain is split into its bottom strongly connected components, the
    /// only places probability can stay forever. Each is solved on its own
    /// (exactly by the GTH algorithm up to 400 states, else by
    /// Gauss–Seidel until the balance residual `‖πQ‖₁` is below `tolerance`
    /// relative to the total flow, or `max_iterations` sweeps), and weighted
    /// by the probability to end up in it. Transient states get 0. For an
    /// irreducible chain this is its stationary distribution.
    pub fn steady_state(&self, tolerance: f64, max_iterations: usize) -> Vec<f64> {
        let n = self.num_states();
        if n == 0 {
            return Vec::new();
        }
        let start = match self.initial {
            Some(i) => {
                let mut p = vec![0.0; n];
                p[i.index()] = 1.0;
                p
            }
            None => vec![1.0 / n as f64; n],
        };
        let mut out = vec![0.0; n];
        let embedded = self.embedded();
        for component in self.bottom_components() {
            let local = self.stationary_within(&component, tolerance, max_iterations);
            let weight = if component.len() == n {
                1.0
            } else {
                let mut targets = vec![false; n];
                for &s in &component {
                    targets[s] = true;
                }
                let reach = unbounded_reach(&embedded, &targets, 1e-13, 10_000_000);
                start.iter().zip(&reach).map(|(p, r)| p * r).sum()
            };
            for (&s, pi) in component.iter().zip(&local) {
                out[s] += weight * pi;
            }
        }
        out
    }

    /// Bottom strongly connected components (no transition leaves them),
    /// by an iterative Tarjan search.
    fn bottom_components(&self) -> Vec<Vec<usize>> {
        let n = self.num_states();
        let (mut index, mut low) = (vec![usize::MAX; n], vec![0; n]);
        let (mut on_stack, mut comp_of) = (vec![false; n], vec![usize::MAX; n]);
        let (mut stack, mut comps): (Vec<usize>, Vec<Vec<usize>>) = (Vec::new(), Vec::new());
        let mut counter = 0;
        for root in 0..n {
            if index[root] != usize::MAX {
                continue;
            }
            let mut call = vec![(root, 0usize)];
            index[root] = counter;
            low[root] = counter;
            counter += 1;
            stack.push(root);
            on_stack[root] = true;
            while let Some(&(v, i)) = call.last() {
                if let Some(&(w, _)) = self.rows[v].get(i) {
                    call.last_mut().expect("frame").1 += 1;
                    if index[w] == usize::MAX {
                        index[w] = counter;
                        low[w] = counter;
                        counter += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        call.push((w, 0));
                    } else if on_stack[w] {
                        low[v] = low[v].min(index[w]);
                    }
                } else {
                    call.pop();
                    if let Some(&(parent, _)) = call.last() {
                        low[parent] = low[parent].min(low[v]);
                    }
                    if low[v] == index[v] {
                        let mut members = Vec::new();
                        while let Some(w) = stack.pop() {
                            on_stack[w] = false;
                            comp_of[w] = comps.len();
                            members.push(w);
                            if w == v {
                                break;
                            }
                        }
                        comps.push(members);
                    }
                }
            }
        }
        comps
            .iter()
            .enumerate()
            .filter(|(c, members)| {
                members
                    .iter()
                    .all(|&s| self.rows[s].iter().all(|&(t, _)| comp_of[t] == *c))
            })
            .map(|(_, members)| members.clone())
            .collect()
    }

    /// Stationary distribution of the (irreducible) sub-chain on `states`.
    fn stationary_within(
        &self,
        states: &[usize],
        tolerance: f64,
        max_iterations: usize,
    ) -> Vec<f64> {
        let m = states.len();
        if m == 1 {
            return vec![1.0];
        }
        let mut local = vec![usize::MAX; self.num_states()];
        for (i, &s) in states.iter().enumerate() {
            local[s] = i;
        }
        if m <= GTH_LIMIT {
            // GTH: eliminate states one by one (no subtractions, so no
            // cancellation however stiff the rates are), then back-substitute.
            let mut a = vec![vec![0.0; m]; m];
            for (i, &s) in states.iter().enumerate() {
                for &(t, r) in &self.rows[s] {
                    a[i][local[t]] += r;
                }
            }
            let mut out = vec![0.0; m];
            let mut leave = vec![0.0; m];
            for k in (1..m).rev() {
                let s: f64 = a[k][..k].iter().sum();
                leave[k] = s;
                for i in 0..k {
                    let into_k = a[i][k];
                    if into_k == 0.0 {
                        continue;
                    }
                    // `i < k`, so row `k` is split off to read it while
                    // row `i` is updated.
                    let (head, tail) = a.split_at_mut(k);
                    for (j, (cell, via)) in head[i][..k].iter_mut().zip(&tail[0]).enumerate() {
                        if j != i {
                            *cell += into_k * via / s;
                        }
                    }
                }
            }
            out[0] = 1.0;
            for k in 1..m {
                out[k] = (0..k).map(|i| out[i] * a[i][k]).sum::<f64>() / leave[k];
            }
            let total: f64 = out.iter().sum();
            return out.into_iter().map(|x| x / total).collect();
        }
        // Gauss–Seidel on π_j · exit_j = Σ_i π_i · q(i, j).
        let mut preds: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for (i, &s) in states.iter().enumerate() {
            for &(t, r) in &self.rows[s] {
                preds[local[t]].push((i, r));
            }
        }
        let exit: Vec<f64> = states.iter().map(|&s| self.exit[s]).collect();
        let mut pi = vec![1.0 / m as f64; m];
        for _ in 0..max_iterations {
            for j in 0..m {
                pi[j] = preds[j].iter().map(|&(i, r)| pi[i] * r).sum::<f64>() / exit[j];
            }
            let total: f64 = pi.iter().sum();
            pi.iter_mut().for_each(|x| *x /= total);
            let flow: f64 = pi.iter().zip(&exit).map(|(p, e)| p * e).sum();
            let residual: f64 = (0..m)
                .map(|j| {
                    (preds[j].iter().map(|&(i, r)| pi[i] * r).sum::<f64>() - pi[j] * exit[j]).abs()
                })
                .sum();
            if residual <= tolerance * flow {
                break;
            }
        }
        pi
    }
}

/// Largest bottom component solved exactly (dense `O(m³)`); bigger ones
/// use Gauss–Seidel.
const GTH_LIMIT: usize = 400;
