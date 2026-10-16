//! Reachability solvers behind CSL/PCTL-style queries.
//!
//! - [`dtmc_matrix`] validates that a [`Wlts`](bisim_core::Wlts) is
//!   stochastic and returns its transition matrix in sparse row form.
//! - [`unbounded_reach`] solves `P[F target]` on a DTMC by value
//!   iteration, with graph precomputation of probability-0 states.
//! - [`bounded_steps`] answers step-bounded `P[F<=k target]` exactly.
//! - [`Ctmc::reach_bounded`] (in `ctmc.rs`) answers time-bounded
//!   `P[F<=t target]` by uniformisation with absorbing targets.
//! - [`Ctmc::embedded`] exposes the jump chain, so unbounded CTMC
//!   reachability reduces to [`unbounded_reach`].

use std::fmt;
use std::hash::Hash;

use bisim_core::{Real, Wlts};

/// Error from [`dtmc_matrix`]: the system is not a DTMC.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    /// A weight outside `[0, 1]`.
    NotProbabilistic { source: usize, weight: f64 },
    /// A row that does not sum to 1.
    RowSum { source: usize, sum: f64 },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::NotProbabilistic { source, weight } => write!(
                f,
                "transition out of state {source} has weight {weight}, not a probability"
            ),
            MatrixError::RowSum { source, sum } => write!(
                f,
                "outgoing probabilities of state {source} sum to {sum}, not 1"
            ),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Sparse transition matrix of a DTMC, validated: every weight in
/// `[0, 1]` and every row sums to 1 (within `1e-9`). Self-loops are kept
/// (they are real DTMC transitions, unlike CTMC self-loops).
pub fn dtmc_matrix<L: Eq + Hash + Clone>(
    lts: &Wlts<L, Real>,
) -> Result<Vec<Vec<(usize, f64)>>, MatrixError> {
    let mut rows = vec![Vec::new(); lts.num_states()];
    for s in lts.states() {
        let mut sum = 0.0;
        for e in lts.successors(s) {
            let w = e.weight.value();
            if !(0.0..=1.0).contains(&w) {
                return Err(MatrixError::NotProbabilistic {
                    source: s.index(),
                    weight: w,
                });
            }
            sum += w;
            rows[s.index()].push((e.target.index(), w));
        }
        if (sum - 1.0).abs() > 1e-9 {
            return Err(MatrixError::RowSum {
                source: s.index(),
                sum,
            });
        }
    }
    Ok(rows)
}

/// Probability to ever hit `targets` from each state (`P[F target]`).
///
/// Value iteration from below: targets stay 1, probability-0 states
/// (found by backward graph reachability, never improved) stay 0, the
/// rest iterate `x(s) = Σ P(s,t)·x(t)` to `tolerance` or `max_iter`.
/// This converges to the least fixpoint, i.e. the true reachability
/// probabilities (standard for finite DTMCs).
pub fn unbounded_reach(
    p: &[Vec<(usize, f64)>],
    targets: &[bool],
    tolerance: f64,
    max_iter: usize,
) -> Vec<f64> {
    assert_eq!(p.len(), targets.len());
    let n = p.len();
    // Prob0: states that cannot reach any target along the graph.
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (s, row) in p.iter().enumerate() {
        for &(t, _) in row {
            rev[t].push(s);
        }
    }
    let mut can = vec![false; n];
    let mut stack: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter(|(_, t)| **t)
        .map(|(s, _)| s)
        .collect();
    while let Some(s) = stack.pop() {
        if can[s] {
            continue;
        }
        can[s] = true;
        stack.extend(rev[s].iter().copied());
    }
    let mut x = vec![0.0; n];
    for (i, t) in targets.iter().enumerate() {
        if *t {
            x[i] = 1.0;
        }
    }
    for _ in 0..max_iter {
        let mut diff: f64 = 0.0;
        for s in 0..n {
            if targets[s] || !can[s] {
                continue;
            }
            let next: f64 = p[s].iter().map(|&(t, r)| r * x[t]).sum();
            diff = diff.max((next - x[s]).abs());
            x[s] = next;
        }
        if diff < tolerance {
            break;
        }
    }
    x
}

/// Exact step-bounded reachability: probability to hit `targets` within
/// `k` steps from each state (`P[F<=k target]` on a DTMC).
pub fn bounded_steps(p: &[Vec<(usize, f64)>], targets: &[bool], k: u64) -> Vec<f64> {
    assert_eq!(p.len(), targets.len());
    let mut x: Vec<f64> = targets.iter().map(|t| f64::from(*t)).collect();
    for _ in 0..k {
        let mut next = vec![0.0; p.len()];
        for s in 0..p.len() {
            next[s] = if targets[s] {
                1.0
            } else {
                p[s].iter().map(|&(t, r)| r * x[t]).sum()
            };
        }
        x = next;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use bisim_core::NamedBuilder;

    fn chain() -> Wlts<&'static str, Real> {
        // 0 -0.25-> 0, 0 -0.75-> 1(absorbing): P[F {1}] = 1 from 0.
        let mut b = NamedBuilder::<&str, &str, Real>::new();
        b.initial("s0")
            .transition("s0", "a", "s0", Real(0.25))
            .transition("s0", "a", "s1", Real(0.75))
            .transition("s1", "b", "s1", Real(1.0));
        b.build().0
    }

    #[test]
    fn reachability_matches_closed_form() {
        let sys = chain();
        let p = dtmc_matrix(&sys).unwrap();
        let x = unbounded_reach(&p, &[false, true], 1e-12, 100_000);
        assert!((x[0] - 1.0).abs() < 1e-9, "eventually absorbed: {}", x[0]);
        assert_eq!(x[1], 1.0);
        // Step-bounded: 1 - 0.25^k.
        for k in [0u64, 1, 2, 10] {
            let b = bounded_steps(&p, &[false, true], k);
            assert!(
                (b[0] - (1.0 - 0.25f64.powi(k as i32))).abs() < 1e-12,
                "k={k}"
            );
        }
        // Unreachable targets stay 0 without iterating forever.
        let x = unbounded_reach(&p, &[true, false], 1e-12, 100_000);
        assert_eq!(x, vec![1.0, 0.0]);
    }

    #[test]
    fn non_stochastic_systems_error() {
        let (sys, _) = NamedBuilder::from_triples([("x", 'a', "y")]);
        let sys = sys.map_weights(|_| Real(2.0));
        // Weight 2 is not a probability at all.
        assert!(matches!(
            dtmc_matrix(&sys),
            Err(MatrixError::NotProbabilistic { .. })
        ));
        // Probabilities that do not sum to 1 are a different error.
        let mut b = NamedBuilder::<&str, &str, Real>::new();
        b.initial("x")
            .transition("x", "a", "x", Real(0.3))
            .transition("x", "a", "y", Real(0.6));
        let (sys, _) = b.build();
        assert!(matches!(dtmc_matrix(&sys), Err(MatrixError::RowSum { .. })));
    }
}
