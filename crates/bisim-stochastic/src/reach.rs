//! Reachability solvers behind CSL/PCTL-style queries.
//!
//! - [`dtmc_matrix`] validates that a [`Wlts`] is
//!   stochastic and returns its transition matrix in sparse row form.
//! - [`unbounded_reach`] solves `P[F target]` on a DTMC by interval
//!   iteration, with graph precomputation of probability-0 and -1 states.
//! - [`bounded_steps`] answers step-bounded `P[F<=k target]` exactly.
//! - [`Ctmc::reach_bounded`](crate::Ctmc::reach_bounded) answers time-bounded
//!   `P[F<=t target]` by uniformisation with absorbing targets.
//! - [`Ctmc::embedded`](crate::Ctmc::embedded) exposes the jump chain, so unbounded CTMC
//!   reachability reduces to [`unbounded_reach`].

use std::fmt;
use std::hash::Hash;

use bisim_core::{Real, Wlts};

/// Error from [`dtmc_matrix`]: the system is not a DTMC.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
/// Graph precomputation first: states that cannot reach a target have
/// probability 0, and states that cannot reach such a state without first
/// passing a target have probability 1. Without the second step, value
/// iteration from below stops early on chains that almost never leave a
/// loop (a step that changes little is not convergence).
///
/// The remaining states have no end component, so their equations have a
/// unique solution, and interval iteration is sound: a lower iterate from 0
/// and an upper iterate from 1 close in on it, and iteration stops once
/// they are within `tolerance` of each other (or after `max_iter` sweeps).
/// The result is the lower iterate, at most `tolerance` below the truth.
pub fn unbounded_reach(
    p: &[Vec<(usize, f64)>],
    targets: &[bool],
    tolerance: f64,
    max_iter: usize,
) -> Vec<f64> {
    assert_eq!(p.len(), targets.len());
    let n = p.len();
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (s, row) in p.iter().enumerate() {
        for &(t, _) in row {
            rev[t].push(s);
        }
    }
    // Backward reachability from `seeds`, never passing through `blocked`.
    let backward = |seeds: Vec<usize>, blocked: &[bool]| {
        let mut seen = vec![false; n];
        let mut stack = seeds;
        while let Some(s) = stack.pop() {
            if seen[s] {
                continue;
            }
            seen[s] = true;
            stack.extend(rev[s].iter().copied().filter(|&r| !blocked[r]));
        }
        seen
    };
    let target_states: Vec<usize> = (0..n).filter(|&s| targets[s]).collect();
    let can = backward(target_states, &vec![false; n]);
    // States that can reach a probability-0 state avoiding targets.
    let zero_states: Vec<usize> = (0..n).filter(|&s| !can[s]).collect();
    let below_one = backward(zero_states, targets);

    let mut lo = vec![0.0; n];
    let mut hi = vec![0.0; n];
    let mut open = Vec::new();
    for s in 0..n {
        if targets[s] || !below_one[s] {
            lo[s] = 1.0;
            hi[s] = 1.0;
        } else if can[s] {
            hi[s] = 1.0;
            open.push(s);
        }
    }
    for _ in 0..max_iter {
        let mut gap: f64 = 0.0;
        for &s in &open {
            lo[s] = p[s].iter().map(|&(t, r)| r * lo[t]).sum();
            hi[s] = p[s].iter().map(|&(t, r)| r * hi[t]).sum::<f64>().min(1.0);
            gap = gap.max(hi[s] - lo[s]);
        }
        if gap < tolerance {
            break;
        }
    }
    lo
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
    fn nearly_closed_loops_still_reach_the_target() {
        // s0 -1-> s1, s1 -(1-e)-> s0, s1 -e-> s2 (absorbing). The target is
        // reached with probability 1, however small e is; a stopping rule
        // on the size of one step would report ~e instead.
        for eps in [1e-3, 1e-6, 1e-8] {
            let mut b = NamedBuilder::<&str, &str, Real>::new();
            b.initial("s0")
                .transition("s0", "a", "s1", Real(1.0))
                .transition("s1", "a", "s0", Real(1.0 - eps))
                .transition("s1", "a", "s2", Real(eps))
                .transition("s2", "a", "s2", Real(1.0));
            let (sys, names) = b.build();
            let p = dtmc_matrix(&sys).unwrap();
            let target: Vec<bool> = names.iter().map(|n| *n == "s2").collect();
            let x = unbounded_reach(&p, &target, 1e-9, 1_000_000);
            assert_eq!(x[0], 1.0, "eps {eps}");
        }
    }

    #[test]
    fn intermediate_probabilities_are_bracketed() {
        // s0: 0.5 -> target, 0.4 -> s0, 0.1 -> sink: P = 0.5 / 0.6.
        let p = vec![
            vec![(1, 0.5), (0, 0.4), (2, 0.1)],
            vec![(1, 1.0)],
            vec![(2, 1.0)],
        ];
        let x = unbounded_reach(&p, &[false, true, false], 1e-12, 100_000);
        assert!((x[0] - 5.0 / 6.0).abs() < 1e-11, "{}", x[0]);
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
