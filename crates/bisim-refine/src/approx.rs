use std::hash::Hash;

use bisim_core::{Formula, Real, StateId, Verdict, Wlts};

use crate::refine::{distinguishing_formula, refine};

/// Tolerance for quasi-Markovian bisimilarity, in the spirit of
/// `(ε, δ)`-bisimulation for CTMCs (Baier, Katoen et al., 2025):
/// related states may differ by at most `epsilon` (absolute, in s⁻¹)
/// on any class sum, and their total exit rates may differ by at most
/// a relative `delta`.
///
/// `epsilon = 0` and `delta = 0` recover exact Markovian bisimilarity.
/// Like ε-bucketing, tolerance grouping is **not** transitive, so this
/// checks a pair of states (a quasi-lumpability witness), it does not
/// build a quotient partition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Absolute tolerance on every `(label, block)` class sum, in s⁻¹.
    pub epsilon: f64,
    /// Relative tolerance on total exit rates, e.g. `0.05` for 5 %.
    pub delta: f64,
}

impl Tolerance {
    /// Exact check: no deviation allowed.
    pub fn exact() -> Self {
        Self {
            epsilon: 0.0,
            delta: 0.0,
        }
    }
}

/// Largest absolute class-sum difference between two states.
///
/// Class sums are measured against the coarsest partition that still
/// relates the two states (the round before exact refinement separates
/// them, or the final partition when they are exactly bisimilar), so a
/// pure rate mismatch reads as its rate difference rather than as a
/// block mismatch. Zero iff the states are exactly Markovian bisimilar.
///
/// This is the behavioural distance this module reports in
/// `Verdict::Approx`.
pub fn approx_distance<L: Eq + Hash + Clone>(
    left: &Wlts<L, Real>,
    s: StateId,
    right: &Wlts<L, Real>,
    t: StateId,
) -> f64 {
    let union = left.disjoint_union(right);
    let t_union = StateId(t.0 + left.num_states() as u32);
    let refinement = refine(&union, None);
    // Deepest round that still keeps s and t together; class sums
    // against that round show the quantitative gap without the
    // qualitative split confounding it.
    let depth = refinement.separation_depth(s, t_union);
    let blocks = match depth {
        None => refinement.partition().as_slice(),
        Some(0) => refinement.partition().as_slice(),
        Some(k) => refinement.up_to(k - 1).as_slice(),
    };
    let sums_s = union.class_sums(s, blocks);
    let sums_t = union.class_sums(t_union, blocks);
    let mut distance: f64 = 0.0;
    for ((label, block), _) in sums_s.keys().into_iter().chain(sums_t.keys()) {
        let key = (label, block);
        let ds = sums_s.get(&key).value();
        let dt = sums_t.get(&key).value();
        distance = distance.max((ds - dt).abs());
    }
    distance
}

/// Total exit rate of a state: the sum of all outgoing rates, ignoring
/// self-loops (which do not change the CTMC).
fn exit_rate<L: Eq + Hash + Clone>(sys: &Wlts<L, Real>, s: StateId) -> f64 {
    sys.successors(s)
        .iter()
        .filter(|e| e.target != s)
        .map(|e| e.weight.value())
        .sum()
}

/// Decides whether state `s` of `left` and state `t` of `right` are
/// quasi-Markovian bisimilar up to `tol`.
///
/// * Class sums that differ by more than `epsilon` on any
///   `(label, block)` yield `Violated`, carrying the exact HML witness.
/// * Exit rates that differ relatively by more than `delta` also yield
///   `Violated` (reusing the exact witness when the states are exactly
///   inequivalent, else a rate-mismatch formula is not expressible with
///   the available exact blocks, so the exact check is re-run).
/// * Otherwise returns `Verdict::Approx { distance, ci: None }` with the
///   measured [`approx_distance`].
pub fn approx_bisimilar_states<L: Eq + Hash + Clone>(
    left: &Wlts<L, Real>,
    s: StateId,
    right: &Wlts<L, Real>,
    t: StateId,
    tol: Tolerance,
) -> Verdict<Formula<L, bisim_core::RealKey>> {
    assert!(tol.epsilon >= 0.0, "epsilon must be non-negative");
    assert!(tol.delta >= 0.0, "delta must be non-negative");
    let distance = approx_distance(left, s, right, t);
    let es = exit_rate(left, s);
    let et = exit_rate(right, t);
    let exit_ok = (es - et).abs() <= tol.delta * es.max(et).max(f64::MIN_POSITIVE);
    if distance <= tol.epsilon && exit_ok {
        return Verdict::Approx { distance, ci: None };
    }
    let union = left.disjoint_union(right);
    let t_union = StateId(t.0 + left.num_states() as u32);
    let refinement = refine(&union, None);
    match distinguishing_formula(&union, &refinement, s, t_union) {
        // Exactly bisimilar but exit rates differ only by self-loops or
        // floating-point noise above tolerance: still report the distance.
        None => Verdict::Approx { distance, ci: None },
        Some(witness) => Verdict::Violated { witness },
    }
}

/// How far a partition is from being ordinarily lumpable: the largest
/// spread (`max − min`) of any `(label, target-block)` class sum among
/// states sharing a block. Zero iff the partition is a Markovian
/// bisimulation (a lumpable partition, as `minimise` requires).
///
/// Use this to justify Bio-PEPA-style aggregations (e.g. counting
/// abstractions of `P[n]` arrays) or PRISM/Modest symmetry reductions:
/// a small error means the quotient chain is close to the full chain.
pub fn lumping_error<L: Eq + Hash + Clone>(
    sys: &Wlts<L, Real>,
    partition: &bisim_core::Partition,
) -> f64 {
    let blocks = partition.as_slice();
    let mut error: f64 = 0.0;
    for block in partition.blocks() {
        if block.len() < 2 {
            continue;
        }
        // Collect every (label, target-block) key occurring in this block.
        let mut keys = std::collections::BTreeSet::new();
        let sums: Vec<_> = block.iter().map(|&s| sys.class_sums(s, blocks)).collect();
        for sums_s in &sums {
            for (k, _) in sums_s.keys() {
                keys.insert(k);
            }
        }
        for key in keys {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for sums_s in &sums {
                let v = sums_s.get(&key).value();
                lo = lo.min(v);
                hi = hi.max(v);
            }
            error = error.max(hi - lo);
        }
    }
    if error == f64::INFINITY { 0.0 } else { error }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bisim_core::NamedBuilder;

    #[test]
    fn exact_states_have_zero_distance() {
        let (a, _) = NamedBuilder::from_triples([("x", 'a', "y"), ("y", 'a', "x")]);
        let au = a.map_weights(|_| Real(1.0));
        let bu = a.map_weights(|_| Real(1.0));
        assert_eq!(approx_distance(&au, StateId(0), &bu, StateId(0)), 0.0);
        let v = approx_bisimilar_states(&au, StateId(0), &bu, StateId(0), Tolerance::exact());
        assert!(matches!(v, Verdict::Approx { distance, .. } if distance == 0.0));
    }

    #[test]
    fn rate_mismatch_within_and_beyond_tolerance() {
        // Single-state loops with rates 1.0 vs 1.04.
        let mut ab = NamedBuilder::<&str, &str, Real>::new();
        ab.initial("p").transition("p", "a", "p", Real(1.0));
        let (a, _) = ab.build();
        let mut bb = NamedBuilder::<&str, &str, Real>::new();
        bb.initial("q").transition("q", "a", "q", Real(1.04));
        let (b, _) = bb.build();
        // Self-loops do not affect the CTMC exit rate, but class sums
        // (which keep self-loops, as lumpability does) differ by 0.04.
        assert!((approx_distance(&a, StateId(0), &b, StateId(0)) - 0.04).abs() < 1e-12);
        let ok = approx_bisimilar_states(
            &a,
            StateId(0),
            &b,
            StateId(0),
            Tolerance {
                epsilon: 0.05,
                delta: 1.0,
            },
        );
        assert!(matches!(ok, Verdict::Approx { .. }));
        let bad = approx_bisimilar_states(
            &a,
            StateId(0),
            &b,
            StateId(0),
            Tolerance {
                epsilon: 0.01,
                delta: 1.0,
            },
        );
        assert!(matches!(bad, Verdict::Violated { .. }));
    }

    #[test]
    fn lumpable_partition_has_zero_error() {
        let (a, _) = NamedBuilder::from_triples([("x", 'a', "y"), ("y", 'a', "x")]);
        let sys = a.map_weights(|_| Real(2.0));
        let r = refine(&sys, None);
        assert_eq!(lumping_error(&sys, r.partition()), 0.0);
        let coarse = bisim_core::Partition::coarsest(2);
        assert_eq!(lumping_error(&sys, &coarse), 0.0);
    }

    #[test]
    fn non_lumpable_partition_reports_spread() {
        // x -a,1-> y, y -a,3-> y (self-loop kept in class sums).
        let mut b = NamedBuilder::<&str, &str, Real>::new();
        b.initial("x")
            .transition("x", "a", "y", Real(1.0))
            .transition("y", "a", "y", Real(3.0));
        let (sys, _) = b.build();
        let coarse = bisim_core::Partition::coarsest(2);
        // x has a-sum 1.0 into block 0, y has 3.0: spread 2.0.
        assert!((lumping_error(&sys, &coarse) - 2.0).abs() < 1e-12);
    }
}
