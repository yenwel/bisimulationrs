use std::collections::HashMap;
use std::hash::Hash;

use bisim_core::{Formula, LabelId, Partition, Semiring, Signature, StateId, Wlts};

/// Result of partition refinement: the coarsest stable partition and the
/// partition after every round.
#[derive(Debug, Clone)]
pub struct Refinement {
    /// `history[k]` is the `k`-step equivalence `∼ₖ`; the last entry is `∼`.
    pub history: Vec<Partition>,
}

impl Refinement {
    /// The behavioural equivalence (greatest fixpoint).
    pub fn partition(&self) -> &Partition {
        self.history.last().expect("history is never empty")
    }

    /// The number of rounds needed to reach the fixpoint.
    pub fn rounds(&self) -> usize {
        self.history.len() - 1
    }

    /// The `n`-step approximant `∼ₙ` (equal to `∼` for `n ≥ rounds()`).
    pub fn up_to(&self, n: usize) -> &Partition {
        &self.history[n.min(self.history.len() - 1)]
    }

    /// The first round in which `s` and `t` are separated, if any.
    pub fn separation_depth(&self, s: StateId, t: StateId) -> Option<usize> {
        self.history.iter().position(|p| !p.same_block(s, t))
    }
}

/// Naive signature refinement (Kleene iteration of the final chain).
///
/// Starting from `initial` (or the single-block partition), each round splits
/// every block by the signature of its states with respect to the previous
/// partition. It stops once a round splits nothing. Each round costs
/// `O(n + m)` hashing work and there are at most `n` rounds.
///
/// This is the reference implementation that faster engines are tested against.
pub fn refine<S: Signature>(sys: &S, initial: Option<&Partition>) -> Refinement {
    let n = sys.num_states();
    let first = initial.cloned().unwrap_or_else(|| Partition::coarsest(n));
    assert_eq!(first.len(), n, "initial partition must cover every state");

    let mut history = vec![first];
    loop {
        let current = history.last().expect("non-empty");
        let block_of = current.as_slice();
        let next = Partition::from_keys(
            (0..n as u32).map(|s| (block_of[s as usize], sys.signature(StateId(s), block_of))),
        );
        if next.num_blocks() == current.num_blocks() {
            return Refinement { history };
        }
        history.push(next);
    }
}

/// A formula that holds in `s` but not in `t`, or `None` if they are
/// equivalent (or were separated only by the initial partition).
///
/// `refinement` must be the result of [`refine`] on `sys` without an initial
/// partition. The formula's modal depth equals the round in which the two
/// states were separated, so it is a shortest explanation in that sense.
pub fn distinguishing_formula<L, W>(
    sys: &Wlts<L, W>,
    refinement: &Refinement,
    s: StateId,
    t: StateId,
) -> Option<Formula<L, W::Key>>
where
    L: Eq + Hash + Clone,
    W: Semiring,
{
    refinement.separation_depth(s, t)?;
    let mut memo = HashMap::new();
    witness(sys, &refinement.history, s, t, &mut memo)
}

type Memo<L, K> = HashMap<(StateId, StateId), Option<Formula<L, K>>>;

fn witness<L, W>(
    sys: &Wlts<L, W>,
    history: &[Partition],
    s: StateId,
    t: StateId,
    memo: &mut Memo<L, W::Key>,
) -> Option<Formula<L, W::Key>>
where
    L: Eq + Hash + Clone,
    W: Semiring,
{
    if let Some(f) = memo.get(&(s, t)) {
        return f.clone();
    }
    let depth = history.iter().position(|p| !p.same_block(s, t));
    let result = match depth {
        // Never separated, or separated by the initial partition (no atomic
        // propositions are available to explain that).
        None | Some(0) => None,
        Some(k) => {
            let prev = history[k - 1].as_slice();
            let sums_s = sys.class_sums(s, prev);
            let sums_t = sys.class_sums(t, prev);
            let (label, block) = first_difference(&sums_s.keys(), &sums_t.keys())
                .expect("separated states have different signatures");

            // A state of `block` reached by `label` from s or t, and one state
            // per other block that `label` reaches from s or t.
            let mut inside = None;
            let mut others: Vec<(u32, StateId)> = Vec::new();
            for e in sys.successors(s).iter().chain(sys.successors(t)) {
                if e.label != label {
                    continue;
                }
                let b = prev[e.target.index()];
                if b == block {
                    inside.get_or_insert(e.target);
                } else if !others.iter().any(|&(ob, _)| ob == b) {
                    others.push((b, e.target));
                }
            }
            let x = inside.expect("differing class sum implies a successor in the block");
            let mut conj = Vec::with_capacity(others.len());
            for (_, y) in others {
                conj.push(witness(sys, history, x, y, memo)?);
            }
            let target = match conj.len() {
                0 => Formula::True,
                1 => conj.pop().expect("one conjunct"),
                _ => Formula::And(conj),
            };
            let weight = sums_s.get(&(label, block)).key();
            Some(Formula::Modal {
                label: sys.label(label).clone(),
                weight,
                target: Box::new(target),
            })
        }
    };
    memo.insert((s, t), result.clone());
    result
}

/// First `(label, block)` whose weight key differs between two signatures.
fn first_difference<K: Eq + Clone>(
    a: &[((LabelId, u32), K)],
    b: &[((LabelId, u32), K)],
) -> Option<(LabelId, u32)> {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some((ka, wa)), Some((kb, wb))) if ka == kb => {
                if wa != wb {
                    return Some(*ka);
                }
                i += 1;
                j += 1;
            }
            (Some((ka, _)), Some((kb, _))) => return Some(*ka.min(kb)),
            (Some((ka, _)), None) => return Some(*ka),
            (None, Some((kb, _))) => return Some(*kb),
            (None, None) => unreachable!(),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use bisim_core::NamedBuilder;

    #[test]
    fn records_every_round() {
        // a.a.0 vs a.0: separated after two rounds.
        let (sys, _) = NamedBuilder::from_triples([("p0", 'a', "p1"), ("p1", 'a', "p2")]);
        let r = refine(&sys, None);
        assert_eq!(r.partition().num_blocks(), 3);
        assert_eq!(r.separation_depth(StateId(0), StateId(1)), Some(2));
        assert_eq!(r.up_to(1).num_blocks(), 2);

        let f = distinguishing_formula(&sys, &r, StateId(0), StateId(1)).unwrap();
        assert_eq!(f.depth(), 2);
        assert!(f.holds(&sys, StateId(0)));
        assert!(!f.holds(&sys, StateId(1)));
        assert_eq!(f.to_string(), "⟨a⟩⟨a⟩tt");
    }
}
