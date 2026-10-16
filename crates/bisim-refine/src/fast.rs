//! Fast partition refinement with a dirty-flag worklist.
//!
//! [`refine_fast`] computes the same coarsest stable partition as
//! [`refine`](super::refine), but signatures are cached per state and
//! recomputed only for dirty states: a split dirties exactly the
//! predecessors of states that changed block, and the largest part keeps
//! its block id (Hopcroft's trick), so every state moves blocks at most
//! `log n` times (cf. the refinable partition with clean/dirty states in
//! CoPaR, Wißmann et al.). Stable regions of a large system are never
//! rehashed.
//!
//! This is chaotic iteration of the same monotone operator, hence the
//! same greatest fixpoint — differential-tested against [`refine`] — but
//! it is not the full `O(m log n)` Paige–Tarjan algorithm (weighted
//! class-sum signatures defeat the counting trick), and it keeps no
//! round history: callers needing `∼ₙ` or distinguishing formulas still
//! use [`refine`](super::refine). Use this for `minimise` and for
//! analyses that only need the final partition.
//!
//! Measured profile (`cargo bench -p bisimulationrs`): several times
//! faster than [`refine`] on deep/narrow systems (chains) and dense
//! graphs, up to ~25% slower on shallow symmetric pipelines where almost
//! every block splits every round (the worklist machinery has nothing to
//! prune). A full counting-based Paige–Tarjan backend remains future work.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

use bisim_core::{LabelId, Partition, Semiring, Signature, StateId, Wlts};

/// Cached signature type of [`Wlts`].
type SigOf<W> = Vec<((LabelId, u32), <W as bisim_core::SignatureKey>::Key)>;

/// The coarsest stable partition of `sys`, refining `initial` (or the
/// single-block partition). Output is canonical, hence equal to
/// [`refine`](super::refine)`.`partition()` on the same input.
pub fn refine_fast<L: Eq + Hash + Clone, W: Semiring>(
    sys: &Wlts<L, W>,
    initial: Option<&Partition>,
) -> Partition {
    let n = sys.num_states();
    let first = initial.cloned().unwrap_or_else(|| Partition::coarsest(n));
    assert_eq!(first.len(), n, "initial partition must cover every state");
    if n == 0 {
        return first;
    }

    // Stable block ids (never reused); canonicalised once at the end.
    let mut block_of: Vec<u32> = first.as_slice().to_vec();
    // Normalise ids to 0..k in case the caller passed sparse numbering.
    let renumber: HashMap<u32, u32> = {
        let mut map = HashMap::new();
        for b in block_of.iter() {
            let next = map.len() as u32;
            map.entry(*b).or_insert(next);
        }
        map
    };
    for b in block_of.iter_mut() {
        *b = renumber[b];
    }
    let mut members: Vec<Vec<StateId>> = vec![Vec::new(); renumber.len()];
    for (s, &b) in block_of.iter().enumerate() {
        members[b as usize].push(StateId(s as u32));
    }

    // Predecessors (any label), built lazily on the first split: models
    // that converge immediately never pay for them. A split only affects
    // these signatures.
    let mut pred: Option<Vec<Vec<StateId>>> = None;
    let predecessors = |sys: &Wlts<L, W>| {
        let mut pred = vec![Vec::new(); n];
        for s in sys.states() {
            for e in sys.successors(s) {
                pred[e.target.index()].push(s);
            }
        }
        pred
    };

    let mut dirty = vec![false; n];
    let mut sig: Vec<Option<SigOf<W>>> = (0..n).map(|_| None).collect();
    let mut in_queue = vec![false; members.len()];
    let mut queue: VecDeque<u32> = VecDeque::new();
    let touch = |queue: &mut VecDeque<u32>,
                 in_queue: &mut Vec<bool>,
                 dirty: &mut Vec<bool>,
                 block_of: &[u32],
                 s: StateId| {
        dirty[s.index()] = true;
        let b = block_of[s.index()] as usize;
        if !in_queue[b] {
            in_queue[b] = true;
            queue.push_back(b as u32);
        }
    };
    for s in sys.states() {
        touch(&mut queue, &mut in_queue, &mut dirty, &block_of, s);
    }

    while let Some(c) = queue.pop_front() {
        in_queue[c as usize] = false;
        if members[c as usize].is_empty() {
            continue;
        }
        if !members[c as usize].iter().any(|s| dirty[s.index()]) {
            continue;
        }
        // Recompute dirty signatures only; clean ones are cached.
        for &s in &members[c as usize] {
            if dirty[s.index()] {
                sig[s.index()] = Some(sys.signature(s, &block_of));
                dirty[s.index()] = false;
            }
        }
        // Group by reference (keys cloned once per group below).
        let mut groups: HashMap<&SigOf<W>, Vec<StateId>> = HashMap::new();
        for &s in &members[c as usize] {
            groups
                .entry(sig[s.index()].as_ref().expect("computed or cached"))
                .or_default()
                .push(s);
        }
        if groups.len() <= 1 {
            continue;
        }
        // Keep the largest part in `c` (each state moves O(log n) times);
        // only predecessors of moved states can change signature.
        let mut parts: Vec<(SigOf<W>, Vec<StateId>)> =
            groups.into_iter().map(|(k, v)| (k.clone(), v)).collect();
        let biggest = parts
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, p))| p.len())
            .map(|(i, _)| i)
            .expect("at least two parts");
        let (_, keep) = parts.swap_remove(biggest);
        members[c as usize] = keep;
        for (_, part) in parts {
            // Reserve the slot before reassigning: moved states with
            // self-loops touch their own (new) block id immediately.
            let id = members.len() as u32;
            members.push(part);
            in_queue.push(false);
            if pred.is_none() {
                pred = Some(predecessors(sys));
            }
            let pred = pred.as_ref().expect("built");
            for s in members[id as usize].clone() {
                block_of[s.index()] = id;
                for &p in &pred[s.index()] {
                    touch(&mut queue, &mut in_queue, &mut dirty, &block_of, p);
                }
            }
            in_queue[id as usize] = true;
            queue.push_back(id);
        }
    }

    Partition::from_keys(block_of)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refine::refine;
    use bisim_core::NamedBuilder;

    #[test]
    fn matches_naive_on_chain() {
        let (sys, _) = NamedBuilder::from_triples([("p0", 'a', "p1"), ("p1", 'a', "p2")]);
        assert_eq!(
            refine_fast(&sys, None),
            refine(&sys, None).partition().clone()
        );
        // Deadlock vs loop are distinguished.
        let (sys, _) =
            NamedBuilder::from_triples([("x", 'a', "x"), ("y", 'a', "x"), ("z", 'b', "z")]);
        assert_eq!(
            refine_fast(&sys, None),
            refine(&sys, None).partition().clone()
        );
    }

    #[test]
    fn empty_and_singleton() {
        let builder = NamedBuilder::<&str, char, bool>::new();
        let (sys, _) = builder.build();
        // No triples: no states at all (builder creates states on demand).
        assert_eq!(sys.num_states(), 0);
        assert_eq!(
            refine_fast(&sys, None),
            refine(&sys, None).partition().clone()
        );
        let (one, _) = NamedBuilder::from_triples([("x", 'a', "x")]);
        assert_eq!(
            refine_fast(&one, None),
            refine(&one, None).partition().clone()
        );
    }
}
