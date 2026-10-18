//! Fast partition refinement with a dirty-flag worklist.
//!
//! [`refine_fast`] computes the same coarsest stable partition as
//! [`refine`](super::refine), but only recomputes what a split can have
//! changed. A split dirties exactly the predecessors of the states that
//! changed block, and the largest part keeps its block id (Hopcroft's
//! trick), so every state moves blocks at most `log n` times (cf. the
//! refinable partition with clean/dirty states in CoPaR, Wißmann et al.).
//!
//! The clean states of a block all share one cached signature, so a block
//! is processed by hashing its dirty states only and comparing them with
//! that one signature: the cost of a step is the number of dirty states
//! (plus the states that move), not the size of the block. Stable regions
//! of a large system are never rehashed.
//!
//! This is chaotic iteration of the same monotone operator, hence the
//! same greatest fixpoint — differential-tested against [`refine`] — but
//! it is not the full `O(m log n)` Paige–Tarjan algorithm (weighted
//! class-sum signatures defeat the counting trick), and it keeps no
//! round history: callers needing `∼ₙ` or distinguishing formulas still
//! use [`refine`](super::refine). Use this for `minimise` and for
//! analyses that only need the final partition.
//!
//! Measured profile (`cargo bench -p bisimulationrs`): see the benchmark
//! suite; shallow symmetric pipelines, where almost every block splits
//! every round, are the one shape with nothing to prune. A full
//! counting-based Paige–Tarjan backend remains future work.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

use bisim_core::{LabelId, Partition, Semiring, Signature, StateId, Wlts};

/// Cached signature type of [`Wlts`].
type SigOf<W> = Vec<((LabelId, u32), <W as bisim_core::SignatureKey>::Key)>;

/// Removes `s` from `members` in O(1), keeping `pos` (index of every state
/// in its block's member list) valid.
fn remove_member(members: &mut Vec<StateId>, pos: &mut [u32], s: StateId) {
    let i = pos[s.index()] as usize;
    let last = *members.last().expect("block has members");
    members.swap_remove(i);
    if last != s {
        pos[last.index()] = i as u32;
    }
}

/// The coarsest stable partition of `sys`, refining `initial` (or the
/// single-block partition). Output is canonical, hence equal to
/// [`refine`](super::refine)`.`partition() on the same input.
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
    let mut members: Vec<Vec<StateId>> = first.blocks();
    let mut pos = vec![0u32; n];
    for block in &members {
        for (i, s) in block.iter().enumerate() {
            pos[s.index()] = i as u32;
        }
    }
    // The signature shared by the clean states of a block.
    let mut block_sig: Vec<Option<SigOf<W>>> = members.iter().map(|_| None).collect();

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

    // Every state starts dirty; each block lists its dirty states.
    let mut dirty = vec![true; n];
    let mut dirty_in: Vec<Vec<StateId>> = members.clone();
    let mut in_queue = vec![true; members.len()];
    let mut queue: VecDeque<u32> = (0..members.len() as u32).collect();
    let mut is_dirty_now = vec![false; n];

    while let Some(c) = queue.pop_front() {
        let c = c as usize;
        in_queue[c] = false;
        let dirty_states = std::mem::take(&mut dirty_in[c]);
        if dirty_states.is_empty() {
            continue;
        }
        for &s in &dirty_states {
            dirty[s.index()] = false;
        }
        // Hash the dirty states only.
        let mut groups: HashMap<SigOf<W>, Vec<StateId>> = HashMap::new();
        for &s in &dirty_states {
            groups
                .entry(sys.signature(s, &block_of))
                .or_default()
                .push(s);
        }
        // The clean states, plus the dirty ones that still agree with them.
        let clean_n = members[c].len() - dirty_states.len();
        let mut merged: Vec<StateId> = Vec::new();
        if clean_n > 0 {
            let shared = block_sig[c]
                .as_ref()
                .expect("clean states have a signature");
            merged = groups.remove(shared).unwrap_or_default();
        }
        let clean_size = clean_n + merged.len();
        let mut parts: Vec<(SigOf<W>, Vec<StateId>)> = groups.into_iter().collect();
        if parts.is_empty() || (clean_size == 0 && parts.len() == 1) {
            // No split. A block without clean states has just learnt its
            // shared signature.
            if clean_size == 0 {
                block_sig[c] = Some(parts.pop().expect("one group").0);
            }
            continue;
        }

        // Keep the largest group in `c` (each state moves O(log n) times).
        let biggest = parts
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, v))| v.len())
            .map(|(i, (_, v))| (i, v.len()))
            .expect("at least one other group");
        if clean_size >= biggest.1 {
            // The clean group stays; the other groups leave.
            for (_, part) in &parts {
                for &s in part {
                    remove_member(&mut members[c], &mut pos, s);
                }
            }
        } else {
            // A dirty group stays; the clean states leave with the rest.
            let (sig, keep) = parts.swap_remove(biggest.0);
            for &s in &dirty_states {
                is_dirty_now[s.index()] = true;
            }
            let mut leaving: Vec<StateId> = members[c]
                .iter()
                .copied()
                .filter(|s| !is_dirty_now[s.index()])
                .collect();
            for &s in &dirty_states {
                is_dirty_now[s.index()] = false;
            }
            if clean_size > 0 {
                // The clean group, with the dirty states that agree with it.
                let shared = block_sig[c].take().expect("clean states have a signature");
                leaving.extend(merged);
                parts.push((shared, leaving));
            }
            for (i, &s) in keep.iter().enumerate() {
                pos[s.index()] = i as u32;
            }
            members[c] = keep;
            block_sig[c] = Some(sig);
        }

        // Move the leaving groups into new blocks, then dirty the
        // predecessors (after all block ids are final).
        let mut moved: Vec<StateId> = Vec::new();
        for (sig, part) in parts {
            let id = members.len() as u32;
            for (i, &s) in part.iter().enumerate() {
                block_of[s.index()] = id;
                pos[s.index()] = i as u32;
            }
            moved.extend(&part);
            members.push(part);
            block_sig.push(Some(sig));
            dirty_in.push(Vec::new());
            in_queue.push(false);
        }
        if pred.is_none() {
            pred = Some(predecessors(sys));
        }
        let pred = pred.as_ref().expect("built");
        for s in moved {
            for &p in &pred[s.index()] {
                if !dirty[p.index()] {
                    dirty[p.index()] = true;
                    let b = block_of[p.index()] as usize;
                    dirty_in[b].push(p);
                    if !in_queue[b] {
                        in_queue[b] = true;
                        queue.push_back(b as u32);
                    }
                }
            }
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
