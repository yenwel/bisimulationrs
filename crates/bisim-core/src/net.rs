//! Safe labelled Petri nets: the causal (true-concurrency) view of a system.
//!
//! A transition system says *which* steps can happen; a net also says
//! *what each step needs and produces*. A transition consumes the tokens in
//! its preset and produces the tokens in its postset, so one step depends
//! on another exactly when it consumes a token the other produced. Two
//! steps that touch disjoint places are independent (concurrent), however
//! they happen to be interleaved.
//!
//! Nets here are *safe*: a place holds at most one token, so a marking is a
//! set of places. [`SafeNet::fire`] reports a contact (a token produced into
//! a place that is already marked) instead of silently merging tokens.

use std::collections::{HashMap, VecDeque};

use crate::{Lts, StateId, WltsBuilder};

/// A set of marked places, sorted.
pub type Marking = Vec<u32>;

/// One transition: what it consumes and produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetTransition<L> {
    pub label: L,
    /// Consumed places, sorted, distinct.
    pub pre: Vec<u32>,
    /// Produced places, sorted, distinct.
    pub post: Vec<u32>,
}

/// A safe labelled Petri net with an initial marking.
#[derive(Debug, Clone, PartialEq)]
pub struct SafeNet<L> {
    /// A name per place, for explanations.
    pub places: Vec<String>,
    pub transitions: Vec<NetTransition<L>>,
    pub initial: Marking,
}

/// Firing produced a token into a place that already held one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contact {
    pub transition: usize,
    pub place: u32,
}

impl std::fmt::Display for Contact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "transition {} puts a second token on place {}: the net is not safe",
            self.transition, self.place
        )
    }
}

impl std::error::Error for Contact {}

fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v.dedup();
    v
}

impl<L> SafeNet<L> {
    /// A net, with presets, postsets and the initial marking normalised.
    pub fn new(places: Vec<String>, transitions: Vec<NetTransition<L>>, initial: Marking) -> Self {
        Self {
            places,
            transitions: transitions
                .into_iter()
                .map(|t| NetTransition {
                    label: t.label,
                    pre: sorted(t.pre),
                    post: sorted(t.post),
                })
                .collect(),
            initial: sorted(initial),
        }
    }

    /// Transitions enabled at `marking` (preset marked).
    pub fn enabled<'a>(&'a self, marking: &'a [u32]) -> impl Iterator<Item = usize> + 'a {
        self.transitions
            .iter()
            .enumerate()
            .filter(move |(_, t)| t.pre.iter().all(|p| marking.binary_search(p).is_ok()))
            .map(|(i, _)| i)
    }

    /// The marking after firing `t` at `marking`.
    pub fn fire(&self, marking: &[u32], t: usize) -> Result<Marking, Contact> {
        let tr = &self.transitions[t];
        let mut next: Vec<u32> = marking
            .iter()
            .copied()
            .filter(|p| tr.pre.binary_search(p).is_err())
            .collect();
        for &p in &tr.post {
            if next.contains(&p) {
                return Err(Contact {
                    transition: t,
                    place: p,
                });
            }
            next.push(p);
        }
        next.sort_unstable();
        Ok(next)
    }
}

impl<L: Clone + Eq + std::hash::Hash> SafeNet<L> {
    /// The interleaving view: markings reachable from the initial one and
    /// the transitions between them (at most `max_states` markings).
    pub fn reachability_graph(
        &self,
        max_states: usize,
    ) -> Result<(Lts<L>, Vec<Marking>), ReachError> {
        let mut index: HashMap<Marking, u32> = HashMap::new();
        let mut markings = vec![self.initial.clone()];
        index.insert(self.initial.clone(), 0);
        let mut b = WltsBuilder::new();
        b.ensure_states(1);
        b.set_initial(StateId(0));
        let mut queue = VecDeque::from([0u32]);
        while let Some(i) = queue.pop_front() {
            let m = markings[i as usize].clone();
            for t in self.enabled(&m).collect::<Vec<_>>() {
                let next = self.fire(&m, t).map_err(ReachError::Contact)?;
                let j = match index.get(&next) {
                    Some(&j) => j,
                    None => {
                        if markings.len() >= max_states {
                            return Err(ReachError::TooLarge(max_states));
                        }
                        let j = markings.len() as u32;
                        index.insert(next.clone(), j);
                        markings.push(next);
                        b.ensure_states(j as usize + 1);
                        queue.push_back(j);
                        j
                    }
                };
                b.add_transition(
                    StateId(i),
                    self.transitions[t].label.clone(),
                    StateId(j),
                    true,
                );
            }
        }
        Ok((b.build(), markings))
    }
}

/// Why the reachability graph could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReachError {
    Contact(Contact),
    TooLarge(usize),
}

impl std::fmt::Display for ReachError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contact(c) => c.fmt(f),
            Self::TooLarge(n) => write!(f, "more than {n} reachable markings"),
        }
    }
}

impl std::error::Error for ReachError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_steps_commute_and_contacts_are_reported() {
        // a: 0 → 2, b: 1 → 3 (independent); c: 2 → 1 (contact while 1 is marked).
        let net = SafeNet::new(
            (0..4).map(|i| format!("p{i}")).collect(),
            vec![
                NetTransition {
                    label: 'a',
                    pre: vec![0],
                    post: vec![2],
                },
                NetTransition {
                    label: 'b',
                    pre: vec![1],
                    post: vec![3],
                },
                NetTransition {
                    label: 'c',
                    pre: vec![2],
                    post: vec![1],
                },
            ],
            vec![1, 0],
        );
        assert_eq!(net.initial, vec![0, 1]);
        let ab = net.fire(&net.fire(&net.initial, 0).unwrap(), 1).unwrap();
        let ba = net.fire(&net.fire(&net.initial, 1).unwrap(), 0).unwrap();
        assert_eq!(ab, ba);
        let after_a = net.fire(&net.initial, 0).unwrap();
        assert_eq!(
            net.fire(&after_a, 2),
            Err(Contact {
                transition: 2,
                place: 1
            })
        );
        assert!(matches!(
            net.reachability_graph(100),
            Err(ReachError::Contact(_))
        ));
    }
}
