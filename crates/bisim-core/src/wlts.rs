use std::hash::Hash;

use crate::ids::{Interner, LabelId, StateId};
use crate::semiring::{Semiring, SignatureKey};
use crate::signature::Signature;
use crate::sparse::SparseMap;

/// One outgoing transition.
#[derive(Debug, Clone)]
pub struct Edge<W> {
    pub label: LabelId,
    pub target: StateId,
    pub weight: W,
}

/// Finite weighted labelled transition system, a coalgebra `X → FS(X, W)^L`.
///
/// Transitions are stored in compressed sparse row form, sorted by
/// `(source, label, target)`. Parallel transitions with the same source, label
/// and target are merged by adding their weights, so multiplicities are kept
/// (crucial for rates: PEPA's `(a,r).P + (a,r).P` has rate `2r`).
#[derive(Debug, Clone)]
pub struct Wlts<L, W> {
    labels: Interner<L>,
    offsets: Vec<u32>,
    edges: Vec<Edge<W>>,
    initial: Option<StateId>,
}

/// A plain labelled transition system.
pub type Lts<L> = Wlts<L, bool>;

impl<L: Eq + Hash + Clone, W: Semiring> Wlts<L, W> {
    pub fn num_states(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn num_transitions(&self) -> usize {
        self.edges.len()
    }

    pub fn initial(&self) -> Option<StateId> {
        self.initial
    }

    pub fn states(&self) -> impl Iterator<Item = StateId> + use<L, W> {
        (0..self.num_states() as u32).map(StateId)
    }

    pub fn successors(&self, s: StateId) -> &[Edge<W>] {
        let lo = self.offsets[s.index()] as usize;
        let hi = self.offsets[s.index() + 1] as usize;
        &self.edges[lo..hi]
    }

    pub fn label(&self, id: LabelId) -> &L {
        self.labels.resolve(id.0)
    }

    pub fn label_id(&self, label: &L) -> Option<LabelId> {
        self.labels.get(label).map(LabelId)
    }

    pub fn labels(&self) -> &[L] {
        self.labels.values()
    }

    /// Iterates all transitions as `(source, label, target, weight)`.
    pub fn transitions(&self) -> impl Iterator<Item = (StateId, &L, StateId, &W)> {
        self.states().flat_map(move |s| {
            self.successors(s)
                .iter()
                .map(move |e| (s, self.label(e.label), e.target, &e.weight))
        })
    }

    /// Total weight of `label`-transitions from `s` into each block of `block_of`.
    pub fn class_sums(&self, s: StateId, block_of: &[u32]) -> SparseMap<(LabelId, u32), W> {
        self.successors(s)
            .iter()
            .map(|e| ((e.label, block_of[e.target.index()]), e.weight.clone()))
            .collect()
    }

    /// Disjoint union `self + other`. States of `other` are shifted by
    /// `self.num_states()`; the result has no initial state.
    pub fn disjoint_union(&self, other: &Self) -> Self {
        let offset = self.num_states() as u32;
        let mut b = WltsBuilder::new();
        b.ensure_states(self.num_states() + other.num_states());
        for (s, l, t, w) in self.transitions() {
            b.add_transition(s, l.clone(), t, w.clone());
        }
        for (s, l, t, w) in other.transitions() {
            b.add_transition(
                StateId(s.0 + offset),
                l.clone(),
                StateId(t.0 + offset),
                w.clone(),
            );
        }
        b.build()
    }

    /// Converts weights, e.g. to forget rates (`|w| !w.is_zero()`).
    pub fn map_weights<W2: Semiring>(&self, mut f: impl FnMut(&W) -> W2) -> Wlts<L, W2> {
        let mut b = WltsBuilder::new();
        b.ensure_states(self.num_states());
        for (s, l, t, w) in self.transitions() {
            b.add_transition(s, l.clone(), t, f(w));
        }
        if let Some(i) = self.initial {
            b.set_initial(i);
        }
        b.build()
    }

    /// Quotient by a partition: one state per block, with class sums as weights.
    ///
    /// Only meaningful when the partition is a bisimulation; then every state
    /// of a block has the same class sums and the quotient is bisimilar to `self`.
    pub fn quotient(&self, partition: &crate::Partition) -> Self {
        let mut b = WltsBuilder::new();
        b.ensure_states(partition.num_blocks());
        for block in partition.blocks() {
            let rep = block[0];
            let src = StateId(partition.block_of(rep));
            for ((label, target), w) in self.class_sums(rep, partition.as_slice()).iter() {
                b.add_transition(src, self.label(*label).clone(), StateId(*target), w.clone());
            }
        }
        if let Some(i) = self.initial {
            b.set_initial(StateId(partition.block_of(i)));
        }
        b.build()
    }
}

impl<L: Eq + Hash + Clone, W: Semiring> Signature for Wlts<L, W> {
    type Sig = Vec<((LabelId, u32), <W as SignatureKey>::Key)>;

    fn num_states(&self) -> usize {
        Wlts::num_states(self)
    }

    fn signature(&self, s: StateId, block_of: &[u32]) -> Self::Sig {
        self.class_sums(s, block_of).keys()
    }
}

/// Builder for [`Wlts`] over dense state ids.
#[derive(Debug, Clone)]
pub struct WltsBuilder<L, W> {
    labels: Interner<L>,
    num_states: usize,
    raw: Vec<(StateId, LabelId, StateId, W)>,
    initial: Option<StateId>,
}

impl<L: Eq + Hash + Clone, W: Semiring> Default for WltsBuilder<L, W> {
    fn default() -> Self {
        Self {
            labels: Interner::new(),
            num_states: 0,
            raw: Vec::new(),
            initial: None,
        }
    }
}

impl<L: Eq + Hash + Clone, W: Semiring> WltsBuilder<L, W> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_state(&mut self) -> StateId {
        self.num_states += 1;
        StateId(self.num_states as u32 - 1)
    }

    pub fn ensure_states(&mut self, n: usize) {
        self.num_states = self.num_states.max(n);
    }

    pub fn set_initial(&mut self, s: StateId) {
        self.ensure_states(s.index() + 1);
        self.initial = Some(s);
    }

    pub fn add_transition(&mut self, source: StateId, label: L, target: StateId, weight: W) {
        self.ensure_states(source.index().max(target.index()) + 1);
        let label = LabelId(self.labels.intern(label));
        self.raw.push((source, label, target, weight));
    }

    pub fn build(mut self) -> Wlts<L, W> {
        self.raw.sort_by_key(|&(s, l, t, _)| (s, l, t));
        let mut edges: Vec<Edge<W>> = Vec::with_capacity(self.raw.len());
        let mut offsets = vec![0u32; self.num_states + 1];
        let mut last: Option<(StateId, LabelId, StateId)> = None;
        for (s, label, target, weight) in self.raw {
            if last == Some((s, label, target)) {
                let e = edges.last_mut().expect("merged edge exists");
                e.weight = e.weight.add(&weight);
            } else {
                edges.push(Edge {
                    label,
                    target,
                    weight,
                });
                offsets[s.index() + 1] += 1;
                last = Some((s, label, target));
            }
        }
        // Weights that cancel out (or were zero to begin with) are not transitions.
        let mut kept = Vec::with_capacity(edges.len());
        let mut s = 0;
        let mut seen = 0;
        let mut counts = vec![0u32; self.num_states];
        for e in edges {
            while seen == offsets[s + 1] {
                s += 1;
                seen = 0;
            }
            seen += 1;
            if !e.weight.is_zero() {
                counts[s] += 1;
                kept.push(e);
            }
        }
        for (i, c) in counts.into_iter().enumerate() {
            offsets[i + 1] = offsets[i] + c;
        }
        Wlts {
            labels: self.labels,
            offsets,
            edges: kept,
            initial: self.initial,
        }
    }
}

/// Builder that names states with arbitrary keys (strings, enums, tuples).
#[derive(Debug, Clone)]
pub struct NamedBuilder<S, L, W> {
    states: Interner<S>,
    inner: WltsBuilder<L, W>,
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone, W: Semiring> Default for NamedBuilder<S, L, W> {
    fn default() -> Self {
        Self {
            states: Interner::new(),
            inner: WltsBuilder::new(),
        }
    }
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone, W: Semiring> NamedBuilder<S, L, W> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&mut self, name: S) -> StateId {
        let id = StateId(self.states.intern(name));
        self.inner.ensure_states(id.index() + 1);
        id
    }

    pub fn initial(&mut self, name: S) -> &mut Self {
        let id = self.state(name);
        self.inner.set_initial(id);
        self
    }

    pub fn transition(&mut self, source: S, label: L, target: S, weight: W) -> &mut Self {
        let s = self.state(source);
        let t = self.state(target);
        self.inner.add_transition(s, label, t, weight);
        self
    }

    /// Builds the system and returns the state names indexed by `StateId`.
    pub fn build(self) -> (Wlts<L, W>, Vec<S>) {
        (self.inner.build(), self.states.values().to_vec())
    }
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone> NamedBuilder<S, L, bool> {
    /// Adds an unweighted transition.
    pub fn edge(&mut self, source: S, label: L, target: S) -> &mut Self {
        self.transition(source, label, target, true)
    }

    /// Builds an LTS from `(source, label, target)` triples; the first source
    /// is the initial state.
    pub fn from_triples(triples: impl IntoIterator<Item = (S, L, S)>) -> (Lts<L>, Vec<S>) {
        let mut b = Self::new();
        for (i, (s, l, t)) in triples.into_iter().enumerate() {
            if i == 0 {
                b.initial(s.clone());
            }
            b.edge(s, l, t);
        }
        b.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Partition, Real};

    #[test]
    fn parallel_transitions_accumulate() {
        let mut b = NamedBuilder::<&str, &str, Real>::new();
        b.initial("P")
            .transition("P", "a", "Q", Real(1.0))
            .transition("P", "a", "Q", Real(2.0))
            .transition("Q", "b", "P", Real(0.0));
        let (sys, names) = b.build();
        assert_eq!(names, vec!["P", "Q"]);
        assert_eq!(sys.num_transitions(), 1);
        assert_eq!(sys.successors(StateId(0))[0].weight, Real(3.0));
        assert!(sys.successors(StateId(1)).is_empty());
    }

    #[test]
    fn disjoint_union_and_quotient() {
        let (a, _) = NamedBuilder::from_triples([("x", 'a', "y"), ("y", 'a', "x")]);
        let u = a.disjoint_union(&a);
        assert_eq!(u.num_states(), 4);
        assert_eq!(u.num_transitions(), 4);

        let q = u.quotient(&Partition::coarsest(4));
        assert_eq!(q.num_states(), 1);
        assert_eq!(q.num_transitions(), 1);
    }
}
