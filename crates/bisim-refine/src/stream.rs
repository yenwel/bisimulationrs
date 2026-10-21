//! Verdicts over growing prefixes of two systems.
//!
//! [`PrefixChecker`] compares two systems that are only observed
//! piece by piece: transitions arrive over time, and a state may be declared
//! *complete* once all of its outgoing transitions have been seen. The
//! verdicts are sound for what is still to come:
//!
//! - [`Violated`](Verdict::Violated) is final: it is only reported when a
//!   transition that *was* observed has no matching answer from a state that
//!   is complete, so no later observation can repair it. It carries a
//!   distinguishing formula as usual.
//! - [`Holds`](Verdict::Holds) means the relation holds and every state
//!   involved is complete.
//! - [`HoldsSoFar`](Verdict::HoldsSoFar) means no difference has been found,
//!   but some state involved can still gain transitions.
//!
//! A state that is never declared complete can always gain an answer, so
//! missing answers from it never count against the relation. A complete
//! state cannot gain transitions: observing one is an error, which is what
//! keeps `Violated` final.
//!
//! The three games are solved incrementally (see `live.rs`): each observed
//! transition or completed state updates them in place, so the cost per
//! event is proportional to the positions it touches, not to the prefix.
//! [`replay_verdict`](PrefixChecker::replay_verdict) solves them from
//! scratch instead; it is the reference the incremental answer is tested
//! against.
//!
//! For long runs, [`compact`](PrefixChecker::compact) collapses the
//! positions whose outcome is settled (refuted, or holding with only
//! complete states below) into their final verdict; nothing unsettled is
//! dropped, so the verdict stays exact. The observed transitions are kept
//! (they define the prefix and the replay).

use std::fmt;
use std::hash::Hash;

use bisim_core::{ComparisonVerdict, Formula, Lts, NamedBuilder, StateId, Verdict};

use crate::game::{Game, GameBudget};
use crate::live::{LiveGame, Store};

/// A transition observed from a state already declared complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixError<S> {
    /// The transition was observed on the left system.
    pub left: bool,
    /// The complete state it starts from.
    pub state: S,
}

impl<S: fmt::Debug> fmt::Display for PrefixError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side = if self.left { "left" } else { "right" };
        write!(
            f,
            "transition from {:?} on the {side} system, which was declared complete",
            self.state
        )
    }
}

impl<S: fmt::Debug> std::error::Error for PrefixError<S> {}

/// The three incremental games.
#[derive(Debug)]
struct Games {
    /// The right system simulates the left one.
    forward: LiveGame,
    /// The left system simulates the right one.
    backward: LiveGame,
    bisimulation: LiveGame,
}

impl Games {
    fn new<S, L>(store: &Store<S, L>, roots: (u32, u32), max_pairs: usize) -> Self {
        Self {
            forward: LiveGame::new(store, roots, false, true, max_pairs),
            backward: LiveGame::new(store, roots, true, true, max_pairs),
            bisimulation: LiveGame::new(store, roots, false, false, max_pairs),
        }
    }

    fn each(&self) -> [&LiveGame; 3] {
        [&self.forward, &self.backward, &self.bisimulation]
    }

    fn all(&mut self) -> [&mut LiveGame; 3] {
        [
            &mut self.forward,
            &mut self.backward,
            &mut self.bisimulation,
        ]
    }
}

/// Compares the initial states of two systems observed incrementally.
pub struct PrefixChecker<S, L> {
    store: Store<S, L>,
    roots: (u32, u32),
    budget: GameBudget,
    games: Games,
    /// The verdict last returned by [`poll_change`](Self::poll_change).
    last: Option<[u8; 3]>,
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone> fmt::Debug for PrefixChecker<S, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let edges = |side: usize| {
            self.store.sides[side]
                .succ
                .iter()
                .map(Vec::len)
                .sum::<usize>()
        };
        f.debug_struct("PrefixChecker")
            .field("left_transitions", &edges(0))
            .field("right_transitions", &edges(1))
            .field("pairs", &self.games.bisimulation.pairs())
            .finish()
    }
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone> PrefixChecker<S, L> {
    /// Starts with the two initial states and no transitions.
    pub fn new(left_initial: S, right_initial: S) -> Self {
        let mut store = Store::new();
        let roots = (
            store.sides[0].id(left_initial),
            store.sides[1].id(right_initial),
        );
        let budget = GameBudget::default();
        let games = Games::new(&store, roots, budget.max_pairs);
        Self {
            store,
            roots,
            budget,
            games,
            last: None,
        }
    }

    /// Limits each game to `budget.max_pairs` positions; past that the
    /// verdict is [`Unknown`](Verdict::Unknown) (a violation found before
    /// stays). `max_depth` only applies to
    /// [`replay_verdict`](Self::replay_verdict).
    pub fn with_budget(mut self, budget: GameBudget) -> Self {
        self.budget = budget;
        self.games = Games::new(&self.store, self.roots, budget.max_pairs);
        self
    }

    /// Adds observed transitions `(source, label, target)` of the left
    /// system. Fails, adding nothing, if a source was declared complete.
    pub fn observe_left(
        &mut self,
        transitions: impl IntoIterator<Item = (S, L, S)>,
    ) -> Result<(), PrefixError<S>> {
        self.observe(0, transitions)
    }

    /// Adds observed transitions of the right system. Fails, adding nothing,
    /// if a source was declared complete.
    pub fn observe_right(
        &mut self,
        transitions: impl IntoIterator<Item = (S, L, S)>,
    ) -> Result<(), PrefixError<S>> {
        self.observe(1, transitions)
    }

    fn observe(
        &mut self,
        side: usize,
        transitions: impl IntoIterator<Item = (S, L, S)>,
    ) -> Result<(), PrefixError<S>> {
        let transitions: Vec<(S, L, S)> = transitions.into_iter().collect();
        for (s, _, _) in &transitions {
            let st = &self.store.sides[side];
            if st.get(s).is_some_and(|i| st.complete[i as usize]) {
                return Err(PrefixError {
                    left: side == 0,
                    state: s.clone(),
                });
            }
        }
        for (s, l, t) in transitions {
            let (s, t) = (self.store.sides[side].id(s), self.store.sides[side].id(t));
            let l = self.store.label(l);
            self.store.sides[side].succ[s as usize].push((l, t));
            for game in self.games.all() {
                game.on_edge(&self.store, side, s, l, t);
            }
        }
        Ok(())
    }

    /// Declares that every transition out of `state` in the left system has
    /// been observed.
    pub fn complete_left(&mut self, state: S) {
        self.complete(0, state);
    }

    /// Declares that every transition out of `state` in the right system has
    /// been observed.
    pub fn complete_right(&mut self, state: S) {
        self.complete(1, state);
    }

    fn complete(&mut self, side: usize, state: S) {
        let s = self.store.sides[side].id(state);
        if self.store.sides[side].complete[s as usize] {
            return;
        }
        self.store.sides[side].complete[s as usize] = true;
        for game in self.games.all() {
            game.on_complete(&self.store, side, s);
        }
    }

    /// Collapses the settled positions of the three games: refuted ones
    /// keep only their refuting move (the witness stays recoverable), and
    /// held ones whose states are complete and reach only complete states
    /// are dropped and remembered as held pairs. Unsettled positions are
    /// kept as they are, so every later verdict is the one
    /// [`replay_verdict`](Self::replay_verdict) gives. Costs time linear in
    /// the observed transitions and kept positions: call it every so often
    /// (say, when [`live_positions`](Self::live_positions) has doubled).
    pub fn compact(&mut self) {
        self.compact_with(false);
    }

    /// [`compact`](Self::compact), and also minimises the completed parts
    /// of both systems by bisimilarity (with [`refine_fast`](crate::refine_fast)),
    /// remembering held pairs per pair of classes: a later pair of complete
    /// states is then decided from its classes without being explored.
    ///
    /// Verdicts still equal the replay, with one exception: the pair budget
    /// counts explored positions, so pairs decided from their classes are
    /// not counted and a finite budget runs out later than the replay's.
    pub fn compact_and_minimise(&mut self) {
        self.compact_with(true);
    }

    fn compact_with(&mut self, minimise: bool) {
        self.store.settle(minimise);
        for game in self.games.all() {
            game.compact(&self.store, minimise);
        }
    }

    /// Positions held in memory by the three games (unsettled, refuted,
    /// and the initial pairs).
    pub fn live_positions(&self) -> usize {
        self.games.each().iter().map(|g| g.live_positions()).sum()
    }

    /// Settled pairs (or pairs of classes) the three games remember as
    /// holding forever, instead of keeping their positions.
    pub fn held_pairs(&self) -> usize {
        self.games.each().iter().map(|g| g.held()).sum()
    }

    /// Estimated heap bytes of the games and the observed systems, from
    /// container capacities (state and label values count their inline
    /// size only).
    pub fn heap_bytes(&self) -> usize {
        self.store.heap_bytes()
            + self
                .games
                .each()
                .iter()
                .map(|g| g.heap_bytes())
                .sum::<usize>()
    }

    /// The three comparisons for the prefix seen so far.
    pub fn verdict(&self) -> ComparisonVerdict<Formula<L, bool>> {
        let g = &self.games;
        combine(
            || g.forward.verdict(&self.store),
            || g.backward.verdict(&self.store),
            || g.bisimulation.verdict(&self.store),
        )
    }

    /// The verdict, if its kind changed since the last call (the first call
    /// always reports). Suits event loops that alert only on change.
    pub fn poll_change(&mut self) -> Option<ComparisonVerdict<Formula<L, bool>>> {
        let v = self.verdict();
        let kinds = [
            kind(&v.right_simulates_left),
            kind(&v.mutually_similar),
            kind(&v.bisimilar),
        ];
        if self.last == Some(kinds) {
            return None;
        }
        self.last = Some(kinds);
        Some(v)
    }

    /// The same verdict, solved from scratch by the batch game on the
    /// prefix seen so far: O(prefix) per call. The reference
    /// implementation for tests and benchmarks.
    pub fn replay_verdict(&self) -> ComparisonVerdict<Formula<L, bool>> {
        let (left, l0, left_open) = self.snapshot(0);
        let (right, r0, right_open) = self.snapshot(1);
        let union = left.disjoint_union(&right);
        let r0 = StateId(r0.0 + left.num_states() as u32);
        let unfinished: Vec<bool> = left_open.into_iter().chain(right_open).collect();
        let game = |s: StateId, t: StateId, simulation: bool| {
            Game::on_prefix(&union, self.budget, unfinished.clone(), simulation).play(s, t)
        };
        combine(
            || game(l0, r0, true),
            || game(r0, l0, true),
            || game(l0, r0, false),
        )
    }

    /// One side as a batch system, its initial state, and which states are
    /// still unfinished (indexed by state).
    fn snapshot(&self, side: usize) -> (Lts<L>, StateId, Vec<bool>) {
        let st = &self.store.sides[side];
        let root = if side == 0 {
            self.roots.0
        } else {
            self.roots.1
        };
        let mut b = NamedBuilder::<u32, L, bool>::new();
        b.initial(root);
        for (s, succ) in st.succ.iter().enumerate() {
            b.state(s as u32);
            for &(l, t) in succ {
                b.edge(s as u32, self.store.labels[l as usize].clone(), t);
            }
        }
        let (lts, names) = b.build();
        let unfinished = names.iter().map(|&n| !st.complete[n as usize]).collect();
        let initial = lts.initial().expect("initial state was set");
        (lts, initial, unfinished)
    }
}

/// Combines the forward, backward and bisimulation games as the comparison
/// reports them; `backward` and `bisimulation` are only consulted when
/// needed.
fn combine<X: Clone>(
    forward: impl FnOnce() -> Verdict<X>,
    backward: impl FnOnce() -> Verdict<X>,
    bisimulation: impl FnOnce() -> Verdict<X>,
) -> ComparisonVerdict<X> {
    let right_simulates_left = forward();
    let mutually_similar = match &right_simulates_left {
        Verdict::Violated { .. } => right_simulates_left.clone(),
        first => {
            let back = backward();
            match (first, &back) {
                (_, Verdict::Violated { .. }) => back,
                (a, b) => weakest(a, b),
            }
        }
    };
    let bisimilar = match &mutually_similar {
        Verdict::Violated { .. } => mutually_similar.clone(),
        similar => weakest(similar, &bisimulation()),
    };
    ComparisonVerdict {
        right_simulates_left,
        mutually_similar,
        bisimilar,
    }
}

/// A small code per verdict kind, ignoring witnesses.
fn kind<X>(v: &Verdict<X>) -> u8 {
    match v {
        Verdict::Holds => 0,
        Verdict::HoldsSoFar => 1,
        Verdict::Violated { .. } => 2,
        Verdict::Unknown => 3,
        Verdict::HoldsUpTo { .. } => 4,
        _ => 5,
    }
}

/// The weaker of two non-violated verdicts about the same prefix; a
/// violation of the second wins.
fn weakest<X: Clone>(a: &Verdict<X>, b: &Verdict<X>) -> Verdict<X> {
    match (a, b) {
        (_, v @ Verdict::Violated { .. }) => v.clone(),
        (Verdict::Holds, other) => other.clone(),
        (other, Verdict::Holds) => other.clone(),
        (Verdict::Unknown, _) | (_, Verdict::Unknown) => Verdict::Unknown,
        (a, _) => a.clone(),
    }
}
