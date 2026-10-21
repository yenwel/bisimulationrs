//! The bisimulation game, kept up to date while two systems grow.
//!
//! [`game`](super::game) solves the game once, on a fixed system. Here the
//! two systems are observed piece by piece, and every new transition or
//! completed state updates the solved game in place: work per event is
//! proportional to the positions it touches, not to the prefix seen so far.
//!
//! This is sound because of the prefix discipline enforced by
//! [`PrefixChecker`](super::PrefixChecker): a state gains transitions only
//! until it is declared complete. Then the set of refuted positions only
//! grows:
//!
//! - a new transition of the *moving* state adds a Spoiler move, which can
//!   only refute more;
//! - a new transition of the *answering* state adds an answer, but only to a
//!   move whose answerer is still open, and such a move never refutes;
//! - completing a state lets its unanswered moves refute.
//!
//! So the counter-based propagation of the batch game (refute a position
//! once some move has no unrefuted answer left) is maintained exactly, and
//! the answer after every event equals solving the game from scratch on the
//! prefix (checked against the replay in the property tests).
//!
//! # Compaction
//!
//! Positions accumulate with every discovered pair, so
//! [`compact`](LiveGame::compact) collapses the ones whose outcome is
//! settled; nothing unsettled is ever dropped. A position is settled when
//!
//! - it is refuted: refutations only grow, so it stays lost. Only its
//!   refuting move is kept (its answers are refuted too), which is exactly
//!   what the witness needs;
//! - or it holds and both its states are *closed* (complete, reaching only
//!   complete states): every position below it is final, so it holds
//!   forever. It is dropped and remembered as a held pair, and any move it
//!   answers can never refute, so that move forgets its answers.
//!
//! Once the initial position is refuted the verdict is final: the game
//! stops listening to events, and compaction keeps only the positions its
//! witness needs.
//!
//! With minimisation, the closed parts of both systems are also quotiented
//! by bisimilarity ([`refine_fast`] on the classes found so far plus the
//! newly closed states), and held pairs are remembered per pair of
//! classes: bisimilar states win and lose the same games, so a new pair of
//! closed states is answered from its classes without exploring it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::mem::size_of;

use bisim_core::{Formula, StateId, Verdict, WltsBuilder};

use crate::fast::refine_fast;

/// No class yet: the state is not closed, or not minimised.
const NO_CLASS: u32 = u32::MAX;

/// One observed system: interned states, their transitions, completeness.
#[derive(Debug)]
pub(crate) struct SideStore<S> {
    ids: HashMap<S, u32>,
    pub(crate) names: Vec<S>,
    /// Outgoing `(label, target)` per state, in arrival order.
    pub(crate) succ: Vec<Vec<(u32, u32)>>,
    pub(crate) complete: Vec<bool>,
    /// Complete, and every reachable state is complete (as of the last
    /// compaction). Never reverts: complete states gain no transitions.
    closed: Vec<bool>,
    /// Bisimulation class of a closed state, once minimised.
    class_of: Vec<u32>,
}

impl<S: Eq + Hash + Clone> SideStore<S> {
    pub(crate) fn new() -> Self {
        Self {
            ids: HashMap::new(),
            names: Vec::new(),
            succ: Vec::new(),
            complete: Vec::new(),
            closed: Vec::new(),
            class_of: Vec::new(),
        }
    }

    pub(crate) fn id(&mut self, s: S) -> u32 {
        if let Some(&i) = self.ids.get(&s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.ids.insert(s.clone(), i);
        self.names.push(s);
        self.succ.push(Vec::new());
        self.complete.push(false);
        self.closed.push(false);
        self.class_of.push(NO_CLASS);
        i
    }

    pub(crate) fn get(&self, s: &S) -> Option<u32> {
        self.ids.get(s).copied()
    }

    /// Marks the states that no longer reach an incomplete one: everything
    /// not found backwards from the incomplete states.
    fn settle(&mut self) {
        let n = self.names.len();
        let mut pred: Vec<Vec<u32>> = vec![Vec::new(); n];
        for s in (0..n).filter(|&s| !self.closed[s]) {
            for &(_, t) in &self.succ[s] {
                pred[t as usize].push(s as u32);
            }
        }
        let mut reaches_open: Vec<bool> = self.complete.iter().map(|&c| !c).collect();
        let mut stack: Vec<u32> = (0..n as u32).filter(|&s| reaches_open[s as usize]).collect();
        while let Some(t) = stack.pop() {
            for &s in &pred[t as usize] {
                if !reaches_open[s as usize] {
                    reaches_open[s as usize] = true;
                    stack.push(s);
                }
            }
        }
        for (closed, open) in self.closed.iter_mut().zip(reaches_open) {
            *closed |= !open;
        }
    }

    /// Estimated heap bytes (names count their inline size only).
    fn heap_bytes(&self) -> usize {
        self.ids.capacity() * (size_of::<(S, u32)>() + 1)
            + self.names.capacity() * size_of::<S>()
            + self.succ.capacity() * size_of::<Vec<(u32, u32)>>()
            + self
                .succ
                .iter()
                .map(|v| v.capacity() * size_of::<(u32, u32)>())
                .sum::<usize>()
            + self.complete.capacity()
            + self.closed.capacity()
            + self.class_of.capacity() * size_of::<u32>()
    }
}

/// Both systems and the shared label table.
#[derive(Debug)]
pub(crate) struct Store<S, L> {
    pub(crate) sides: [SideStore<S>; 2],
    label_ids: HashMap<L, u32>,
    pub(crate) labels: Vec<L>,
    /// The minimised closed parts: `(label, class)` successors per class,
    /// shared by both sides.
    class_succ: Vec<Vec<(u32, u32)>>,
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone> Store<S, L> {
    pub(crate) fn new() -> Self {
        Self {
            sides: [SideStore::new(), SideStore::new()],
            label_ids: HashMap::new(),
            labels: Vec::new(),
            class_succ: Vec::new(),
        }
    }

    /// Updates which states are closed; with `minimise`, also gives every
    /// closed state its bisimulation class.
    pub(crate) fn settle(&mut self, minimise: bool) {
        for side in &mut self.sides {
            side.settle();
        }
        if minimise {
            self.minimise();
        }
    }

    /// Classes for the newly closed states. The closed part only grows and
    /// never changes below a closed state, so the classes found earlier are
    /// final: refining the quotient (one node per class) together with the
    /// new states either adds a new state to a class or makes a new class.
    fn minimise(&mut self) {
        let classes = self.class_succ.len();
        let mut fresh: Vec<(usize, u32)> = Vec::new();
        let mut node: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for (side, st) in self.sides.iter().enumerate() {
            node[side] = st.class_of.clone();
            for (s, slot) in node[side].iter_mut().enumerate() {
                if st.closed[s] && *slot == NO_CLASS {
                    *slot = (classes + fresh.len()) as u32;
                    fresh.push((side, s as u32));
                }
            }
        }
        if fresh.is_empty() {
            return;
        }
        let mut b = WltsBuilder::<u32, bool>::new();
        b.ensure_states(classes + fresh.len());
        for (c, succ) in self.class_succ.iter().enumerate() {
            for &(l, d) in succ {
                b.add_transition(StateId(c as u32), l, StateId(d), true);
            }
        }
        for (i, &(side, s)) in fresh.iter().enumerate() {
            for &(l, t) in &self.sides[side].succ[s as usize] {
                let target = node[side][t as usize];
                debug_assert_ne!(target, NO_CLASS, "closed states reach closed states");
                b.add_transition(StateId((classes + i) as u32), l, StateId(target), true);
            }
        }
        let quotient = b.build();
        let partition = refine_fast(&quotient, None);
        let mut class_of_block = vec![NO_CLASS; partition.num_blocks()];
        for c in 0..classes {
            let block = partition.block_of(StateId(c as u32)) as usize;
            debug_assert_eq!(class_of_block[block], NO_CLASS, "classes are distinct");
            class_of_block[block] = c as u32;
        }
        let mut representatives = Vec::new();
        for i in classes..classes + fresh.len() {
            let block = partition.block_of(StateId(i as u32)) as usize;
            if class_of_block[block] == NO_CLASS {
                class_of_block[block] = (classes + representatives.len()) as u32;
                representatives.push(i);
            }
        }
        for &i in &representatives {
            let mut succ: Vec<(u32, u32)> = quotient
                .successors(StateId(i as u32))
                .iter()
                .map(|e| {
                    let block = partition.block_of(e.target) as usize;
                    (*quotient.label(e.label), class_of_block[block])
                })
                .collect();
            succ.sort_unstable();
            succ.dedup();
            succ.shrink_to_fit();
            self.class_succ.push(succ);
        }
        for (i, &(side, s)) in fresh.iter().enumerate() {
            let block = partition.block_of(StateId((classes + i) as u32)) as usize;
            self.sides[side].class_of[s as usize] = class_of_block[block];
        }
    }

    /// Estimated heap bytes of both systems and the label table (labels
    /// and names count their inline size only).
    pub(crate) fn heap_bytes(&self) -> usize {
        self.sides.iter().map(SideStore::heap_bytes).sum::<usize>()
            + self.label_ids.capacity() * (size_of::<(L, u32)>() + 1)
            + self.labels.capacity() * size_of::<L>()
            + self.class_succ.capacity() * size_of::<Vec<(u32, u32)>>()
            + self
                .class_succ
                .iter()
                .map(|v| v.capacity() * size_of::<(u32, u32)>())
                .sum::<usize>()
    }

    pub(crate) fn label(&mut self, l: L) -> u32 {
        if let Some(&i) = self.label_ids.get(&l) {
            return i;
        }
        let i = self.labels.len() as u32;
        self.label_ids.insert(l.clone(), i);
        self.labels.push(l);
        i
    }
}

/// A Spoiler move from a position, with Duplicator's answers so far.
#[derive(Debug)]
struct Move {
    /// Spoiler moves the engine's right state.
    from_right: bool,
    label: u32,
    /// Where Spoiler's state goes.
    target: u32,
    answers: Vec<usize>,
    /// Answers not refuted yet.
    open: usize,
    /// Some answer holds forever: the move can never refute, so its
    /// answers are no longer kept.
    safe: bool,
}

#[derive(Debug)]
struct Position {
    /// `(left, right)` in the engine's orientation.
    pair: (u32, u32),
    moves: Vec<Move>,
    /// `(position, move)` for every move this position answers.
    answering: Vec<(usize, usize)>,
    refuted_by: Option<usize>,
    /// Refuted and compacted: only the refuting move is kept, and new
    /// transitions only discover pairs (see [`LiveGame::discover`]).
    compacted: bool,
}

/// Where an answer leads.
enum Found {
    Position(usize),
    /// A settled pair that holds forever.
    Held,
    /// The pair budget ran out.
    Exhausted,
}

/// One game (simulation or bisimulation, in one direction), solved
/// incrementally over a [`Store`].
#[derive(Debug)]
pub(crate) struct LiveGame {
    /// The engine's left side is the store's right side.
    flip: bool,
    /// Spoiler only moves on the left.
    simulation: bool,
    max_pairs: usize,
    /// Positions discovered so far, compacted or not (the budget counts
    /// these, as the batch game does).
    discovered: usize,
    index: HashMap<(u32, u32), usize>,
    positions: Vec<Position>,
    /// Positions involving each open state, per engine side.
    by_state: [Vec<Vec<usize>>; 2],
    /// Settled pairs that hold forever, dropped from `positions`.
    held_pairs: HashSet<(u32, u32)>,
    /// The same per pair of classes, once minimised.
    held_classes: HashSet<(u32, u32)>,
    /// A refuted position per pair of classes, once minimised.
    refuted_classes: HashMap<(u32, u32), usize>,
    /// Positions where some state is still open.
    open_positions: usize,
    /// The pair budget ran out; the game stopped growing.
    exhausted: bool,
    queue: VecDeque<usize>,
}

impl LiveGame {
    /// A game from the initial pair `(left, right)` of the store
    /// (swapped when `flip`).
    pub(crate) fn new<S, L>(
        store: &Store<S, L>,
        roots: (u32, u32),
        flip: bool,
        simulation: bool,
        max_pairs: usize,
    ) -> Self {
        let mut game = Self {
            flip,
            simulation,
            max_pairs,
            discovered: 0,
            index: HashMap::new(),
            positions: Vec::new(),
            by_state: [Vec::new(), Vec::new()],
            held_pairs: HashSet::new(),
            held_classes: HashSet::new(),
            refuted_classes: HashMap::new(),
            open_positions: 0,
            exhausted: false,
            queue: VecDeque::new(),
        };
        let pair = if flip { (roots.1, roots.0) } else { roots };
        game.position(store, pair);
        game.drain(store);
        game
    }

    /// Store side of an engine side.
    fn side(&self, engine_side: usize) -> usize {
        if self.flip {
            1 - engine_side
        } else {
            engine_side
        }
    }

    fn is_complete<S, L>(&self, store: &Store<S, L>, engine_side: usize, s: u32) -> bool {
        store.sides[self.side(engine_side)].complete[s as usize]
    }

    /// The pair's classes, if both states have one.
    fn class_key<S, L>(&self, store: &Store<S, L>, (x, y): (u32, u32)) -> Option<(u32, u32)> {
        let cx = store.sides[self.side(0)].class_of[x as usize];
        let cy = store.sides[self.side(1)].class_of[y as usize];
        (cx != NO_CLASS && cy != NO_CLASS).then_some((cx, cy))
    }

    fn position<S, L>(&mut self, store: &Store<S, L>, pair: (u32, u32)) -> Found {
        if let Some(&p) = self.index.get(&pair) {
            return Found::Position(p);
        }
        if self.held_pairs.contains(&pair) {
            return Found::Held;
        }
        // A pair of closed states is decided by its classes: the batch game
        // would explore it, but only closed (complete) positions lie below.
        if let Some(key) = self.class_key(store, pair) {
            if self.held_classes.contains(&key) {
                return Found::Held;
            }
            if let Some(&p) = self.refuted_classes.get(&key) {
                return Found::Position(p);
            }
        }
        if self.discovered >= self.max_pairs {
            self.exhausted = true;
            return Found::Exhausted;
        }
        self.discovered += 1;
        let p = self.positions.len();
        self.positions.push(Position {
            pair,
            moves: Vec::new(),
            answering: Vec::new(),
            refuted_by: None,
            compacted: false,
        });
        self.index.insert(pair, p);
        if !self.is_complete(store, 0, pair.0) || !self.is_complete(store, 1, pair.1) {
            self.open_positions += 1;
            self.track(p);
        }
        self.queue.push_back(p);
        Found::Position(p)
    }

    /// Lists `p` under its states, so their events reach it.
    fn track(&mut self, p: usize) {
        let pair = self.positions[p].pair;
        for (side, s) in [(0, pair.0), (1, pair.1)] {
            let list = &mut self.by_state[side];
            if list.len() <= s as usize {
                list.resize_with(s as usize + 1, Vec::new);
            }
            list[s as usize].push(p);
        }
    }

    /// Expands newly discovered positions, breadth first.
    fn drain<S, L>(&mut self, store: &Store<S, L>) {
        while let Some(p) = self.queue.pop_front() {
            if self.exhausted {
                self.queue.clear();
                return;
            }
            let (x, y) = self.positions[p].pair;
            for (from_right, mover) in [(false, x), (true, y)] {
                if from_right && self.simulation {
                    continue;
                }
                let side = self.side(usize::from(from_right));
                for &(label, target) in &store.sides[side].succ[mover as usize] {
                    self.add_move(store, p, from_right, label, target);
                }
            }
        }
    }

    /// Adds Spoiler's move `label → target` at `p`, with every answer the
    /// answering state offers now.
    fn add_move<S, L>(
        &mut self,
        store: &Store<S, L>,
        p: usize,
        from_right: bool,
        label: u32,
        target: u32,
    ) {
        let (x, y) = self.positions[p].pair;
        let (answer_side, answerer) = if from_right { (0, x) } else { (1, y) };
        let side = self.side(answer_side);
        let mut answers = Vec::new();
        let mut safe = false;
        for &(l, t) in &store.sides[side].succ[answerer as usize] {
            if l != label {
                continue;
            }
            let pair = if from_right { (t, target) } else { (target, t) };
            match self.position(store, pair) {
                Found::Position(q) => answers.push(q),
                Found::Held => safe = true,
                Found::Exhausted => return,
            }
        }
        let m = self.positions[p].moves.len();
        let open = if safe {
            answers.clear();
            1
        } else {
            answers
                .iter()
                .filter(|&&q| self.positions[q].refuted_by.is_none())
                .count()
        };
        for &q in &answers {
            // A refuted answer never propagates again.
            if self.positions[q].refuted_by.is_none() {
                self.positions[q].answering.push((p, m));
            }
        }
        self.positions[p].moves.push(Move {
            from_right,
            label,
            target,
            answers,
            open,
            safe,
        });
        // A position already lost must not be refuted (and propagated) twice.
        if open == 0
            && self.positions[p].refuted_by.is_none()
            && self.is_complete(store, answer_side, answerer)
        {
            self.refute(store, p, m);
        }
    }

    /// The store has just gained `s —label→ t` on store side `side`.
    pub(crate) fn on_edge<S, L>(
        &mut self,
        store: &Store<S, L>,
        side: usize,
        s: u32,
        label: u32,
        t: u32,
    ) {
        if self.exhausted || self.decided() {
            return;
        }
        let engine_side = self.side(side);
        let Some(existing) = self.by_state[engine_side].get(s as usize).cloned() else {
            return;
        };
        for p in existing {
            if self.exhausted {
                return;
            }
            if self.positions[p].compacted {
                if !self.discover(store, p, engine_side, label, t) {
                    return;
                }
                continue;
            }
            // `s` moves: a new Spoiler move.
            if engine_side == 0 || !self.simulation {
                self.add_move(store, p, engine_side == 1, label, t);
            }
            // `s` answers: a new answer to the other side's moves.
            let answers_from_right = engine_side == 0;
            for m in 0..self.positions[p].moves.len() {
                let mv = &self.positions[p].moves[m];
                if mv.from_right != answers_from_right || mv.label != label {
                    continue;
                }
                let pair = if answers_from_right {
                    (t, mv.target)
                } else {
                    (mv.target, t)
                };
                let q = match self.position(store, pair) {
                    Found::Position(q) => q,
                    Found::Held => {
                        let mv = &mut self.positions[p].moves[m];
                        if !mv.safe {
                            mv.safe = true;
                            mv.open += 1;
                            mv.answers = Vec::new();
                        }
                        continue;
                    }
                    Found::Exhausted => return,
                };
                if self.positions[p].moves[m].safe {
                    continue;
                }
                self.positions[p].moves[m].answers.push(q);
                if self.positions[q].refuted_by.is_none() {
                    self.positions[q].answering.push((p, m));
                    self.positions[p].moves[m].open += 1;
                }
            }
        }
        self.drain(store);
    }

    /// The new transition `s —label→ t` (`s` on `engine_side`) at a
    /// compacted, refuted position `p`: its outcome is fixed, but the pairs
    /// the new moves and answers lead to are still discovered, as the
    /// batch game explores them. False once the budget runs out.
    fn discover<S, L>(
        &mut self,
        store: &Store<S, L>,
        p: usize,
        engine_side: usize,
        label: u32,
        t: u32,
    ) -> bool {
        let (x, y) = self.positions[p].pair;
        // `s` moves: Duplicator's answers from the other state.
        if engine_side == 0 || !self.simulation {
            let (answer_side, answerer) = if engine_side == 0 { (1, y) } else { (0, x) };
            for &(l, u) in &store.sides[self.side(answer_side)].succ[answerer as usize] {
                let pair = if engine_side == 0 { (t, u) } else { (u, t) };
                if l == label && matches!(self.position(store, pair), Found::Exhausted) {
                    return false;
                }
            }
        }
        // `s` answers: the other state's moves with this label.
        if engine_side == 1 || !self.simulation {
            let (mover_side, mover) = if engine_side == 0 { (1, y) } else { (0, x) };
            for &(l, u) in &store.sides[self.side(mover_side)].succ[mover as usize] {
                let pair = if engine_side == 0 { (t, u) } else { (u, t) };
                if l == label && matches!(self.position(store, pair), Found::Exhausted) {
                    return false;
                }
            }
        }
        true
    }

    /// Store side `side` has just declared `s` complete.
    pub(crate) fn on_complete<S, L>(&mut self, store: &Store<S, L>, side: usize, s: u32) {
        if self.decided() {
            return;
        }
        let engine_side = self.side(side);
        let Some(existing) = self.by_state[engine_side].get(s as usize).cloned() else {
            return;
        };
        let answers_from_right = engine_side == 0;
        for p in existing {
            let (x, y) = self.positions[p].pair;
            let other = if engine_side == 0 { (1, y) } else { (0, x) };
            if self.is_complete(store, other.0, other.1) {
                self.open_positions -= 1;
            }
            if self.exhausted || self.positions[p].refuted_by.is_some() {
                continue;
            }
            let lost = self.positions[p]
                .moves
                .iter()
                .position(|mv| mv.from_right == answers_from_right && mv.open == 0);
            if let Some(m) = lost {
                self.refute(store, p, m);
            }
        }
    }

    /// Spoiler wins `p` with move `m`; propagates to the moves `p` answers.
    /// A move left without answers refutes only once its answerer is
    /// complete; otherwise [`on_complete`](Self::on_complete) picks it up.
    fn refute<S, L>(&mut self, store: &Store<S, L>, p: usize, m: usize) {
        self.positions[p].refuted_by = Some(m);
        // First in, first out keeps witnesses shallow (as in the batch game).
        let mut lost = VecDeque::from([p]);
        while let Some(q) = lost.pop_front() {
            for i in 0..self.positions[q].answering.len() {
                let (r, m) = self.positions[q].answering[i];
                if self.positions[r].refuted_by.is_some() {
                    continue;
                }
                let mv = &mut self.positions[r].moves[m];
                if mv.safe {
                    continue;
                }
                mv.open -= 1;
                if mv.open > 0 {
                    continue;
                }
                let from_right = mv.from_right;
                let (x, y) = self.positions[r].pair;
                let answerer_complete = if from_right {
                    self.is_complete(store, 0, x)
                } else {
                    self.is_complete(store, 1, y)
                };
                if answerer_complete {
                    self.positions[r].refuted_by = Some(m);
                    lost.push_back(r);
                }
            }
        }
    }

    /// The verdict for the initial pair on the prefix seen so far.
    pub(crate) fn verdict<S, L: Clone>(&self, store: &Store<S, L>) -> Verdict<Formula<L, bool>> {
        if self.positions.is_empty() {
            Verdict::Unknown
        } else if self.positions[0].refuted_by.is_some() {
            let witness = self.witness(store, 0, &mut HashMap::new());
            Verdict::Violated { witness }
        } else if self.exhausted {
            Verdict::Unknown
        } else if self.open_positions > 0 {
            Verdict::HoldsSoFar
        } else {
            Verdict::Holds
        }
    }

    /// The initial position is refuted: nothing can change the verdict.
    fn decided(&self) -> bool {
        self.positions
            .first()
            .is_some_and(|root| root.refuted_by.is_some())
    }

    /// Number of positions discovered so far.
    pub(crate) fn pairs(&self) -> usize {
        self.discovered
    }

    /// Positions kept in memory: the unsettled ones, the refuted ones and
    /// the initial one.
    pub(crate) fn live_positions(&self) -> usize {
        self.positions.len()
    }

    /// Settled pairs (or pairs of classes) remembered as holding forever.
    pub(crate) fn held(&self) -> usize {
        self.held_pairs.len() + self.held_classes.len()
    }

    /// Estimated heap bytes of the game.
    pub(crate) fn heap_bytes(&self) -> usize {
        let positions = self
            .positions
            .iter()
            .map(|pos| {
                pos.moves.capacity() * size_of::<Move>()
                    + pos
                        .moves
                        .iter()
                        .map(|mv| mv.answers.capacity() * size_of::<usize>())
                        .sum::<usize>()
                    + pos.answering.capacity() * size_of::<(usize, usize)>()
            })
            .sum::<usize>();
        let lists = |lists: &Vec<Vec<usize>>| {
            lists.capacity() * size_of::<Vec<usize>>()
                + lists
                    .iter()
                    .map(|l| l.capacity() * size_of::<usize>())
                    .sum::<usize>()
        };
        self.positions.capacity() * size_of::<Position>()
            + positions
            + self.index.capacity() * (size_of::<((u32, u32), usize)>() + 1)
            + lists(&self.by_state[0])
            + lists(&self.by_state[1])
            + (self.held_pairs.capacity() + self.held_classes.capacity())
                * (size_of::<(u32, u32)>() + 1)
            + self.refuted_classes.capacity() * (size_of::<((u32, u32), usize)>() + 1)
            + self.queue.capacity() * size_of::<usize>()
    }

    /// Collapses the settled positions (see the module docs), given the
    /// closed states (and classes, with `minimise`) of
    /// [`Store::settle`]. Costs time linear in the positions kept; the
    /// verdict is unchanged.
    pub(crate) fn compact<S, L>(&mut self, store: &Store<S, L>, minimise: bool) {
        debug_assert!(self.queue.is_empty(), "compaction runs between events");
        let n = self.positions.len();
        if n == 0 {
            return;
        }
        if self.decided() {
            self.keep_witness();
            return;
        }
        let closed = |side: usize, s: u32| store.sides[self.side(side)].closed[s as usize];
        let held: Vec<bool> = self
            .positions
            .iter()
            .map(|pos| {
                pos.refuted_by.is_none() && closed(0, pos.pair.0) && closed(1, pos.pair.1)
            })
            .collect();

        // Unsettled positions; the rest are rebuilt below.
        let unsettled: Vec<usize> = (0..n)
            .filter(|&p| !held[p] && self.positions[p].refuted_by.is_none())
            .collect();
        // Moves answered by a held position can never refute.
        for &p in &unsettled {
            for mv in &mut self.positions[p].moves {
                if !mv.safe && mv.answers.iter().any(|&q| held[q]) {
                    mv.safe = true;
                    mv.answers = Vec::new();
                }
            }
        }
        // Back references that can no longer propagate a refutation.
        for &p in &unsettled {
            let answering: Vec<(usize, usize)> = self.positions[p]
                .answering
                .iter()
                .copied()
                .filter(|&(r, m)| {
                    let pos = &self.positions[r];
                    !held[r] && pos.refuted_by.is_none() && !pos.moves[m].safe
                })
                .collect();
            self.positions[p].answering = answering;
        }

        // The initial position keeps slot 0 even when held: the verdict
        // reads it.
        let mut new_id = vec![usize::MAX; n];
        let mut kept = 0;
        for p in (0..n).filter(|&p| p == 0 || !held[p]) {
            new_id[p] = kept;
            kept += 1;
        }
        let old = std::mem::take(&mut self.positions);
        let mut positions = Vec::with_capacity(kept);
        for (p, mut pos) in old.into_iter().enumerate() {
            if held[p] {
                if minimise {
                    let key = self.class_key(store, pos.pair).expect("held pairs are closed");
                    self.held_classes.insert(key);
                } else {
                    self.held_pairs.insert(pos.pair);
                }
                if p != 0 {
                    continue;
                }
                pos.moves = Vec::new();
                pos.answering = Vec::new();
            } else if let Some(m) = pos.refuted_by {
                // Only the refuting move matters now: it is the witness,
                // and its answers are refuted, hence kept.
                let mut mv = pos.moves.swap_remove(m);
                for q in &mut mv.answers {
                    *q = new_id[*q];
                }
                mv.answers.shrink_to_fit();
                pos.moves = vec![mv];
                pos.refuted_by = Some(0);
                pos.answering = Vec::new();
                pos.compacted = true;
            } else {
                for mv in &mut pos.moves {
                    for q in &mut mv.answers {
                        *q = new_id[*q];
                    }
                }
                for (r, _) in &mut pos.answering {
                    *r = new_id[*r];
                }
                pos.answering.shrink_to_fit();
            }
            positions.push(pos);
        }
        self.positions = positions;

        if minimise {
            // Every held pair is closed, hence classed now.
            for pair in std::mem::take(&mut self.held_pairs) {
                let key = self.class_key(store, pair).expect("held pairs are closed");
                self.held_classes.insert(key);
            }
            for p in self.refuted_classes.values_mut() {
                *p = new_id[*p];
            }
            for p in 0..self.positions.len() {
                let pos = &self.positions[p];
                if pos.refuted_by.is_some() {
                    if let Some(key) = self.class_key(store, pos.pair) {
                        self.refuted_classes.entry(key).or_insert(p);
                    }
                }
            }
        }

        let mut index = HashMap::with_capacity(self.positions.len());
        for (p, pos) in self.positions.iter().enumerate() {
            if !(p == 0 && held[0]) {
                index.insert(pos.pair, p);
            }
        }
        self.index = index;
        for list in self.by_state.iter_mut().flatten() {
            *list = Vec::new();
        }
        for p in 0..self.positions.len() {
            let (x, y) = self.positions[p].pair;
            if !self.is_complete(store, 0, x) || !self.is_complete(store, 1, y) {
                self.track(p);
            }
        }
        self.held_pairs.shrink_to_fit();
        self.held_classes.shrink_to_fit();
        self.queue.shrink_to_fit();
    }

    /// Keeps only the refuted positions the initial one's witness reaches
    /// through refuting moves, each with just that move.
    fn keep_witness(&mut self) {
        let mut new_id = vec![usize::MAX; self.positions.len()];
        let mut order = vec![0];
        new_id[0] = 0;
        let mut i = 0;
        while i < order.len() {
            let pos = &self.positions[order[i]];
            let m = pos.refuted_by.expect("witness positions are refuted");
            for &q in &pos.moves[m].answers {
                if new_id[q] == usize::MAX {
                    new_id[q] = order.len();
                    order.push(q);
                }
            }
            i += 1;
        }
        let mut old: Vec<Option<Position>> = std::mem::take(&mut self.positions)
            .into_iter()
            .map(Some)
            .collect();
        self.positions = order
            .iter()
            .map(|&p| {
                let mut pos = old[p].take().expect("each position once");
                let mut mv = pos.moves.swap_remove(pos.refuted_by.expect("refuted"));
                for q in &mut mv.answers {
                    *q = new_id[*q];
                }
                mv.answers.shrink_to_fit();
                Position {
                    pair: pos.pair,
                    moves: vec![mv],
                    answering: Vec::new(),
                    refuted_by: Some(0),
                    compacted: true,
                }
            })
            .collect();
        self.index = HashMap::new();
        self.by_state = [Vec::new(), Vec::new()];
        self.held_pairs = HashSet::new();
        self.held_classes = HashSet::new();
        self.refuted_classes = HashMap::new();
        self.queue = VecDeque::new();
    }

    /// A formula true in the left state of `p` and false in its right one
    /// (same construction as the batch game). Every answer of the refuting
    /// move was refuted earlier, so the recursion is well founded.
    fn witness<S, L: Clone>(
        &self,
        store: &Store<S, L>,
        p: usize,
        memo: &mut HashMap<usize, Formula<L, bool>>,
    ) -> Formula<L, bool> {
        if let Some(f) = memo.get(&p) {
            return f.clone();
        }
        let pos = &self.positions[p];
        let mv = &pos.moves[pos
            .refuted_by
            .expect("only refuted positions have witnesses")];
        let mut conj: Vec<Formula<L, bool>> = mv
            .answers
            .iter()
            .map(|&q| {
                let f = self.witness(store, q, memo);
                if mv.from_right {
                    Formula::Not(Box::new(f))
                } else {
                    f
                }
            })
            .collect();
        let target = match conj.len() {
            0 => Formula::True,
            1 => conj.pop().expect("one conjunct"),
            _ => Formula::And(conj),
        };
        let modal = Formula::Modal {
            label: store.labels[mv.label as usize].clone(),
            weight: true,
            target: Box::new(target),
        };
        let f = if mv.from_right {
            Formula::Not(Box::new(modal))
        } else {
            modal
        };
        memo.insert(p, f.clone());
        f
    }
}
