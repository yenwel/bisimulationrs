//! On-the-fly bisimulation game (Stirling) for boolean LTSs.
//!
//! [`game_bisimilar`] decides bisimilarity of two states by playing the
//! Spoiler/Duplicator game, discovering positions (pairs of states) lazily
//! and breadth first from the initial query: only pairs reachable from
//! `(s, t)` via matching moves are ever explored. A shallow disagreement is
//! found without refining the whole systems, which is what makes this the
//! on-the-fly counterpart to [`refine`](super::refine).
//!
//! Every position holds until Spoiler is shown to win it: a position is
//! refuted as soon as one of Spoiler's moves has no answer left that is not
//! refuted (counter-based propagation, linear in the explored game graph).
//! Positions that are never refuted hold coinductively, so cycles need no
//! special case and nothing is assumed and later retracted.
//!
//! Two budgets bound the search: at most `max_pairs` expanded positions and
//! distance `max_depth` from the initial pair. Exhausting the pair budget
//! reports [`Unknown`](bisim_core::Verdict::Unknown) (unless Spoiler has
//! already won). Positions beyond the depth limit are not expanded; an
//! answer that then still holds reports
//! [`HoldsUpTo`](bisim_core::Verdict::HoldsUpTo): the moves of every pair
//! up to distance `max_depth` match, i.e. `∼ₖ` for `k = max_depth + 1`.
//! A completed search reports `Holds` (exact) or a Spoiler witness
//! (`Violated`).

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

use bisim_core::{Formula, LabelId, Lts, StateId, Verdict};

/// Search budgets for [`game_bisimilar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameBudget {
    /// Maximum number of distinct pairs explored (`0` decides nothing).
    pub max_pairs: usize,
    /// Maximum distance of an explored pair from the initial one (`0` still
    /// finds immediate mismatches).
    pub max_depth: u32,
}

impl Default for GameBudget {
    fn default() -> Self {
        Self {
            max_pairs: 100_000,
            max_depth: u32::MAX,
        }
    }
}

/// Decides whether state `s` of `left` and state `t` of `right` are
/// bisimilar, exploring pairs on demand under `budget`. A violation
/// carries an existential HML formula true in `s` and false in `t`
/// (or its negation when Spoiler wins from the right).
pub fn game_bisimilar<L: Eq + Hash + Clone>(
    left: &Lts<L>,
    s: StateId,
    right: &Lts<L>,
    t: StateId,
    budget: GameBudget,
) -> Verdict<Formula<L, bool>> {
    let union = left.disjoint_union(right);
    let t = StateId(t.0 + left.num_states() as u32);
    Game::new(&union, budget).play(s, t)
}

/// A Spoiler move: `(from_right, label, answers)`, each answer the
/// `(left, right)` pair Duplicator reaches.
pub(crate) type SpoilerMove = (bool, LabelId, Vec<(StateId, StateId)>);

/// Spoiler's moves from the position `(x, y)` of `sys`, as
/// `(from_right, label, answers)`: each answer is the `(left, right)` pair
/// Duplicator can reach with a matching transition. With `simulation`,
/// Spoiler moves only on the left.
pub(crate) fn spoiler_moves<L: Eq + Hash + Clone>(
    sys: &Lts<L>,
    (x, y): (StateId, StateId),
    simulation: bool,
) -> Vec<SpoilerMove> {
    let mut moves = Vec::new();
    for (from_right, mover, answerer) in [(false, x, y), (true, y, x)] {
        if from_right && simulation {
            continue;
        }
        for em in sys.successors(mover) {
            let answers = sys
                .successors(answerer)
                .iter()
                .filter(|ea| ea.label == em.label)
                .map(|ea| {
                    if from_right {
                        (ea.target, em.target)
                    } else {
                        (em.target, ea.target)
                    }
                })
                .collect();
            moves.push((from_right, em.label, answers));
        }
    }
    moves
}

/// One Spoiler move from a position, with Duplicator's answers.
struct Move {
    /// Spoiler moves the right state (and Duplicator answers on the left).
    from_right: bool,
    label: LabelId,
    /// Positions reached by each answer, as `(left, right)` pairs.
    answers: Vec<usize>,
    /// Answers not refuted yet: the position is lost when this reaches 0.
    open: usize,
    /// The answering state may still gain transitions, so having no answer
    /// left is not yet a refutation.
    answerer_open: bool,
}

/// A pair of states `(left, right)` of the game.
struct Position {
    pair: (StateId, StateId),
    /// Distance from the initial pair.
    depth: u32,
    moves: Vec<Move>,
    /// `(position, move)` for every move this position answers.
    answering: Vec<(usize, usize)>,
    /// The move Duplicator cannot answer, once Spoiler has won here.
    refuted_by: Option<usize>,
}

pub(crate) struct Game<'a, L> {
    sys: &'a Lts<L>,
    budget: GameBudget,
    index: HashMap<(StateId, StateId), usize>,
    positions: Vec<Position>,
    /// Discovered positions not expanded yet, in breadth-first order.
    queue: VecDeque<usize>,
    /// Positions expanded so far (the work measure the budget bounds).
    pub(crate) expanded: usize,
    /// States that may still gain transitions (a growing prefix); indexed
    /// by state, empty when every state is final.
    unfinished: Vec<bool>,
    /// Spoiler only moves on the left: the simulation game.
    simulation: bool,
    /// Some explored position involved an unfinished state.
    saw_unfinished: bool,
}

impl<'a, L: Eq + Hash + Clone> Game<'a, L> {
    pub(crate) fn new(sys: &'a Lts<L>, budget: GameBudget) -> Self {
        Self {
            sys,
            budget,
            index: HashMap::new(),
            positions: Vec::new(),
            queue: VecDeque::new(),
            expanded: 0,
            unfinished: Vec::new(),
            simulation: false,
            saw_unfinished: false,
        }
    }

    /// A game on a prefix: `unfinished` states may still gain transitions,
    /// so a missing answer from one of them does not refute; `simulation`
    /// restricts Spoiler to the left state.
    pub(crate) fn on_prefix(
        sys: &'a Lts<L>,
        budget: GameBudget,
        unfinished: Vec<bool>,
        simulation: bool,
    ) -> Self {
        Self {
            unfinished,
            simulation,
            ..Self::new(sys, budget)
        }
    }

    fn is_unfinished(&self, s: StateId) -> bool {
        self.unfinished.get(s.index()).copied().unwrap_or(false)
    }

    pub(crate) fn play(&mut self, s: StateId, t: StateId) -> Verdict<Formula<L, bool>> {
        let root = self.position((s, t), 0);
        let (mut capped, mut exhausted) = (false, false);
        while let Some(p) = self.queue.pop_front() {
            // Breadth first: every later position is at least as deep.
            if self.positions[p].depth > self.budget.max_depth {
                capped = true;
                break;
            }
            if self.expanded >= self.budget.max_pairs {
                exhausted = true;
                break;
            }
            self.expanded += 1;
            self.expand(p);
            if self.positions[root].refuted_by.is_some() {
                let witness = self.witness(root, &mut HashMap::new());
                return Verdict::Violated { witness };
            }
        }
        if exhausted {
            Verdict::Unknown
        } else if capped {
            Verdict::HoldsUpTo {
                depth: self.budget.max_depth,
            }
        } else if self.saw_unfinished {
            Verdict::HoldsSoFar
        } else {
            Verdict::Holds
        }
    }

    /// The position of `pair`, discovered at `depth` if it is new.
    fn position(&mut self, pair: (StateId, StateId), depth: u32) -> usize {
        if let Some(&p) = self.index.get(&pair) {
            return p;
        }
        let p = self.positions.len();
        self.positions.push(Position {
            pair,
            depth,
            moves: Vec::new(),
            answering: Vec::new(),
            refuted_by: None,
        });
        self.index.insert(pair, p);
        self.queue.push_back(p);
        p
    }

    /// Lists Spoiler's moves from `p` and Duplicator's answers, then
    /// refutes `p` if some move is already unanswerable.
    fn expand(&mut self, p: usize) {
        let sys = self.sys;
        let ((x, y), depth) = (self.positions[p].pair, self.positions[p].depth);
        let mut moves = Vec::new();
        self.saw_unfinished |= self.is_unfinished(x) || self.is_unfinished(y);
        for (from_right, label, pairs) in spoiler_moves(sys, (x, y), self.simulation) {
            let answerer_open = self.is_unfinished(if from_right { x } else { y });
            let answers: Vec<usize> = pairs
                .into_iter()
                .map(|pair| self.position(pair, depth + 1))
                .collect();
            let open = answers
                .iter()
                .filter(|&&q| self.positions[q].refuted_by.is_none())
                .count();
            moves.push(Move {
                from_right,
                label,
                answers,
                open,
                answerer_open,
            });
        }
        for (m, mv) in moves.iter().enumerate() {
            for &q in &mv.answers {
                self.positions[q].answering.push((p, m));
            }
        }
        let lost = moves
            .iter()
            .position(|mv| mv.open == 0 && !mv.answerer_open);
        self.positions[p].moves = moves;
        if let Some(m) = lost {
            self.refute(p, m);
        }
    }

    /// Spoiler wins `p` with move `m`; propagates to the moves `p` answers.
    fn refute(&mut self, p: usize, m: usize) {
        self.positions[p].refuted_by = Some(m);
        // First in, first out: positions are refuted in order of the number
        // of Spoiler moves needed, which keeps witnesses shallow.
        let mut lost = VecDeque::from([p]);
        while let Some(q) = lost.pop_front() {
            for i in 0..self.positions[q].answering.len() {
                let (r, m) = self.positions[q].answering[i];
                if self.positions[r].refuted_by.is_some() {
                    continue;
                }
                let mv = &mut self.positions[r].moves[m];
                mv.open -= 1;
                if mv.open == 0 && !mv.answerer_open {
                    self.positions[r].refuted_by = Some(m);
                    lost.push_back(r);
                }
            }
        }
    }

    /// A formula true in the left state of `p` and false in its right one.
    /// Every answer of the refuting move was refuted earlier, so this
    /// recursion is well founded.
    fn witness(&self, p: usize, memo: &mut HashMap<usize, Formula<L, bool>>) -> Formula<L, bool> {
        if let Some(f) = memo.get(&p) {
            return f.clone();
        }
        let pos = &self.positions[p];
        let mv = &pos.moves[pos
            .refuted_by
            .expect("only refuted positions have witnesses")];
        // Each answer's witness holds on its left state; when Spoiler moved
        // on the right, it must hold on the right instead: negate it.
        let mut conj: Vec<Formula<L, bool>> = mv
            .answers
            .iter()
            .map(|&q| {
                let f = self.witness(q, memo);
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
            label: self.sys.label(mv.label).clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::bisimilar_states;
    use bisim_core::NamedBuilder;

    fn budget() -> GameBudget {
        GameBudget::default()
    }

    #[test]
    fn finds_shallow_mismatch_without_full_refinement() {
        // a.a.0 vs a.0: Spoiler wins in two plies.
        let (sys, _) = NamedBuilder::from_triples([("p0", 'a', "p1"), ("p1", 'a', "p2")]);
        let v = game_bisimilar(&sys, StateId(0), &sys, StateId(1), budget());
        let Verdict::Violated { witness } = v else {
            panic!("expected a violation, got {v:?}");
        };
        assert_eq!(witness.to_string(), "⟨a⟩⟨a⟩tt");
        assert!(witness.holds(&sys.disjoint_union(&sys), StateId(0)));
        // Pairs explored: only the reachable ones, not the full square.
        assert!(solver_pairs(&sys) < sys.num_states() * sys.num_states());
    }

    fn solver_pairs(sys: &Lts<char>) -> usize {
        let union = sys.disjoint_union(sys);
        let mut game = Game::new(&union, budget());
        let _ = game.play(StateId(0), StateId(1));
        game.expanded
    }

    #[test]
    fn long_chains_do_not_overflow_the_stack() {
        // Solving does not recurse: identical chains of 200,000 states
        // used to overflow the stack (one recursion level per pair).
        let mut b = bisim_core::WltsBuilder::<char, bool>::new();
        for i in 0..200_000 {
            b.add_transition(StateId(i), 'a', StateId(i + 1), true);
        }
        let chain = b.build();
        let unlimited = GameBudget {
            max_pairs: usize::MAX,
            max_depth: u32::MAX,
        };
        let v = game_bisimilar(&chain, StateId(0), &chain, StateId(0), unlimited);
        assert_eq!(v, Verdict::Holds);
    }

    #[test]
    fn agrees_with_refinement_on_all_pairs() {
        // Chain, cycle and branching: every pair, both directions.
        let (sys, _) = NamedBuilder::from_triples([
            ("c0", 'a', "c1"),
            ("c1", 'a', "c2"),
            ("y0", 'a', "y0"),
            ("y1", 'a', "y0"),
            ("b", 'a', "b1"),
            ("b", 'a', "b2"),
            ("b1", 'b', "b1"),
        ]);
        for s in sys.states() {
            for t in sys.states() {
                let g = game_bisimilar(&sys, s, &sys, t, budget());
                let r = bisimilar_states(&sys, s, &sys, t);
                assert_eq!(g.holds(), r.holds(), "pair {s:?} vs {t:?}");
                if let Verdict::Violated { witness } = g {
                    let union = sys.disjoint_union(&sys);
                    let tt = StateId(t.0 + sys.num_states() as u32);
                    assert!(witness.holds(&union, s), "witness holds in left: {witness}");
                    assert!(
                        !witness.holds(&union, tt),
                        "witness fails in right: {witness}"
                    );
                }
            }
        }
    }

    #[test]
    fn budgets_report_unknown_and_holds_up_to() {
        let (sys, _) = NamedBuilder::from_triples([("p0", 'a', "p1"), ("p1", 'a', "p2")]);
        // No pairs allowed: nothing can be decided.
        let v = game_bisimilar(
            &sys,
            StateId(0),
            &sys,
            StateId(0),
            GameBudget {
                max_pairs: 0,
                max_depth: u32::MAX,
            },
        );
        assert_eq!(v, Verdict::Unknown);
        // Depth 0 on a pair that needs depth: holds so far, at depth 0.
        // (A self-loop pair would instead close by the cycle rule: exact.)
        let v = game_bisimilar(
            &sys,
            StateId(0),
            &sys,
            StateId(0),
            GameBudget {
                max_pairs: 100,
                max_depth: 0,
            },
        );
        assert_eq!(v, Verdict::HoldsUpTo { depth: 0 });
        // Depth 0 still finds immediate mismatches (no recursion needed).
        let v = game_bisimilar(
            &sys,
            StateId(2),
            &sys,
            StateId(0),
            GameBudget {
                max_pairs: 100,
                max_depth: 0,
            },
        );
        assert!(matches!(v, Verdict::Violated { .. }));
    }

    /// State of `sys` named `name` (from `NamedBuilder::from_triples`).
    fn state(names: &[&str], name: &str) -> StateId {
        StateId(names.iter().position(|n| *n == name).expect("named state") as u32)
    }

    /// Asserts a violation whose witness holds in `s` and fails in `t`.
    fn assert_violated(sys: &Lts<char>, s: StateId, t: StateId, v: Verdict<Formula<char, bool>>) {
        let Verdict::Violated { witness } = v else {
            panic!("expected a violation, got {v:?}");
        };
        let union = sys.disjoint_union(sys);
        let tt = StateId(t.0 + sys.num_states() as u32);
        assert!(witness.holds(&union, s), "witness fails in left: {witness}");
        assert!(
            !witness.holds(&union, tt),
            "witness holds in right: {witness}"
        );
    }

    #[test]
    fn failed_assumptions_are_not_reused() {
        // x ≁ y (only x has a d-move), hence x' ≁ y' (x' -c-> x, y' -c-> y)
        // and r1 ≁ r2 (their only e-moves reach x' and y'). Solving the
        // a-moves first explores (x', y') under the assumption that (x, y)
        // holds; that assumption fails, so the answer for (x', y') must not
        // be reused for the e-moves. (z copies x and w copies y, so every
        // a-move is answered.)
        let (sys, names) = NamedBuilder::from_triples([
            ("r1", 'a', "x"),
            ("r1", 'a', "w"),
            ("r2", 'a', "y"),
            ("r2", 'a', "z"),
            ("r1", 'e', "x'"),
            ("r2", 'e', "y'"),
            ("x", 'b', "x'"),
            ("x'", 'c', "x"),
            ("x", 'd', "xd"),
            ("y", 'b', "y'"),
            ("y'", 'c', "y"),
            ("z", 'b', "z'"),
            ("z'", 'c', "z"),
            ("z", 'd', "zd"),
            ("w", 'b', "w'"),
            ("w'", 'c', "w"),
        ]);
        let (r1, r2) = (state(&names, "r1"), state(&names, "r2"));
        assert!(!bisimilar_states(&sys, r1, &sys, r2).holds());
        let v = game_bisimilar(&sys, r1, &sys, r2, budget());
        assert_violated(&sys, r1, r2, v);
    }

    #[test]
    fn depth_capped_answers_are_not_reused_closer_to_the_root() {
        // x and y differ in two steps (⟨a⟩⟨b⟩tt). The l-path reaches (x, y)
        // at depth 3, where the depth limit hides the difference; the m-move
        // reaches it at depth 1, where it is in range and must be found.
        let (sys, names) = NamedBuilder::from_triples([
            ("r", 'l', "r1"),
            ("r1", 'l', "r2"),
            ("r2", 'l', "x"),
            ("s", 'l', "s1"),
            ("s1", 'l', "s2"),
            ("s2", 'l', "y"),
            ("r", 'm', "x"),
            ("s", 'm', "y"),
            ("x", 'a', "x1"),
            ("x1", 'b', "x2"),
            ("y", 'a', "y1"),
            ("y1", 'c', "y2"),
        ]);
        let (r, s) = (state(&names, "r"), state(&names, "s"));
        let limited = GameBudget {
            max_pairs: 1_000,
            max_depth: 3,
        };
        let v = game_bisimilar(&sys, r, &sys, s, limited);
        if let Verdict::Violated { witness } = &v {
            assert_eq!(witness.to_string(), "⟨m⟩⟨a⟩⟨b⟩tt");
        }
        assert_violated(&sys, r, s, v);
    }

    #[test]
    fn right_side_wins_are_negated() {
        // Left loops on `a`, right also offers `b`: Spoiler wins from the
        // right, and the negation keeps the true-in-left convention.
        let (left, _) = NamedBuilder::from_triples([("x", 'a', "x")]);
        let (right, _) = NamedBuilder::from_triples([("y", 'a', "y"), ("y", 'b', "y")]);
        let v = game_bisimilar(&left, StateId(0), &right, StateId(0), budget());
        let Verdict::Violated { witness } = v else {
            panic!("expected a violation");
        };
        assert_eq!(witness.to_string(), "¬⟨b⟩tt");
        let union = left.disjoint_union(&right);
        let t = StateId(1);
        assert!(witness.holds(&union, StateId(0)));
        assert!(!witness.holds(&union, t));
    }
}
