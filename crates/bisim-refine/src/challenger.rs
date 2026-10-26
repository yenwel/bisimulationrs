//! A learned challenger for the bisimulation game (experimental, ROADMAP
//! §4e; feature `learned-challenger`).
//!
//! A small network (the dependency-free [`Mlp`] of `bisim-metric`) learns a
//! value for positions `(x, y)` of the game of [`game_bisimilar`]:
//!
//! ```text
//! V(x, y) = 1                               if Spoiler has a move with no answer
//!         = 0                               if neither state moves
//!         = γ · max_move min_answer V(answer) otherwise
//! ```
//!
//! the discounted value of the game for Spoiler (`γᵏ⁻¹` when Spoiler needs
//! `k` rounds, `0` for bisimilar states). It is trained by fitted value
//! iteration with a slowly following target network on positions met in
//! play: first random play, then self-play in which Spoiler follows the
//! value (with exploration) and Duplicator answers with the answer of least
//! value. The input is a pair of local state features (enabled labels, and
//! which labels are enabled always or sometimes one step later), so the
//! value generalises to positions never expanded.
//!
//! [`Challenger::challenge`] then searches for a winning Spoiler strategy
//! depth first by iterative deepening, trying only the `beam` Spoiler moves
//! of highest value at each position and the answers of least value first
//! (they are the likeliest to hold, so a failing move is abandoned early).
//!
//! **Sound for "differs" only.** A strategy it finds is turned into an HML
//! formula and re-checked exactly (`Formula::holds` on both states) before
//! it is reported as `Violated`. The beam makes the search incomplete, so a
//! failed search answers `Unknown`; "same" is never claimed. Use
//! [`game_bisimilar`] or [`bisimilar`](crate::bisimilar) for that.
//!
//! [`game_bisimilar`]: crate::game_bisimilar

use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use bisim_core::{Formula, LabelId, Lts, StateId, Verdict};
use bisim_metric::{Adam, Mlp};

use crate::game::{Game, GameBudget, SpoilerMove, spoiler_moves};

/// Training settings for [`train_challenger`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChallengerOptions {
    /// Discount `γ` of the game value.
    pub gamma: f64,
    /// Width of the two hidden layers.
    pub hidden: usize,
    /// Positions expanded to collect training data, over all rounds.
    pub train_positions: usize,
    /// Rounds of collection and training (the first plays at random, later
    /// ones play against the value learned so far).
    pub rounds: usize,
    /// Gradient steps per round.
    pub steps: usize,
    pub batch: usize,
    /// Moves per play before restarting from a root pair.
    pub episode_len: usize,
    /// Probability that Spoiler moves at random during self-play.
    pub explore: f64,
    pub learning_rate: f64,
    /// Rate at which the target network follows (`τ`).
    pub target_rate: f64,
    pub seed: u64,
}

impl Default for ChallengerOptions {
    fn default() -> Self {
        Self {
            gamma: 0.8,
            hidden: 32,
            train_positions: 2_000,
            rounds: 2,
            steps: 1_500,
            batch: 32,
            episode_len: 40,
            explore: 0.3,
            learning_rate: 3e-3,
            target_rate: 0.02,
            seed: 0x5EED,
        }
    }
}

/// Search limits for [`Challenger::challenge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChallengeBudget {
    /// Distinct positions expanded by the search (training not included).
    pub max_pairs: usize,
    /// Rounds of the deepest strategy tried (the witness's modal depth).
    pub max_depth: u32,
    /// Spoiler moves tried per position, best first (`usize::MAX`: all).
    pub beam: usize,
}

impl Default for ChallengeBudget {
    fn default() -> Self {
        Self {
            max_pairs: 100_000,
            max_depth: 32,
            beam: 2,
        }
    }
}

/// Work done by training or by one challenge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChallengeStats {
    /// Distinct positions whose moves were enumerated.
    pub positions: usize,
    /// Witnesses that failed the exact re-check (expected `0`; a failure is
    /// reported as `Unknown`, never as a violation).
    pub rejected: usize,
}

type Moves = Rc<Vec<SpoilerMove>>;

/// A trained value network for the game between two systems.
#[derive(Debug)]
pub struct Challenger<L> {
    /// `left + right`, states of `right` shifted by `offset`.
    union: Lts<L>,
    offset: u32,
    net: Mlp,
    features: Vec<Option<Rc<Vec<f64>>>>,
    /// Work spent on training.
    pub training: ChallengeStats,
    /// The final training loss (mean over the last 100 steps).
    pub final_loss: f64,
}

/// Deterministic randomness (SplitMix64).
#[derive(Debug)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Trains a challenger for positions between states of `left` and `right`,
/// playing from `roots` (pairs of a left and a right state; the two initial
/// states when empty).
///
/// # Panics
/// When `roots` is empty and a system has no initial state.
pub fn train_challenger<L: Eq + Hash + Clone>(
    left: &Lts<L>,
    right: &Lts<L>,
    roots: &[(StateId, StateId)],
    options: ChallengerOptions,
) -> Challenger<L> {
    let union = left.disjoint_union(right);
    let offset = left.num_states() as u32;
    let roots: Vec<(StateId, StateId)> = if roots.is_empty() {
        vec![(
            left.initial().expect("left has an initial state"),
            right.initial().expect("right has an initial state"),
        )]
    } else {
        roots.to_vec()
    };
    let roots: Vec<_> = roots
        .into_iter()
        .map(|(s, t)| (s, StateId(t.0 + offset)))
        .collect();
    let dim = 2 * feature_len(&union);
    let mut ch = Challenger {
        features: vec![None; union.num_states()],
        net: Mlp::new(&[dim, options.hidden, options.hidden, 1], options.seed),
        union,
        offset,
        training: ChallengeStats::default(),
        final_loss: 0.0,
    };
    let mut rng = Rng(options.seed);
    let mut adam = Adam::new(&ch.net, options.learning_rate);
    let mut seen: HashMap<(StateId, StateId), Moves> = HashMap::new();
    let mut buffer: Vec<(StateId, StateId)> = Vec::new();
    let mut losses = Vec::new();
    let rounds = options.rounds.max(1);
    for round in 0..rounds {
        // Collect: play from the roots until this round's share is expanded.
        let quota = options.train_positions * (round + 1) / rounds;
        let mut stuck = 0;
        while seen.len() < quota && stuck < 1_000 {
            let before = seen.len();
            let mut p = roots[rng.below(roots.len())];
            for _ in 0..options.episode_len {
                let moves = seen
                    .entry(p)
                    .or_insert_with(|| {
                        buffer.push(p);
                        Rc::new(spoiler_moves(&ch.union, p, false))
                    })
                    .clone();
                if moves.is_empty() || moves.iter().any(|m| m.2.is_empty()) {
                    break;
                }
                let m = if round == 0 || rng.uniform() < options.explore {
                    rng.below(moves.len())
                } else {
                    ch.best_moves(&moves)[0]
                };
                let answers = &moves[m].2;
                p = if round == 0 {
                    answers[rng.below(answers.len())]
                } else {
                    // Duplicator answers with the position of least value.
                    *answers
                        .iter()
                        .min_by(|a, b| ch.value(**a).total_cmp(&ch.value(**b)))
                        .expect("answers")
                };
            }
            stuck = if seen.len() == before { stuck + 1 } else { 0 };
        }
        // Fit: one Bellman backup per sampled position, against the target.
        let mut target = ch.net.clone();
        let steps = if buffer.is_empty() { 0 } else { options.steps };
        for _ in 0..steps {
            let batch: Vec<(StateId, StateId)> = (0..options.batch)
                .map(|_| buffer[rng.below(buffer.len())])
                .collect();
            let (inputs, targets): (Vec<_>, Vec<_>) = batch
                .iter()
                .map(|p| {
                    let moves = &seen[p];
                    let y = if moves.iter().any(|m| m.2.is_empty()) {
                        1.0
                    } else if moves.is_empty() {
                        0.0
                    } else {
                        let best = moves
                            .iter()
                            .map(|m| {
                                m.2.iter()
                                    .map(|&a| ch.value_with(&target, a))
                                    .fold(f64::INFINITY, f64::min)
                            })
                            .fold(0.0, f64::max);
                        options.gamma * best.clamp(0.0, 1.0)
                    };
                    (ch.input(*p), vec![y])
                })
                .unzip();
            losses.push(ch.net.fit_step(&mut adam, &inputs, &targets));
            target.follow(&ch.net, options.target_rate);
        }
    }
    let tail = &losses[losses.len().saturating_sub(100)..];
    ch.final_loss = tail.iter().sum::<f64>() / tail.len().max(1) as f64;
    ch.training.positions = seen.len();
    ch
}

/// Length of a state's feature vector for the labels of `sys`.
fn feature_len<L: Eq + Hash + Clone>(sys: &Lts<L>) -> usize {
    3 * sys.labels().len() + 1
}

impl<L: Eq + Hash + Clone> Challenger<L> {
    /// Features of a state: per label, whether it is enabled, enabled after
    /// every step and after some step; then `ln(1 + out-degree)`.
    fn features(&mut self, s: StateId) -> Rc<Vec<f64>> {
        if let Some(f) = &self.features[s.index()] {
            return f.clone();
        }
        let n = self.union.labels().len();
        let mut f = vec![0.0; 3 * n + 1];
        let enabled = |t: StateId| -> Vec<bool> {
            let mut e = vec![false; n];
            for edge in self.union.successors(t) {
                e[edge.label.0 as usize] = true;
            }
            e
        };
        let here = enabled(s);
        let succ = self.union.successors(s);
        let (mut always, mut some) = (vec![!succ.is_empty(); n], vec![false; n]);
        for edge in succ {
            for (l, on) in enabled(edge.target).into_iter().enumerate() {
                always[l] &= on;
                some[l] |= on;
            }
        }
        for l in 0..n {
            f[l] = f64::from(u8::from(here[l]));
            f[n + l] = f64::from(u8::from(always[l]));
            f[2 * n + l] = f64::from(u8::from(some[l]));
        }
        f[3 * n] = (1.0 + succ.len() as f64).ln();
        let f = Rc::new(f);
        self.features[s.index()] = Some(f.clone());
        f
    }

    /// Network input for a position: `|φ(x) − φ(y)|` and `(φ(x) + φ(y)) / 2`
    /// (symmetric, as Spoiler may move on either side).
    fn input(&mut self, (x, y): (StateId, StateId)) -> Vec<f64> {
        let (a, b) = (self.features(x), self.features(y));
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .chain(a.iter().zip(b.iter()).map(|(p, q)| (p + q) / 2.0))
            .collect()
    }

    fn value_with(&mut self, net: &Mlp, p: (StateId, StateId)) -> f64 {
        net.eval(&self.input(p))[0]
    }

    /// The learned value of a position (an estimate, roughly in `[0, 1]`).
    fn value(&mut self, p: (StateId, StateId)) -> f64 {
        let x = self.input(p);
        self.net.eval(&x)[0]
    }

    /// Spoiler's moves, best first: by the least value among the answers.
    fn best_moves(&mut self, moves: &[SpoilerMove]) -> Vec<usize> {
        let mut scored: Vec<(f64, usize)> = moves
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let v =
                    m.2.iter()
                        .map(|&a| self.value(a))
                        .fold(f64::INFINITY, f64::min);
                (v, i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().map(|(_, i)| i).collect()
    }

    /// The learned value of state `s` of the left system against state `t`
    /// of the right one.
    pub fn estimate(&mut self, s: StateId, t: StateId) -> f64 {
        self.value((s, StateId(t.0 + self.offset)))
    }

    /// Searches for a Spoiler strategy refuting `s ∼ t` (`s` in the left
    /// system, `t` in the right). Returns `Violated` with a formula true in
    /// `s` and false in `t`, re-checked exactly, or `Unknown`.
    pub fn challenge(
        &mut self,
        s: StateId,
        t: StateId,
        budget: ChallengeBudget,
    ) -> (Verdict<Formula<L, bool>>, ChallengeStats) {
        let root = (s, StateId(t.0 + self.offset));
        let mut search = Search {
            seen: HashMap::new(),
            refuted: HashMap::new(),
            failed: HashMap::new(),
            budget,
            exhausted: false,
        };
        let mut found = None;
        for rounds in 1..=budget.max_depth {
            found = search.attack(self, root, rounds);
            if found.is_some() || search.exhausted {
                break;
            }
        }
        let mut stats = ChallengeStats {
            positions: search.seen.len(),
            rejected: 0,
        };
        if let Some(f) = found {
            let f = Formula::clone(&f);
            if f.holds(&self.union, root.0) && !f.holds(&self.union, root.1) {
                return (Verdict::Violated { witness: f }, stats);
            }
            stats.rejected = 1;
        }
        (Verdict::Unknown, stats)
    }
}

/// One challenge: a depth-first, beam-limited search for Spoiler.
struct Search<L> {
    seen: HashMap<(StateId, StateId), Moves>,
    /// Positions refuted, with a witness (valid at any depth).
    refuted: HashMap<(StateId, StateId), Rc<Formula<L, bool>>>,
    /// Positions where the search failed with this many rounds left.
    failed: HashMap<(StateId, StateId), u32>,
    budget: ChallengeBudget,
    exhausted: bool,
}

impl<L: Eq + Hash + Clone> Search<L> {
    /// A witness for position `p` within `rounds` Spoiler moves, if the
    /// beam finds one.
    fn attack(
        &mut self,
        ch: &mut Challenger<L>,
        p: (StateId, StateId),
        rounds: u32,
    ) -> Option<Rc<Formula<L, bool>>> {
        if let Some(f) = self.refuted.get(&p) {
            return Some(f.clone());
        }
        if rounds == 0 || self.failed.get(&p).is_some_and(|&r| r >= rounds) {
            return None;
        }
        let moves = match self.seen.get(&p) {
            Some(m) => m.clone(),
            None => {
                if self.seen.len() >= self.budget.max_pairs {
                    self.exhausted = true;
                    return None;
                }
                let m = Rc::new(spoiler_moves(&ch.union, p, false));
                self.seen.insert(p, m.clone());
                m
            }
        };
        let order: Vec<usize> = match moves.iter().position(|m| m.2.is_empty()) {
            Some(m) => vec![m],
            None if rounds == 1 => Vec::new(),
            None => ch.best_moves(&moves),
        };
        for &m in order.iter().take(self.budget.beam.max(1)) {
            let (from_right, label, answers) = &moves[m];
            let mut answers = answers.clone();
            answers.sort_by(|a, b| ch.value(*a).total_cmp(&ch.value(*b)));
            let mut conj = Vec::with_capacity(answers.len());
            for &a in &answers {
                match self.attack(ch, a, rounds - 1) {
                    Some(f) => conj.push(f),
                    None => break,
                }
                if self.exhausted {
                    return None;
                }
            }
            if conj.len() < answers.len() {
                continue;
            }
            let f = Rc::new(modal(&ch.union, *from_right, *label, &conj));
            self.refuted.insert(p, f.clone());
            return Some(f);
        }
        self.failed.insert(p, rounds);
        None
    }
}

/// The witness of a won move, as in `game_bisimilar`: `⟨a⟩⋀ fᵢ` when
/// Spoiler moved on the left, `¬⟨a⟩⋀ ¬fᵢ` when on the right.
fn modal<L: Eq + Hash + Clone>(
    sys: &Lts<L>,
    from_right: bool,
    label: LabelId,
    answers: &[Rc<Formula<L, bool>>],
) -> Formula<L, bool> {
    let mut conj: Vec<Formula<L, bool>> = answers
        .iter()
        .map(|f| {
            let f = Formula::clone(f);
            if from_right {
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
    let f = Formula::Modal {
        label: sys.label(label).clone(),
        weight: true,
        target: Box::new(target),
    };
    if from_right {
        Formula::Not(Box::new(f))
    } else {
        f
    }
}

/// [`game_bisimilar`](crate::game_bisimilar) with the number of positions
/// it expanded, for comparing the work of the two searches.
pub fn game_positions<L: Eq + Hash + Clone>(
    left: &Lts<L>,
    s: StateId,
    right: &Lts<L>,
    t: StateId,
    budget: GameBudget,
) -> (Verdict<Formula<L, bool>>, usize) {
    let union = left.disjoint_union(right);
    let t = StateId(t.0 + left.num_states() as u32);
    let mut game = Game::new(&union, budget);
    let v = game.play(s, t);
    (v, game.expanded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bisimilar_states;
    use bisim_core::{NamedBuilder, WltsBuilder};

    /// A ring of `n` `a`-steps where only state `n − 1` also offers `b`.
    fn ring(n: u32, b_at: Option<u32>) -> Lts<char> {
        let mut b = WltsBuilder::<char, bool>::new();
        for i in 0..n {
            b.add_transition(StateId(i), 'a', StateId((i + 1) % n), true);
        }
        if let Some(k) = b_at {
            b.add_transition(StateId(k), 'b', StateId(k), true);
        }
        b.set_initial(StateId(0));
        b.build()
    }

    #[test]
    fn finds_deep_differences_and_rechecks_them() {
        // The b-move sits 9 steps in on the left, 8 on the right.
        let (left, right) = (ring(12, Some(9)), ring(12, Some(8)));
        let mut ch = train_challenger(&left, &right, &[], ChallengerOptions::default());
        let (v, stats) = ch.challenge(StateId(0), StateId(0), ChallengeBudget::default());
        let Verdict::Violated { witness } = v else {
            panic!("expected a violation, got {v:?}");
        };
        assert_eq!(stats.rejected, 0);
        let union = left.disjoint_union(&right);
        assert!(witness.holds(&union, StateId(0)));
        assert!(!witness.holds(&union, StateId(12)));
        assert!(!bisimilar_states(&left, StateId(0), &right, StateId(0)).holds());
    }

    #[test]
    fn never_claims_equivalence() {
        let (left, right) = (ring(6, Some(2)), ring(6, Some(2)));
        let pairs: Vec<_> = left
            .states()
            .flat_map(|s| right.states().map(move |t| (s, t)))
            .collect();
        let mut ch = train_challenger(&left, &right, &pairs, ChallengerOptions::default());
        let (v, _) = ch.challenge(StateId(0), StateId(0), ChallengeBudget::default());
        assert_eq!(v, Verdict::Unknown);
        // Bisimilar positions learn a value near 0, differing ones near 1.
        // An immediate mismatch learns a value near 1.
        assert!(ch.estimate(StateId(2), StateId(3)) > 0.9);
        // Local features cannot tell (0, 0) (bisimilar) from (0, 3) (three
        // rounds apart): both states offer only `a`, now and one step later.
        // The value is then an average, which is why it only orders moves.
        assert_eq!(
            ch.estimate(StateId(0), StateId(0)),
            ch.estimate(StateId(0), StateId(3))
        );
    }

    #[test]
    fn witnesses_agree_with_the_exact_game() {
        // Every pair of a small system: whenever the challenger answers, the
        // exact checks agree, and the witness holds in s and fails in t.
        let (sys, _) = NamedBuilder::from_triples([
            ("p", 'a', "p1"),
            ("p", 'a', "p2"),
            ("p1", 'b', "e"),
            ("p1", 'c', "e"),
            ("p2", 'b', "e"),
            ("q", 'a', "q1"),
            ("q1", 'b', "e"),
            ("q1", 'c', "e"),
            ("r", 'a', "r"),
            ("u", 'a', "u1"),
            ("u1", 'a', "u"),
        ]);
        let pairs: Vec<_> = sys
            .states()
            .flat_map(|s| sys.states().map(move |t| (s, t)))
            .collect();
        let mut ch = train_challenger(&sys, &sys, &pairs, ChallengerOptions::default());
        let union = sys.disjoint_union(&sys);
        let n = sys.num_states() as u32;
        for &(s, t) in &pairs {
            let exact = bisimilar_states(&sys, s, &sys, t).holds();
            let budget = ChallengeBudget {
                beam: usize::MAX,
                ..ChallengeBudget::default()
            };
            match ch.challenge(s, t, budget).0 {
                Verdict::Violated { witness } => {
                    assert!(!exact, "{s:?} {t:?}");
                    assert!(witness.holds(&union, s));
                    assert!(!witness.holds(&union, StateId(t.0 + n)));
                }
                Verdict::Unknown => {
                    // An unlimited beam is complete up to the depth bound.
                    assert!(exact, "missed {s:?} vs {t:?}");
                }
                v => panic!("unexpected {v:?}"),
            }
        }
    }

    #[test]
    fn budget_exhaustion_is_unknown() {
        let (left, right) = (ring(12, Some(9)), ring(12, Some(8)));
        let mut ch = train_challenger(&left, &right, &[], ChallengerOptions::default());
        let tight = ChallengeBudget {
            max_pairs: 3,
            ..ChallengeBudget::default()
        };
        let (v, stats) = ch.challenge(StateId(0), StateId(0), tight);
        assert_eq!(v, Verdict::Unknown);
        assert!(stats.positions <= 3);
    }

    #[test]
    fn exact_game_positions_are_counted() {
        let (left, right) = (ring(12, Some(9)), ring(12, Some(8)));
        let (v, n) = game_positions(&left, StateId(0), &right, StateId(0), GameBudget::default());
        assert!(v.is_violated());
        assert!(n >= 9, "{n}");
    }
}
