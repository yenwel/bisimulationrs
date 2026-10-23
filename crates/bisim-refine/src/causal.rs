//! History-preserving bisimulation: equivalence up to causality.
//!
//! Interleaving bisimulation cannot tell `a | b` (two independent steps)
//! from `a.b + b.a` (either order, the second caused by the first): both
//! offer the same sequences. *History-preserving* bisimulation also asks
//! that matched steps have matched causes: whenever one system does a step
//! that depends on exactly some of the earlier steps, the other must answer
//! with a step that depends on the corresponding earlier steps. On finite
//! safe nets it is decidable (Vogler, ICALP 1991) and DEXPTIME-complete
//! (Jategaonkar & Meyer, TCS 1996).
//!
//! # How it is decided here
//!
//! The game is played on safe nets ([`SafeNet`]), where a step's causes are
//! the steps that produced the tokens it consumes, and their causes. The
//! full history grows without bound, but the future only depends on it
//! through the tokens still present. Group the past (matched) steps by the
//! pair `(L, R)` of current tokens, left and right, that depend on them:
//!
//! - a new step on the left consuming `pre₁` depends on a group iff `L`
//!   meets `pre₁`, and likewise on the right, so matching steps must agree,
//!   group by group, on whether they depend on it;
//! - after the match, a group the steps depended on also covers the
//!   produced tokens, consumed tokens leave every group, the new step forms
//!   the group `(post₁, post₂)`, and empty groups are dropped;
//! - steps in the same group cannot be told apart by anything that follows,
//!   so groups are sets.
//!
//! A position is then `(marking₁, marking₂, groups)`: finitely many, and an
//! ordinary bisimulation game, solved on the fly with refutation counters
//! like [`game_bisimilar`](crate::game_bisimilar). This abstraction is this
//! library's own construction (in the spirit of Vogler's ordered markings).
//! That the grouped game has the same winner as the definition is proved in
//! Lean (`lean/Bisim/Causal.lean`, `hpBisimilar_iff`), on a model of this
//! game; the tests also compare this code with the definition on acyclic
//! nets, where the histories are finite and can be compared directly.
//!
//! A violation carries a [`CausalFormula`]: a modal formula whose steps also
//! say which earlier steps they depend on.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::hash::Hash;

use bisim_core::{Contact, Marking, SafeNet, Verdict};

/// A modal formula over steps with causes: `⟨a ← 1,3⟩φ` holds when the system
/// can do an `a` that depends on exactly the 1st and 3rd steps taken so far
/// (counting the enclosing `⟨…⟩` from the outside), and then satisfies `φ`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CausalFormula<L> {
    True,
    Not(Box<CausalFormula<L>>),
    And(Vec<CausalFormula<L>>),
    Step {
        label: L,
        /// Earlier steps (1-based, outermost first) the step depends on.
        causes: Vec<usize>,
        then: Box<CausalFormula<L>>,
    },
}

impl<L: fmt::Display> fmt::Display for CausalFormula<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::True => write!(f, "tt"),
            Self::Not(x) => write!(f, "¬{x}"),
            Self::And(xs) => {
                write!(f, "(")?;
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ∧ ")?;
                    }
                    write!(f, "{x}")?;
                }
                write!(f, ")")
            }
            Self::Step {
                label,
                causes,
                then,
            } => {
                if causes.is_empty() {
                    write!(f, "⟨{label} ← ∅⟩{then}")
                } else {
                    let c: Vec<String> = causes.iter().map(usize::to_string).collect();
                    write!(f, "⟨{label} ← {}⟩{then}", c.join(","))
                }
            }
        }
    }
}

/// Which earlier steps each current token depends on.
type History = HashMap<u32, BTreeSet<usize>>;

impl<L: Eq + Clone> CausalFormula<L> {
    /// Does the formula hold in the initial marking of `net`?
    pub fn holds(&self, net: &SafeNet<L>) -> bool {
        self.holds_at(net, &net.initial, &History::new(), 0)
    }

    fn holds_at(&self, net: &SafeNet<L>, marking: &[u32], hist: &History, depth: usize) -> bool {
        match self {
            Self::True => true,
            Self::Not(x) => !x.holds_at(net, marking, hist, depth),
            Self::And(xs) => xs.iter().all(|x| x.holds_at(net, marking, hist, depth)),
            Self::Step {
                label,
                causes,
                then,
            } => net.enabled(marking).any(|t| {
                let tr = &net.transitions[t];
                if tr.label != *label || causes_of(hist, &tr.pre) != *causes {
                    return false;
                }
                let Ok(next) = net.fire(marking, t) else {
                    return false;
                };
                let next_hist = advance(hist, &tr.pre, &tr.post, depth + 1);
                then.holds_at(net, &next, &next_hist, depth + 1)
            }),
        }
    }
}

/// The steps (1-based) that a step consuming `pre` depends on.
fn causes_of(hist: &History, pre: &[u32]) -> Vec<usize> {
    let mut all = BTreeSet::new();
    for p in pre {
        if let Some(h) = hist.get(p) {
            all.extend(h.iter().copied());
        }
    }
    all.into_iter().collect()
}

/// The history after step number `step` consumed `pre` and produced `post`.
fn advance(hist: &History, pre: &[u32], post: &[u32], step: usize) -> History {
    let mut causes: BTreeSet<usize> = causes_of(hist, pre).into_iter().collect();
    causes.insert(step);
    let mut next: History = hist
        .iter()
        .filter(|(p, _)| !pre.contains(p))
        .map(|(p, h)| (*p, h.clone()))
        .collect();
    for &p in post {
        next.insert(p, causes.clone());
    }
    next
}

/// Groups of past steps: which current tokens (left, right) depend on them.
type Groups = Vec<(Vec<u32>, Vec<u32>)>;

fn meets(a: &[u32], b: &[u32]) -> bool {
    a.iter().any(|x| b.binary_search(x).is_ok())
}

/// The groups after matching a left step `(pre₁, post₁)` with a right step
/// `(pre₂, post₂)`, or `None` when their causes do not correspond.
fn matched(
    groups: &Groups,
    (pre1, post1): (&[u32], &[u32]),
    (pre2, post2): (&[u32], &[u32]),
) -> Option<Groups> {
    let mut next = BTreeSet::new();
    for (l, r) in groups {
        let hit = meets(l, pre1);
        if hit != meets(r, pre2) {
            return None;
        }
        let update = |set: &[u32], pre: &[u32], post: &[u32]| {
            let mut out: Vec<u32> = set
                .iter()
                .copied()
                .filter(|x| pre.binary_search(x).is_err())
                .collect();
            if hit {
                out.extend_from_slice(post);
            }
            out.sort_unstable();
            out.dedup();
            out
        };
        let (l2, r2) = (update(l, pre1, post1), update(r, pre2, post2));
        if !l2.is_empty() || !r2.is_empty() {
            next.insert((l2, r2));
        }
    }
    if !post1.is_empty() || !post2.is_empty() {
        next.insert((post1.to_vec(), post2.to_vec()));
    }
    Some(next.into_iter().collect())
}

/// Search budget for [`hp_bisimilar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalBudget {
    /// Most positions explored before answering `Unknown`.
    pub max_positions: usize,
}

impl Default for CausalBudget {
    fn default() -> Self {
        Self {
            max_positions: 200_000,
        }
    }
}

#[derive(Debug)]
struct Move {
    from_right: bool,
    /// The mover's transition.
    transition: usize,
    /// `(position, answering transition)`.
    answers: Vec<(usize, usize)>,
    open: usize,
}

#[derive(Debug)]
struct Position {
    markings: (Marking, Marking),
    groups: Groups,
    moves: Vec<Move>,
    answering: Vec<(usize, usize)>,
    refuted_by: Option<usize>,
}

/// Are the two nets history-preserving bisimilar from their initial
/// markings? A violation carries a [`CausalFormula`] true of `left` and
/// false of `right`. Fails if either net turns out not to be safe.
pub fn hp_bisimilar<L: Eq + Hash + Clone>(
    left: &SafeNet<L>,
    right: &SafeNet<L>,
    budget: CausalBudget,
) -> Result<Verdict<CausalFormula<L>>, Contact> {
    let mut index: HashMap<(Marking, Marking, Groups), usize> = HashMap::new();
    let mut positions: Vec<Position> = Vec::new();
    let mut queue = VecDeque::new();
    let nets = [left, right];

    let mut discover = |key: (Marking, Marking, Groups),
                        positions: &mut Vec<Position>,
                        queue: &mut VecDeque<usize>|
     -> Option<usize> {
        if let Some(&p) = index.get(&key) {
            return Some(p);
        }
        if positions.len() >= budget.max_positions {
            return None;
        }
        let p = positions.len();
        positions.push(Position {
            markings: (key.0.clone(), key.1.clone()),
            groups: key.2.clone(),
            moves: Vec::new(),
            answering: Vec::new(),
            refuted_by: None,
        });
        index.insert(key, p);
        queue.push_back(p);
        Some(p)
    };

    discover(
        (left.initial.clone(), right.initial.clone(), Vec::new()),
        &mut positions,
        &mut queue,
    );
    let mut exhausted = false;
    while let Some(p) = queue.pop_front() {
        let (m1, m2) = positions[p].markings.clone();
        let groups = positions[p].groups.clone();
        let mut moves = Vec::new();
        for from_right in [false, true] {
            let (mover, answerer) = if from_right { (1, 0) } else { (0, 1) };
            let (mm, am) = if from_right { (&m2, &m1) } else { (&m1, &m2) };
            for t in nets[mover].enabled(mm).collect::<Vec<_>>() {
                let tm = &nets[mover].transitions[t];
                let next_mover = nets[mover].fire(mm, t)?;
                let mut answers = Vec::new();
                for u in nets[answerer].enabled(am).collect::<Vec<_>>() {
                    let ta = &nets[answerer].transitions[u];
                    if ta.label != tm.label {
                        continue;
                    }
                    let (l, r) = if from_right { (ta, tm) } else { (tm, ta) };
                    let Some(next_groups) = matched(&groups, (&l.pre, &l.post), (&r.pre, &r.post))
                    else {
                        continue;
                    };
                    let next_answer = nets[answerer].fire(am, u)?;
                    let key = if from_right {
                        (next_answer, next_mover.clone(), next_groups)
                    } else {
                        (next_mover.clone(), next_answer, next_groups)
                    };
                    match discover(key, &mut positions, &mut queue) {
                        Some(q) => answers.push((q, u)),
                        None => exhausted = true,
                    }
                }
                moves.push(Move {
                    from_right,
                    transition: t,
                    answers,
                    open: 0,
                });
            }
        }
        if exhausted {
            break;
        }
        for (m, mv) in moves.iter_mut().enumerate() {
            mv.open = mv
                .answers
                .iter()
                .filter(|(q, _)| positions[*q].refuted_by.is_none())
                .count();
            for &(q, _) in &mv.answers {
                positions[q].answering.push((p, m));
            }
        }
        let lost = moves.iter().position(|mv| mv.open == 0);
        positions[p].moves = moves;
        if let Some(m) = lost {
            refute(&mut positions, p, m);
        }
        if positions[0].refuted_by.is_some() {
            break;
        }
    }
    if positions[0].refuted_by.is_some() {
        let witness = witness(&positions, nets, 0, [&History::new(), &History::new()], 0);
        return Ok(Verdict::Violated { witness });
    }
    Ok(if exhausted {
        Verdict::Unknown
    } else {
        Verdict::Holds
    })
}

fn refute(positions: &mut [Position], p: usize, m: usize) {
    positions[p].refuted_by = Some(m);
    let mut lost = VecDeque::from([p]);
    while let Some(q) = lost.pop_front() {
        for i in 0..positions[q].answering.len() {
            let (r, m) = positions[q].answering[i];
            if positions[r].refuted_by.is_some() {
                continue;
            }
            let mv = &mut positions[r].moves[m];
            mv.open -= 1;
            if mv.open == 0 {
                positions[r].refuted_by = Some(m);
                lost.push_back(r);
            }
        }
    }
}

/// A formula true of the left net and false of the right one at `p`,
/// reached after `depth` matched steps with histories `hist`. Every answer
/// of the refuting move was refuted earlier, so the recursion ends.
fn witness<L: Eq + Hash + Clone>(
    positions: &[Position],
    nets: [&SafeNet<L>; 2],
    p: usize,
    hist: [&History; 2],
    depth: usize,
) -> CausalFormula<L> {
    let pos = &positions[p];
    let mv = &pos.moves[pos
        .refuted_by
        .expect("only refuted positions have witnesses")];
    let mover = usize::from(mv.from_right);
    let tm = &nets[mover].transitions[mv.transition];
    let causes = causes_of(hist[mover], &tm.pre);
    let mover_hist = advance(hist[mover], &tm.pre, &tm.post, depth + 1);
    let conj: Vec<CausalFormula<L>> = mv
        .answers
        .iter()
        .map(|&(q, u)| {
            let ta = &nets[1 - mover].transitions[u];
            let answer_hist = advance(hist[1 - mover], &ta.pre, &ta.post, depth + 1);
            let next = if mv.from_right {
                [&answer_hist, &mover_hist]
            } else {
                [&mover_hist, &answer_hist]
            };
            let f = witness(positions, nets, q, next, depth + 1);
            if mv.from_right {
                CausalFormula::Not(Box::new(f))
            } else {
                f
            }
        })
        .collect();
    // Different answers can need the same formula: state it once.
    let mut conj = conj;
    let mut i = 0;
    while i < conj.len() {
        if conj[..i].contains(&conj[i]) {
            conj.remove(i);
        } else {
            i += 1;
        }
    }
    let then = match conj.len() {
        0 => CausalFormula::True,
        1 => conj.into_iter().next().expect("one conjunct"),
        _ => CausalFormula::And(conj),
    };
    let step = CausalFormula::Step {
        label: tm.label.clone(),
        causes,
        then: Box::new(then),
    };
    if mv.from_right {
        CausalFormula::Not(Box::new(step))
    } else {
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bisim_core::NetTransition;

    fn net(places: usize, ts: &[(char, &[u32], &[u32])], initial: &[u32]) -> SafeNet<char> {
        SafeNet::new(
            (0..places).map(|i| format!("p{i}")).collect(),
            ts.iter()
                .map(|(l, pre, post)| NetTransition {
                    label: *l,
                    pre: pre.to_vec(),
                    post: post.to_vec(),
                })
                .collect(),
            initial.to_vec(),
        )
    }

    #[test]
    fn concurrency_is_not_interleaving() {
        // a | b: two independent steps.
        let par = net(4, &[('a', &[0], &[2]), ('b', &[1], &[3])], &[0, 1]);
        // a.b + b.a: the same orders, each second step caused by the first.
        let seq = net(
            4,
            &[
                ('a', &[0], &[1]),
                ('b', &[1], &[3]),
                ('b', &[0], &[2]),
                ('a', &[2], &[3]),
            ],
            &[0],
        );
        let v = hp_bisimilar(&par, &seq, CausalBudget::default()).unwrap();
        let Verdict::Violated { witness } = v else {
            panic!("expected a causal difference, got {v:?}");
        };
        assert!(witness.holds(&par), "{witness}");
        assert!(!witness.holds(&seq), "{witness}");
        assert_eq!(witness.to_string(), "⟨a ← ∅⟩⟨b ← ∅⟩tt");
        // The interleavings are the same.
        let (lp, _) = par.reachability_graph(100).unwrap();
        let (ls, _) = seq.reachability_graph(100).unwrap();
        assert!(crate::bisimilar(&lp, &ls).holds());
        // Each is history-preserving bisimilar to itself, and b | a to a | b.
        assert_eq!(
            hp_bisimilar(&par, &par, CausalBudget::default()).unwrap(),
            Verdict::Holds
        );
        let swapped = net(4, &[('b', &[0], &[2]), ('a', &[1], &[3])], &[0, 1]);
        assert_eq!(
            hp_bisimilar(&par, &swapped, CausalBudget::default()).unwrap(),
            Verdict::Holds
        );
    }

    #[test]
    fn cycles_terminate() {
        // Two independent loops vs one loop doing both in turn.
        let par = net(2, &[('a', &[0], &[0]), ('b', &[1], &[1])], &[0, 1]);
        let one = net(1, &[('a', &[0], &[0]), ('b', &[0], &[0])], &[0]);
        let v = hp_bisimilar(&par, &one, CausalBudget::default()).unwrap();
        assert!(v.is_violated(), "{v:?}");
        assert_eq!(
            hp_bisimilar(&par, &par, CausalBudget::default()).unwrap(),
            Verdict::Holds
        );
    }
}
