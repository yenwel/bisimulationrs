//! Does a "same behaviour" verdict survive durations that are not
//! exponential? Experimental; see REVIEW.md, Idea 3.
//!
//! With exponential durations, `a ∥ b` and `a.b + b.a` are Markovian
//! bisimilar: when `a` finishes, what is left of `b` is exponential again.
//! With any other duration they differ, because in `a ∥ b` the activity `b`
//! has been running all along, while in `a.b + b.a` it starts afresh. So the
//! verdict depends on the exponential assumption.
//!
//! Here a model is a safe net whose transitions have rates, and durations
//! follow the *race policy with enabling memory*: every enabled transition
//! runs its own clock; the first to finish fires; a transition keeps its
//! clock while it stays enabled, and loses it when a firing takes a token it
//! needs.
//!
//! - [`erlang_expansion`] replaces every exponential clock of rate `r` by an
//!   Erlang clock with `stages` stages of rate `stages·r` (the same mean).
//!   Stage changes are internal moves, labelled `None`.
//! - [`rated_ep_bisimilar`] decides enabling-preserving bisimilarity (van
//!   Glabbeek, Höfner and Wang, CONCUR 2021) with rates: related states
//!   pair their enabled transitions one to one, with equal labels and rates,
//!   and the pairing of transitions that stay enabled is kept.
//!
//! Rated ep-bisimilarity implies that every Erlang expansion (indeed every
//! phase-type replacement) of the two nets is Markovian bisimilar: match
//! clocks through the pairing. The converse is the conjecture being tested.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;

use bisim_core::{Contact, Marking, ReachError, Real, SafeNet, StateId, Wlts, WltsBuilder};

/// Transitions enabled at `marking`, sorted.
fn enabled<L>(net: &SafeNet<L>, marking: &[u32]) -> Vec<usize> {
    net.enabled(marking).collect()
}

/// Does firing `t` leave `u` enabled all along (no shared input place)?
fn untouched<L>(net: &SafeNet<L>, t: usize, u: usize) -> bool {
    let (a, b) = (&net.transitions[t].pre, &net.transitions[u].pre);
    u != t && a.iter().all(|p| b.binary_search(p).is_err())
}

/// The net with every exponential clock of rate `r` replaced by an Erlang
/// clock of `stages` stages, each of rate `stages·r`, under the race policy
/// with enabling memory. States are markings with the stage of every enabled
/// transition; stage changes are labelled `None`, firings `Some(label)`. With
/// `stages = 1` this is the net's Markovian interleaving. At most
/// `max_states` states.
///
/// # Panics
/// If `rates` does not give one positive rate per transition, or
/// `stages = 0`.
pub fn erlang_expansion<L: Eq + Hash + Clone>(
    net: &SafeNet<L>,
    rates: &[f64],
    stages: u32,
    max_states: usize,
) -> Result<Wlts<Option<L>, Real>, ReachError> {
    assert_eq!(
        rates.len(),
        net.transitions.len(),
        "one rate per transition"
    );
    assert!(rates.iter().all(|&r| r > 0.0), "rates must be positive");
    assert!(stages > 0, "at least one stage");
    // A state: the marking and `(transition, stage)` for every enabled one.
    type State = (Marking, Vec<(usize, u32)>);
    let start: State = (
        net.initial.clone(),
        enabled(net, &net.initial)
            .into_iter()
            .map(|t| (t, 0))
            .collect(),
    );
    let mut index: HashMap<State, u32> = HashMap::from([(start.clone(), 0)]);
    let mut states = vec![start];
    let mut b = WltsBuilder::new();
    b.ensure_states(1);
    b.set_initial(StateId(0));
    let mut queue = VecDeque::from([0u32]);
    let k = f64::from(stages);
    while let Some(i) = queue.pop_front() {
        let (marking, clocks) = states[i as usize].clone();
        for (pos, &(t, stage)) in clocks.iter().enumerate() {
            let rate = k * rates[t];
            let (label, next) = if stage + 1 < stages {
                let mut c = clocks.clone();
                c[pos].1 += 1;
                (None, (marking.clone(), c))
            } else {
                let m = net.fire(&marking, t).map_err(ReachError::Contact)?;
                let kept: BTreeMap<usize, u32> = clocks
                    .iter()
                    .filter(|&&(u, _)| untouched(net, t, u))
                    .copied()
                    .collect();
                let c = enabled(net, &m)
                    .into_iter()
                    .map(|u| (u, kept.get(&u).copied().unwrap_or(0)))
                    .collect();
                (Some(net.transitions[t].label.clone()), (m, c))
            };
            let j = match index.get(&next) {
                Some(&j) => j,
                None => {
                    if states.len() >= max_states {
                        return Err(ReachError::TooLarge(max_states));
                    }
                    let j = states.len() as u32;
                    index.insert(next.clone(), j);
                    states.push(next);
                    b.ensure_states(j as usize + 1);
                    queue.push_back(j);
                    j
                }
            };
            b.add_transition(StateId(i), label, StateId(j), Real(rate));
        }
    }
    Ok(b.build())
}

/// A pairing of enabled transitions, left to right, sorted.
type Pairing = Vec<(usize, usize)>;

/// Every one-to-one pairing of `left` with `right` that keeps labels and
/// rates. Empty when the two cannot be paired.
fn pairings<L: Eq>(
    nets: [&SafeNet<L>; 2],
    rates: [&[f64]; 2],
    left: &[usize],
    right: &[usize],
) -> Vec<Pairing> {
    if left.len() != right.len() {
        return Vec::new();
    }
    let same = |t: usize, u: usize| {
        nets[0].transitions[t].label == nets[1].transitions[u].label
            && (rates[0][t] - rates[1][u]).abs() <= 1e-12 * rates[0][t].max(rates[1][u])
    };
    let mut out = Vec::new();
    let mut used = vec![false; right.len()];
    let mut current = Vec::with_capacity(left.len());
    fn go(
        i: usize,
        left: &[usize],
        right: &[usize],
        same: &dyn Fn(usize, usize) -> bool,
        used: &mut [bool],
        current: &mut Vec<(usize, usize)>,
        out: &mut Vec<Pairing>,
    ) {
        if i == left.len() {
            let mut p = current.clone();
            p.sort_unstable();
            out.push(p);
            return;
        }
        for j in 0..right.len() {
            if !used[j] && same(left[i], right[j]) {
                used[j] = true;
                current.push((left[i], right[j]));
                go(i + 1, left, right, same, used, current, out);
                current.pop();
                used[j] = false;
            }
        }
    }
    go(0, left, right, &same, &mut used, &mut current, &mut out);
    out
}

/// Are the two rated nets enabling-preserving bisimilar, with every pairing
/// of enabled transitions one to one and keeping labels and rates (see the
/// module documentation)? `None` when more than `max_positions` positions
/// would be needed.
///
/// A position is two markings and a pairing of their enabled transitions.
/// Firing a transition fires its partner; the transitions that stay
/// enabled must be partners of each other's and stay paired, and the newly
/// enabled ones are paired afresh (Duplicator chooses how).
pub fn rated_ep_bisimilar<L: Eq + Hash + Clone>(
    left: &SafeNet<L>,
    left_rates: &[f64],
    right: &SafeNet<L>,
    right_rates: &[f64],
    max_positions: usize,
) -> Result<Option<bool>, Contact> {
    assert_eq!(left_rates.len(), left.transitions.len());
    assert_eq!(right_rates.len(), right.transitions.len());
    let nets = [left, right];
    let rates = [left_rates, right_rates];
    type Key = (Marking, Marking, Pairing);
    let mut index: HashMap<Key, usize> = HashMap::new();
    let mut keys: Vec<Key> = Vec::new();
    // Per position, per Spoiler move: Duplicator's options.
    let mut moves: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut queue = VecDeque::new();
    let mut add = |key: Key,
                   keys: &mut Vec<Key>,
                   moves: &mut Vec<Vec<Vec<usize>>>,
                   queue: &mut VecDeque<usize>|
     -> Option<usize> {
        if let Some(&p) = index.get(&key) {
            return Some(p);
        }
        if keys.len() >= max_positions {
            return None;
        }
        let p = keys.len();
        index.insert(key.clone(), p);
        keys.push(key);
        moves.push(Vec::new());
        queue.push_back(p);
        Some(p)
    };
    let (e0, f0) = (enabled(left, &left.initial), enabled(right, &right.initial));
    let mut roots = Vec::new();
    for pairing in pairings(nets, rates, &e0, &f0) {
        let key = (left.initial.clone(), right.initial.clone(), pairing);
        match add(key, &mut keys, &mut moves, &mut queue) {
            Some(p) => roots.push(p),
            None => return Ok(None),
        }
    }
    while let Some(p) = queue.pop_front() {
        let (m1, m2, pairing) = keys[p].clone();
        let mut options_per_move = Vec::with_capacity(pairing.len());
        for &(t, u) in &pairing {
            let n1 = left.fire(&m1, t)?;
            let n2 = right.fire(&m2, u)?;
            let kept: Pairing = pairing
                .iter()
                .filter(|&&(a, _)| untouched(left, t, a))
                .copied()
                .collect();
            // The partners of the left transitions that stay enabled must
            // be exactly the right transitions that stay enabled.
            let kept_right = pairing
                .iter()
                .filter(|&&(_, b)| untouched(right, u, b))
                .count();
            let consistent =
                kept.len() == kept_right && kept.iter().all(|&(_, b)| untouched(right, u, b));
            let mut options = Vec::new();
            if consistent {
                let fresh = |net: &SafeNet<L>, m: &[u32], side: usize| -> Vec<usize> {
                    enabled(net, m)
                        .into_iter()
                        .filter(|x| {
                            !kept
                                .iter()
                                .any(|&(a, b)| if side == 0 { a == *x } else { b == *x })
                        })
                        .collect()
                };
                let (new1, new2) = (fresh(left, &n1, 0), fresh(right, &n2, 1));
                for extra in pairings(nets, rates, &new1, &new2) {
                    let mut next = kept.clone();
                    next.extend(extra);
                    next.sort_unstable();
                    match add(
                        (n1.clone(), n2.clone(), next),
                        &mut keys,
                        &mut moves,
                        &mut queue,
                    ) {
                        Some(q) => options.push(q),
                        None => return Ok(None),
                    }
                }
            }
            options_per_move.push(options);
        }
        moves[p] = options_per_move;
    }
    // Spoiler wins at a position with a move all of whose answers lose.
    let mut losing = vec![false; keys.len()];
    loop {
        let mut changed = false;
        for p in 0..keys.len() {
            if !losing[p] && moves[p].iter().any(|opts| opts.iter().all(|&q| losing[q])) {
                losing[p] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(Some(roots.iter().any(|&p| !losing[p])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bisim_core::NetTransition;

    fn net(transitions: &[(char, &[u32], &[u32])], initial: &[u32]) -> SafeNet<char> {
        let places = 1 + transitions
            .iter()
            .flat_map(|(_, pre, post)| pre.iter().chain(post.iter()))
            .chain(initial)
            .copied()
            .max()
            .unwrap_or(0);
        SafeNet::new(
            (0..places).map(|i| format!("p{i}")).collect(),
            transitions
                .iter()
                .map(|&(label, pre, post)| NetTransition {
                    label,
                    pre: pre.to_vec(),
                    post: post.to_vec(),
                })
                .collect(),
            initial.to_vec(),
        )
    }

    fn markovian(a: &Wlts<Option<char>, Real>, b: &Wlts<Option<char>, Real>) -> bool {
        crate::bisimilar(a, b).holds()
    }

    /// `a ∥ b` against `a.b + b.a`: the same with exponential clocks, not
    /// with Erlang ones, and not ep-bisimilar.
    #[test]
    fn concurrency_shows_once_clocks_have_memory() {
        let par = net(&[('a', &[0], &[1]), ('b', &[2], &[3])], &[0, 2]);
        let seq = net(
            &[
                ('a', &[0], &[1]),
                ('b', &[0], &[2]),
                ('b', &[1], &[3]),
                ('a', &[2], &[3]),
            ],
            &[0],
        );
        let (rp, rs) = ([1.0, 2.0], [1.0, 2.0, 2.0, 1.0]);
        let x = |n: &SafeNet<char>, r: &[f64], k| erlang_expansion(n, r, k, 10_000).unwrap();
        assert!(markovian(&x(&par, &rp, 1), &x(&seq, &rs, 1)));
        assert!(!markovian(&x(&par, &rp, 2), &x(&seq, &rs, 2)));
        assert_eq!(
            rated_ep_bisimilar(&par, &rp, &seq, &rs, 10_000).unwrap(),
            Some(false)
        );
        assert_eq!(
            rated_ep_bisimilar(&par, &rp, &par, &rp, 10_000).unwrap(),
            Some(true)
        );
    }

    /// Two clocks of rate 1 race differently from one of rate 2.
    #[test]
    fn two_clocks_are_not_one_faster_clock() {
        let two = net(&[('a', &[0], &[1]), ('a', &[0], &[1])], &[0]);
        let one = net(&[('a', &[0], &[1])], &[0]);
        let x = |n: &SafeNet<char>, r: &[f64], k| erlang_expansion(n, r, k, 10_000).unwrap();
        assert!(markovian(&x(&two, &[1.0, 1.0], 1), &x(&one, &[2.0], 1)));
        assert!(!markovian(&x(&two, &[1.0, 1.0], 3), &x(&one, &[2.0], 3)));
        assert_eq!(
            rated_ep_bisimilar(&two, &[1.0, 1.0], &one, &[2.0], 100).unwrap(),
            Some(false)
        );
    }

    /// One stage is the Markovian interleaving: the expansion has one state
    /// per reachable marking.
    #[test]
    fn one_stage_is_the_interleaving() {
        let par = net(&[('a', &[0], &[1]), ('b', &[2], &[3])], &[0, 2]);
        let x = erlang_expansion(&par, &[1.0, 2.0], 1, 100).unwrap();
        assert_eq!(x.num_states(), 4);
        let x3 = erlang_expansion(&par, &[1.0, 2.0], 3, 100).unwrap();
        // Every pair of stages (3 × 3) before anything fires, then 3 + 3
        // with one done, then the end.
        assert_eq!(x3.num_states(), 9 + 3 + 3 + 1);
    }
}
