//! History-preserving bisimulation against its definition.
//!
//! On acyclic nets every history is finite, so the definition can be played
//! directly: positions carry the full causal history of every token (as the
//! set of matched step numbers it depends on), and matched steps must have
//! equal cause sets. `hp_bisimilar` replaces histories by groups of steps;
//! these tests check that the abstraction loses nothing.

use std::collections::{BTreeSet, HashMap};

use bisim_core::{NetTransition, SafeNet, Verdict};
use bisim_refine::{CausalBudget, bisimilar, hp_bisimilar};

/// xorshift64*: deterministic, dependency-free test randomness.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// An acyclic net: every transition produces only into places numbered
/// above all it consumes.
fn random_acyclic(rng: &mut Rng, places: u32, transitions: usize, labels: u8) -> SafeNet<char> {
    let mut ts = Vec::new();
    for _ in 0..transitions {
        let low = rng.below(u64::from(places - 1)) as u32;
        let mut pre = vec![low];
        if rng.below(3) == 0 && low > 0 {
            pre.push(rng.below(u64::from(low)) as u32);
        }
        let top = *pre.iter().max().unwrap();
        let mut post = vec![top + 1 + rng.below(u64::from(places - top - 1)) as u32];
        if rng.below(3) == 0 {
            post.push(top + 1 + rng.below(u64::from(places - top - 1)) as u32);
        }
        ts.push(NetTransition {
            label: (b'a' + rng.below(u64::from(labels)) as u8) as char,
            pre,
            post,
        });
    }
    let mut initial = vec![0];
    if rng.below(2) == 0 {
        initial.push(1);
    }
    SafeNet::new((0..places).map(|i| format!("p{i}")).collect(), ts, initial)
}

/// PEPA-like: `components` acyclic state machines of `size` places each,
/// moving alone or synchronising pairwise; plenty of concurrency.
fn random_concurrent(
    rng: &mut Rng,
    components: u32,
    size: u32,
    transitions: usize,
) -> SafeNet<char> {
    let place = |c: u32, i: u32| c * size + i;
    let mut ts = Vec::new();
    for _ in 0..transitions {
        let label = (b'a' + rng.below(2) as u8) as char;
        let c = rng.below(u64::from(components)) as u32;
        let i = rng.below(u64::from(size - 1)) as u32;
        let j = i + 1 + rng.below(u64::from(size - 1 - i)) as u32;
        if rng.below(3) == 0 && components > 1 {
            // Synchronise with another component.
            let d = (c + 1 + rng.below(u64::from(components - 1)) as u32) % components;
            let k = rng.below(u64::from(size - 1)) as u32;
            let l = k + 1 + rng.below(u64::from(size - 1 - k)) as u32;
            ts.push(NetTransition {
                label,
                pre: vec![place(c, i), place(d, k)],
                post: vec![place(c, j), place(d, l)],
            });
        } else {
            ts.push(NetTransition {
                label,
                pre: vec![place(c, i)],
                post: vec![place(c, j)],
            });
        }
    }
    let initial = (0..components).map(|c| place(c, 0)).collect();
    SafeNet::new(
        (0..components * size).map(|i| format!("p{i}")).collect(),
        ts,
        initial,
    )
}

/// The same net with places renamed.
fn renamed(net: &SafeNet<char>, perm: &[u32]) -> SafeNet<char> {
    SafeNet::new(
        net.places.clone(),
        net.transitions
            .iter()
            .map(|t| NetTransition {
                label: t.label,
                pre: t.pre.iter().map(|&p| perm[p as usize]).collect(),
                post: t.post.iter().map(|&p| perm[p as usize]).collect(),
            })
            .collect(),
        net.initial.iter().map(|&p| perm[p as usize]).collect(),
    )
}

/// The interleaving of `net` as a sequential net: one place per reachable
/// marking. Same interleavings, no concurrency.
fn sequentialised(net: &SafeNet<char>) -> Option<SafeNet<char>> {
    let (lts, markings) = net.reachability_graph(10_000).ok()?;
    let ts = lts
        .transitions()
        .map(|(s, l, t, _)| NetTransition {
            label: *l,
            pre: vec![s.0],
            post: vec![t.0],
        })
        .collect();
    Some(SafeNet::new(
        (0..markings.len()).map(|i| format!("m{i}")).collect(),
        ts,
        vec![0],
    ))
}

type History = HashMap<u32, BTreeSet<usize>>;

fn causes(h: &History, pre: &[u32]) -> BTreeSet<usize> {
    pre.iter()
        .filter_map(|p| h.get(p))
        .flatten()
        .copied()
        .collect()
}

fn advance(h: &History, pre: &[u32], post: &[u32], step: usize) -> History {
    let mut c = causes(h, pre);
    c.insert(step);
    let mut next: History = h
        .iter()
        .filter(|(p, _)| !pre.contains(p))
        .map(|(p, s)| (*p, s.clone()))
        .collect();
    for &p in post {
        next.insert(p, c.clone());
    }
    next
}

/// The definition, played out: Duplicator wins iff every move of either
/// side has an answer with the same label and corresponding causes from
/// which Duplicator wins again. `None` if a net is not safe.
fn oracle(
    nets: [&SafeNet<char>; 2],
    m: [&[u32]; 2],
    h: [&History; 2],
    step: usize,
) -> Option<bool> {
    for side in 0..2 {
        let other = 1 - side;
        for t in nets[side].enabled(m[side]).collect::<Vec<_>>() {
            let tm = &nets[side].transitions[t];
            let next_m = nets[side].fire(m[side], t).ok()?;
            let next_h = advance(h[side], &tm.pre, &tm.post, step + 1);
            let mut answered = false;
            for u in nets[other].enabled(m[other]).collect::<Vec<_>>() {
                let ta = &nets[other].transitions[u];
                if ta.label != tm.label || causes(h[other], &ta.pre) != causes(h[side], &tm.pre) {
                    continue;
                }
                let next_a = nets[other].fire(m[other], u).ok()?;
                let next_ha = advance(h[other], &ta.pre, &ta.post, step + 1);
                let (mm, hh) = if side == 0 {
                    ([&next_m[..], &next_a[..]], [&next_h, &next_ha])
                } else {
                    ([&next_a[..], &next_m[..]], [&next_ha, &next_h])
                };
                if oracle(nets, mm, hh, step + 1)? {
                    answered = true;
                    break;
                }
            }
            if !answered {
                return Some(false);
            }
        }
    }
    Some(true)
}

#[test]
fn groups_decide_the_same_as_full_histories() {
    let mut rng = Rng(0xCA05_A11D);
    let (mut same, mut differ, mut skipped, mut causal_only, mut causal_differences) =
        (0, 0, 0, 0, 0);
    for round in 0..1_500 {
        let places = 4 + (round % 4) as u32;
        let a = if round % 3 == 2 {
            random_concurrent(&mut rng, 2 + (round % 2) as u32, 3, 3 + round % 3)
        } else {
            random_acyclic(&mut rng, places, 3 + round % 4, 2)
        };
        let b = match round % 3 {
            // Same interleavings, causality forgotten.
            2 => match sequentialised(&a) {
                Some(b) => b,
                None => {
                    skipped += 1;
                    continue;
                }
            },
            0 => {
                let mut perm: Vec<u32> = (0..places).collect();
                for i in (1..perm.len()).rev() {
                    perm.swap(i, rng.below(i as u64 + 1) as usize);
                }
                renamed(&a, &perm)
            }
            _ => random_acyclic(&mut rng, places, 3 + round % 4, 2),
        };
        let empty = History::new();
        if round % 3 == 2 {
            causal_only += 1;
        }
        let Some(want) = oracle([&a, &b], [&a.initial, &b.initial], [&empty, &empty], 0) else {
            skipped += 1;
            continue;
        };
        let got = match hp_bisimilar(&a, &b, CausalBudget::default()) {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        assert_eq!(
            got.holds(),
            want,
            "round {round}: {a:?}\nvs {b:?}\ngot {got:?}"
        );
        if let Verdict::Violated { witness } = &got {
            assert!(
                witness.holds(&a),
                "round {round}: {witness} fails on the left"
            );
            assert!(
                !witness.holds(&b),
                "round {round}: {witness} holds on the right"
            );
        }
        if want {
            same += 1
        } else {
            differ += 1
        }
        if round % 3 == 2 && !want {
            causal_differences += 1;
        }
    }
    println!("{causal_differences} pairs differ only in causality");
    assert!(causal_differences > 100, "{causal_differences}");
    println!("{same} same, {differ} differ, {skipped} skipped, {causal_only} sequentialised pairs");
    assert!(
        same > 200 && differ > 200,
        "{same} same, {differ} differ, {skipped} skipped"
    );
}

#[test]
fn causality_refines_interleaving_and_agrees_on_sequential_nets() {
    let mut rng = Rng(0x5E0_0E17);
    let (mut finer, mut checked) = (0, 0);
    for round in 0..600 {
        let places = 4 + (round % 4) as u32;
        let (a, b) = if round % 2 == 0 {
            let a = random_concurrent(&mut rng, 2, 3, 4);
            let Some(b) = sequentialised(&a) else {
                continue;
            };
            (a, b)
        } else {
            (
                random_acyclic(&mut rng, places, 4, 2),
                random_acyclic(&mut rng, places, 4, 2),
            )
        };
        let (Ok(v), Ok((la, _)), Ok((lb, _))) = (
            hp_bisimilar(&a, &b, CausalBudget::default()),
            a.reachability_graph(10_000),
            b.reachability_graph(10_000),
        ) else {
            continue;
        };
        let interleaving = bisimilar(&la, &lb).holds();
        if v.holds() {
            assert!(
                interleaving,
                "round {round}: causally equal but not interleaving-equal"
            );
        } else if interleaving {
            finer += 1;
        }
        // Sequential nets (one token, one place in and out per step) have
        // no concurrency: the two equivalences coincide.
        let sequential = |n: &SafeNet<char>| {
            n.initial.len() == 1
                && n.transitions
                    .iter()
                    .all(|t| t.pre.len() == 1 && t.post.len() == 1)
        };
        if sequential(&a) && sequential(&b) {
            assert_eq!(
                v.holds(),
                interleaving,
                "round {round}: sequential nets disagree"
            );
        }
        checked += 1;
    }
    assert!(checked > 300, "{checked}");
    println!("{finer} pairs equal by interleaving but causally different");
    assert!(finer > 50, "{finer}");
}
