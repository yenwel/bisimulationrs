//! Does a "same behaviour" verdict survive durations that are not
//! exponential? A random search for REVIEW.md, Idea 3.
//!
//! Many tiny rated nets (one token per component, components moving alone
//! or in pairs) are grouped into classes that are Markovian bisimilar with
//! exponential clocks. Every pair inside a class is then compared:
//!
//! - E2, E3: Markovian bisimilar after replacing every clock by an Erlang
//!   clock of 2 or 3 stages (race policy, enabling memory);
//! - EP: enabling-preserving bisimilar with rates;
//! - HP: history-preserving bisimilar (untimed, rates ignored);
//! - for pairs that E2 separates: is there a difference in what is
//!   *observed* (the probability of each short sequence of events by a
//!   given time, stages hidden)?
//!
//! Conjecture: EP ⇔ "E_k for every k". EP ⇒ E_k holds by construction.
//!
//! Run: `cargo run --release -p bisimulationrs --example durations`

use std::collections::HashMap;

use bisimulationrs::core::{NetTransition, Real, SafeNet, StateId, Wlts, WltsBuilder};
use bisimulationrs::refine::durations::{erlang_expansion, rated_ep_bisimilar};
use bisimulationrs::refine::{CausalBudget, bisimilar, hp_bisimilar};
use bisimulationrs::stochastic::Ctmc;

/// xorshift64*: deterministic randomness.
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

#[derive(Debug, Clone)]
struct Rated {
    net: SafeNet<char>,
    rates: Vec<f64>,
}

impl Rated {
    /// Transitions as text, for printing and for removing duplicates.
    fn describe(&self) -> String {
        let mut ts: Vec<String> = self
            .net
            .transitions
            .iter()
            .zip(&self.rates)
            .map(|(t, r)| format!("{}{:?}->{:?}@{r}", t.label, t.pre, t.post))
            .collect();
        ts.sort();
        format!("init {:?}: {}", self.net.initial, ts.join(" "))
    }
}

/// `components` state machines of `size` places with one token each;
/// a transition moves one component, or two at once. Labels are drawn from
/// the first `labels` letters, rates from `1..=rates`; at most `most`
/// transitions and `parts` components.
fn random_net(
    rng: &mut Rng,
    cyclic: bool,
    labels: u64,
    rates_from: u64,
    most: u64,
    parts: u64,
) -> Rated {
    let components = 1 + rng.below(parts) as u32;
    let size = 2 + rng.below(2) as u32;
    let count = 1 + rng.below(most) as usize;
    let place = |c: u32, i: u32| c * size + i;
    let step = |rng: &mut Rng| {
        let i = rng.below(u64::from(size - 1)) as u32;
        let j = if cyclic {
            (i + 1 + rng.below(u64::from(size - 1)) as u32) % size
        } else {
            i + 1 + rng.below(u64::from(size - 1 - i)) as u32
        };
        (i, j)
    };
    let mut transitions = Vec::new();
    let mut rates = Vec::new();
    for _ in 0..count {
        let label = (b'a' + rng.below(labels) as u8) as char;
        let c = rng.below(u64::from(components)) as u32;
        let (i, j) = step(rng);
        let (pre, post) = if components > 1 && rng.below(3) == 0 {
            let d = (c + 1 + rng.below(u64::from(components - 1)) as u32) % components;
            let (k, l) = step(rng);
            (
                vec![place(c, i), place(d, k)],
                vec![place(c, j), place(d, l)],
            )
        } else {
            (vec![place(c, i)], vec![place(c, j)])
        };
        transitions.push(NetTransition { label, pre, post });
        rates.push(1.0 + rng.below(rates_from) as f64);
    }
    let initial = (0..components).map(|c| place(c, 0)).collect();
    Rated {
        net: SafeNet::new(
            (0..components * size).map(|i| format!("p{i}")).collect(),
            transitions,
            initial,
        ),
        rates,
    }
}

/// One place per reachable marking, one transition per enabled net
/// transition: the same interleaving and rates, no concurrency.
fn sequentialised(r: &Rated) -> Option<Rated> {
    let mut index: HashMap<Vec<u32>, u32> = HashMap::from([(r.net.initial.clone(), 0)]);
    let mut markings = vec![r.net.initial.clone()];
    let (mut transitions, mut rates) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < markings.len() {
        let m = markings[i].clone();
        for t in r.net.enabled(&m).collect::<Vec<_>>() {
            let next = r.net.fire(&m, t).ok()?;
            let j = *index.entry(next.clone()).or_insert_with(|| {
                markings.push(next);
                markings.len() as u32 - 1
            });
            transitions.push(NetTransition {
                label: r.net.transitions[t].label,
                pre: vec![i as u32],
                post: vec![j],
            });
            rates.push(r.rates[t]);
        }
        i += 1;
    }
    Some(Rated {
        net: SafeNet::new(
            (0..markings.len()).map(|k| format!("m{k}")).collect(),
            transitions,
            vec![0],
        ),
        rates,
    })
}

/// Parallel transitions (same label, input and output) merged into one,
/// with the sum of their rates: one clock instead of several.
fn merged(r: &Rated) -> Rated {
    let mut by_key: Vec<(NetTransition<char>, f64)> = Vec::new();
    for (t, &rate) in r.net.transitions.iter().zip(&r.rates) {
        match by_key.iter_mut().find(|(u, _)| u == t) {
            Some((_, total)) => *total += rate,
            None => by_key.push((t.clone(), rate)),
        }
    }
    let (transitions, rates) = by_key.into_iter().unzip();
    Rated {
        net: SafeNet::new(r.net.places.clone(), transitions, r.net.initial.clone()),
        rates,
    }
}

fn expansion(r: &Rated, stages: u32) -> Option<Wlts<Option<char>, Real>> {
    erlang_expansion(&r.net, &r.rates, stages, 20_000).ok()
}

fn markovian(a: &Wlts<Option<char>, Real>, b: &Wlts<Option<char>, Real>) -> bool {
    bisimilar(a, b).holds()
}

/// The probability that the first `trace.len()` observed firings are
/// `trace` and all happen by time `t` (stages are not observed).
fn trace_probability(x: &Wlts<Option<char>, Real>, trace: &[char], t: f64) -> f64 {
    let n = x.num_states();
    let len = trace.len();
    // States (s, matched) for matched < len, then `done` and `failed`.
    let id = |s: usize, j: usize| (j * n + s) as u32;
    let (done, failed) = ((len * n) as u32, (len * n + 1) as u32);
    let mut b = WltsBuilder::new();
    b.ensure_states(len * n + 2);
    b.set_initial(StateId(id(x.initial().unwrap().index(), 0)));
    for (j, &want) in trace.iter().enumerate() {
        for (s, l, target, w) in x.transitions() {
            let to = match l {
                None => id(target.index(), j),
                Some(c) if *c == want && j + 1 == len => done,
                Some(c) if *c == want => id(target.index(), j + 1),
                Some(_) => failed,
            };
            b.add_transition(StateId(id(s.index(), j)), (), StateId(to), *w);
        }
    }
    let chain = Ctmc::from_wlts(&b.build());
    chain.transient(&chain.initial_distribution(), t, 1e-12)[done as usize]
}

/// Is there an observable difference: a sequence of up to `depth` events
/// whose probability by time 0.5, 1, 2 or 5 differs?
fn observably_different(
    a: &Wlts<Option<char>, Real>,
    b: &Wlts<Option<char>, Real>,
    depth: usize,
) -> bool {
    let mut traces: Vec<Vec<char>> = vec![vec![]];
    for _ in 0..depth {
        traces = traces
            .iter()
            .flat_map(|t| {
                ['a', 'b'].into_iter().map(move |c| {
                    let mut u = t.clone();
                    u.push(c);
                    u
                })
            })
            .chain(traces.iter().cloned())
            .collect();
        traces.sort();
        traces.dedup();
    }
    traces.retain(|t| !t.is_empty());
    traces.iter().any(|trace| {
        [0.5, 1.0, 2.0, 5.0].iter().any(|&t| {
            (trace_probability(a, trace, t) - trace_probability(b, trace, t)).abs() > 1e-9
        })
    })
}

/// No reachable marking enables two transitions with the same label and
/// rate: the case `lean/Bisim/Durations.lean` proves.
fn type_unambiguous(r: &Rated) -> bool {
    let mut seen = std::collections::HashSet::from([r.net.initial.clone()]);
    let mut todo = vec![r.net.initial.clone()];
    while let Some(m) = todo.pop() {
        let enabled: Vec<usize> = r.net.enabled(&m).collect();
        for (i, &t) in enabled.iter().enumerate() {
            for &u in &enabled[i + 1..] {
                let same_label = r.net.transitions[t].label == r.net.transitions[u].label;
                if same_label && r.rates[t] == r.rates[u] {
                    return false;
                }
            }
            if let Ok(next) = r.net.fire(&m, t) {
                if seen.insert(next.clone()) {
                    todo.push(next);
                }
            }
        }
    }
    true
}

#[derive(Default)]
struct Tally {
    pairs: usize,
    /// Pairs in which both nets are type-unambiguous.
    unambiguous: usize,
    /// Pairs too large for the budgets.
    skipped: usize,
    ep: usize,
    e2: usize,
    e3: usize,
    hp: usize,
    ep_not_e: Vec<String>,
    e_not_ep: Vec<String>,
    e2_not_e3: usize,
    hp_not_ep: usize,
    /// Of those, how many are ep-bisimilar once parallel transitions (same
    /// label, input and output) are merged into one clock.
    hp_not_ep_split_clocks: usize,
    hp_not_ep_examples: Vec<String>,
    ep_not_hp: usize,
    hidden: usize,
    hidden_examples: Vec<String>,
}

fn compare(a: &Rated, b: &Rated, tally: &mut Tally) -> Option<()> {
    let (a2, b2) = (expansion(a, 2)?, expansion(b, 2)?);
    let (a3, b3) = (expansion(a, 3)?, expansion(b, 3)?);
    let ep = rated_ep_bisimilar(&a.net, &a.rates, &b.net, &b.rates, 200_000).ok()??;
    let hp = hp_bisimilar(&a.net, &b.net, CausalBudget::default())
        .ok()?
        .holds();
    let (e2, e3) = (markovian(&a2, &b2), markovian(&a3, &b3));
    let pair = format!("{}\n      {}", a.describe(), b.describe());
    tally.pairs += 1;
    tally.unambiguous += usize::from(type_unambiguous(a) && type_unambiguous(b));
    tally.ep += usize::from(ep);
    tally.e2 += usize::from(e2);
    tally.e3 += usize::from(e3);
    tally.hp += usize::from(hp);
    if ep && !(e2 && e3) {
        tally.ep_not_e.push(pair.clone());
    }
    if (e2 || e3) && !ep {
        tally.e_not_ep.push(pair.clone());
    }
    tally.e2_not_e3 += usize::from(e2 != e3);
    tally.hp_not_ep += usize::from(hp && !ep);
    if hp && !ep {
        let (ma, mb) = (merged(a), merged(b));
        let after = rated_ep_bisimilar(&ma.net, &ma.rates, &mb.net, &mb.rates, 200_000).ok()??;
        tally.hp_not_ep_split_clocks += usize::from(after);
        if !after && tally.hp_not_ep_examples.len() < 3 {
            tally.hp_not_ep_examples.push(pair.clone());
        }
    }
    tally.ep_not_hp += usize::from(ep && !hp);
    if !e2 && !observably_different(&a2, &b2, 3) && !observably_different(&a2, &b2, 6) {
        tally.hidden += 1;
        if tally.hidden_examples.len() < 3 {
            tally.hidden_examples.push(pair);
        }
    }
    Some(())
}

fn report(name: &str, t: &Tally) {
    println!(
        "{name}: {} pairs Markovian bisimilar with exponential clocks",
        t.pairs
    );
    println!(
        "   EP {:>5}   E2 {:>5}   E3 {:>5}   HP {:>5}",
        t.ep, t.e2, t.e3, t.hp
    );
    println!(
        "   type-unambiguous (proved case): {}   others (evidence only): {}   skipped: {}",
        t.unambiguous,
        t.pairs - t.unambiguous,
        t.skipped
    );
    println!(
        "   EP but not E2 or E3 (would refute EP ⇒ E): {}",
        t.ep_not_e.len()
    );
    println!(
        "   E2 or E3 but not EP (would refute the conjecture): {}",
        t.e_not_ep.len()
    );
    println!("   E2 and E3 disagree: {}", t.e2_not_e3);
    println!(
        "   HP but not EP: {} ({} of them through split clocks)   EP but not HP: {}",
        t.hp_not_ep, t.hp_not_ep_split_clocks, t.ep_not_hp
    );
    println!(
        "   not E2, yet no observable difference in up to 6 events: {}",
        t.hidden
    );
    for (title, list) in [
        ("EP but not E", &t.ep_not_e),
        ("E but not EP", &t.e_not_ep),
        ("no observable difference", &t.hidden_examples),
        (
            "HP but not EP, not through split clocks",
            &t.hp_not_ep_examples,
        ),
    ] {
        for p in list.iter().take(3) {
            println!("   {title}:\n      {p}");
        }
    }
}

type Sample = Vec<(Rated, Wlts<Option<char>, Real>)>;

/// Distinct random nets, and their classes under Markovian bisimilarity
/// with exponential clocks.
fn sample(
    seed: u64,
    count: usize,
    labels: u64,
    rates: u64,
    most: u64,
    parts: u64,
) -> (Sample, Vec<Vec<usize>>) {
    let mut rng = Rng(seed);
    let mut seen = std::collections::HashSet::new();
    let mut nets: Sample = Vec::new();
    for i in 0..count {
        let r = random_net(&mut rng, i % 2 == 0, labels, rates, most, parts);
        if !seen.insert(r.describe()) {
            continue;
        }
        if let Some(x) = expansion(&r, 1) {
            nets.push((r, x));
        }
    }
    let mut classes: Vec<Vec<usize>> = Vec::new();
    for (i, (_, x)) in nets.iter().enumerate() {
        match classes.iter_mut().find(|c| markovian(&nets[c[0]].1, x)) {
            Some(c) => c.push(i),
            None => classes.push(vec![i]),
        }
    }
    (nets, classes)
}

fn within_classes(nets: &Sample, classes: &[Vec<usize>]) -> Tally {
    let mut tally = Tally::default();
    for class in classes {
        for (k, &i) in class.iter().enumerate() {
            for &j in class.iter().skip(k + 1).take(12) {
                if compare(&nets[i].0, &nets[j].0, &mut tally).is_none() {
                    tally.skipped += 1;
                }
            }
        }
    }
    tally
}

fn main() {
    // 1. Random tiny nets, two labels, rates 1 or 2.
    let (nets, classes) = sample(0x5EED_1234, 4000, 2, 2, 4, 3);
    println!("{} distinct nets in {} classes", nets.len(), classes.len());
    report(
        "1. Random pairs within a class",
        &within_classes(&nets, &classes),
    );

    // 2. Every net with concurrency against its sequentialisation.
    let mut tally = Tally::default();
    for (r, _) in &nets {
        if let Some(s) = sequentialised(r) {
            if s.describe() != r.describe() {
                compare(r, &s, &mut tally);
            }
        }
    }
    println!();
    report("2. Nets against their sequentialisation", &tally);

    // 3. One label and one rate: as many coincidences as possible.
    let (nets, classes) = sample(0xC01_1DE5, 4000, 1, 1, 4, 3);
    println!(
        "\n{} distinct nets in {} classes",
        nets.len(),
        classes.len()
    );
    report(
        "3. One label, one rate: pairs within a class",
        &within_classes(&nets, &classes),
    );

    // 4. Larger nets: up to 7 transitions.
    let (nets, classes) = sample(0xB16_0E75, 6000, 2, 2, 7, 3);
    println!(
        "\n{} distinct nets in {} classes",
        nets.len(),
        classes.len()
    );
    report(
        "4. Up to 7 transitions: pairs within a class",
        &within_classes(&nets, &classes),
    );

    // 5. Where the proof does not reach: many clocks of one label and rate,
    // up to 4 components and 8 transitions.
    let (nets, classes) = sample(0x5_1DE_C0C, 4000, 1, 1, 8, 4);
    println!(
        "\n{} distinct nets in {} classes",
        nets.len(),
        classes.len()
    );
    report(
        "5. One label, one rate, up to 4 components: pairs within a class",
        &within_classes(&nets, &classes),
    );
}
