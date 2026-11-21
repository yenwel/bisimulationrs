//! Rated enabling-preserving bisimilarity against Erlang expansions, on
//! random rated nets that are Markovian bisimilar with exponential clocks.
//! EP implies that every Erlang expansion is Markovian bisimilar (by
//! construction); the converse is a conjecture (REVIEW.md, Idea 3), which
//! these pairs have not refuted. The larger search is the `durations`
//! example of the `bisimulationrs` crate.

use bisim_core::{NetTransition, Real, SafeNet, Wlts};
use bisim_refine::bisimilar;
use bisim_refine::durations::{erlang_expansion, rated_ep_bisimilar};

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

type Rated = (SafeNet<char>, Vec<f64>);

/// Up to 3 state machines of 2 or 3 places, one token each; up to 4
/// transitions moving one component or two; labels `a`, `b`; rates 1, 2.
fn random_net(rng: &mut Rng, cyclic: bool) -> Rated {
    let components = 1 + rng.below(3) as u32;
    let size = 2 + rng.below(2) as u32;
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
    let (mut transitions, mut rates) = (Vec::new(), Vec::new());
    for _ in 0..1 + rng.below(4) {
        let label = (b'a' + rng.below(2) as u8) as char;
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
        rates.push(1.0 + rng.below(2) as f64);
    }
    let initial = (0..components).map(|c| place(c, 0)).collect();
    let places = (0..components * size).map(|i| format!("p{i}")).collect();
    (SafeNet::new(places, transitions, initial), rates)
}

fn expansion(r: &Rated, stages: u32) -> Wlts<Option<char>, Real> {
    erlang_expansion(&r.0, &r.1, stages, 20_000).unwrap()
}

#[test]
fn ep_agrees_with_erlang_expansions() {
    let mut rng = Rng(0x5EED_1234);
    let nets: Vec<(Rated, Wlts<Option<char>, Real>)> = (0..400)
        .map(|i| {
            let r = random_net(&mut rng, i % 2 == 0);
            let x = expansion(&r, 1);
            (r, x)
        })
        .collect();
    let mut classes: Vec<Vec<usize>> = Vec::new();
    for (i, (_, x)) in nets.iter().enumerate() {
        match classes
            .iter_mut()
            .find(|c| bisimilar(&nets[c[0]].1, x).holds())
        {
            Some(c) => c.push(i),
            None => classes.push(vec![i]),
        }
    }
    let (mut pairs, mut same, mut different) = (0, 0, 0);
    for class in &classes {
        for (k, &i) in class.iter().enumerate() {
            for &j in class.iter().skip(k + 1).take(4) {
                let (a, b) = (&nets[i].0, &nets[j].0);
                let ep = rated_ep_bisimilar(&a.0, &a.1, &b.0, &b.1, 100_000)
                    .unwrap()
                    .expect("small nets fit the budget");
                for stages in [2, 3] {
                    let e = bisimilar(&expansion(a, stages), &expansion(b, stages)).holds();
                    assert_eq!(ep, e, "EP {ep}, Erlang-{stages} {e}:\n{a:?}\n{b:?}");
                }
                pairs += 1;
                if ep {
                    same += 1;
                } else {
                    different += 1;
                }
            }
        }
    }
    // The sample must contain both kinds of pairs to mean anything.
    assert!(
        pairs > 300 && same > 50 && different > 50,
        "{pairs} {same} {different}"
    );
}
