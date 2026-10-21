//! Property tests of the bisimulation distance on seeded random systems,
//! against lumping by partition refinement as the oracle.

use bisim_core::{Partition, Real, StateId, Wlts, WltsBuilder};
use bisim_metric::{Chain, MetricOptions, bisim_distance, distance, normalise, uniformise};
use bisim_refine::refine_fast;

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

fn random_weighted(rng: &mut Rng, states: u32, edges: usize) -> Wlts<char, Real> {
    let mut b = WltsBuilder::new();
    b.ensure_states(states as usize);
    b.set_initial(StateId(0));
    for _ in 0..edges {
        let s = StateId(rng.below(states as u64) as u32);
        let t = StateId(rng.below(states as u64) as u32);
        let l = (b'a' + rng.below(2) as u8) as char;
        b.add_transition(s, l, t, Real(1.0 + rng.below(3) as f64));
    }
    b.build()
}

/// The same system with states renamed by `perm`.
fn permuted(sys: &Wlts<char, Real>, perm: &[u32]) -> Wlts<char, Real> {
    let mut b = WltsBuilder::new();
    b.ensure_states(sys.num_states());
    b.set_initial(StateId(perm[sys.initial().unwrap().index()]));
    for (s, l, t, w) in sys.transitions() {
        b.add_transition(StateId(perm[s.index()]), *l, StateId(perm[t.index()]), *w);
    }
    b.build()
}

/// Observation: parity of the state index.
fn parity(s: StateId) -> f64 {
    (s.0 % 2) as f64
}

/// Oracle: are `s` (left) and `t` (right) lumpable together in the
/// label-free CTMC (self-loops dropped), starting from equal observations?
fn lumped(
    left: &Wlts<char, Real>,
    s: StateId,
    right: &Wlts<char, Real>,
    t: StateId,
    obs_left: &dyn Fn(StateId) -> f64,
    obs_right: &dyn Fn(StateId) -> f64,
) -> bool {
    // Uniformised to a common rate q (self-loops topping every state up to
    // q), so that equal rates into the own block follow from equal rates
    // into every other block: Markovian bisimulation is then ordinary
    // lumpability.
    let n = left.num_states();
    let q = bisim_metric::common_rate(left, right);
    let mut b = WltsBuilder::new();
    b.ensure_states(n + right.num_states());
    for (sys, offset) in [(left, 0), (right, n)] {
        let mut exit = vec![0.0; sys.num_states()];
        for (x, _, y, w) in sys.transitions() {
            if x != y {
                exit[x.index()] += w.value();
                let (x, y) = ((x.index() + offset) as u32, (y.index() + offset) as u32);
                b.add_transition(StateId(x), 'τ', StateId(y), *w);
            }
        }
        for (x, e) in exit.iter().enumerate() {
            if q - e > 1e-12 {
                let x = StateId((x + offset) as u32);
                b.add_transition(x, 'τ', x, Real(q - e));
            }
        }
    }
    let union = b.build();
    let keys = (0..n)
        .map(|i| obs_left(StateId(i as u32)).to_bits())
        .chain((0..right.num_states()).map(|i| obs_right(StateId(i as u32)).to_bits()));
    let initial = Partition::from_keys(keys);
    let p = refine_fast(&union, Some(&initial));
    p.same_block(s, StateId((t.index() + n) as u32))
}

#[test]
fn distance_is_zero_exactly_for_bisimilar_states() {
    let mut rng = Rng(0xD157_A11C);
    let (mut zero, mut positive) = (0, 0);
    for round in 0..300 {
        let n = 2 + (round % 6) as u32;
        let left = random_weighted(&mut rng, n, n as usize * 2);
        let right = if round % 2 == 0 {
            // A renamed copy: bisimilar when the renaming keeps parity.
            let mut perm: Vec<u32> = (0..n).collect();
            for i in (1..perm.len()).rev() {
                perm.swap(i, rng.below(i as u64 + 1) as usize);
            }
            permuted(&left, &perm)
        } else {
            random_weighted(&mut rng, n, n as usize * 2)
        };
        let (s, t) = (left.initial().unwrap(), right.initial().unwrap());
        let d = bisim_distance(
            (&left, s),
            (&right, t),
            parity,
            parity,
            2.0,
            MetricOptions::default(),
        )
        .unwrap();
        let oracle = lumped(&left, s, &right, t, &parity, &parity);
        if oracle {
            zero += 1;
            assert!(
                d.value < 1e-5,
                "round {round}: bisimilar but distance {}",
                d.value
            );
        } else {
            positive += 1;
            assert!(
                d.value > 1e-5,
                "round {round}: not bisimilar but distance {}",
                d.value
            );
        }
        assert!(d.error_bound < 1e-5, "round {round}: {}", d.error_bound);
        assert!((0.0..=1.0 + 1e-9).contains(&d.value));
    }
    assert!(zero > 20 && positive > 20, "{zero} {positive}");
}

/// Three chains sharing one uniformisation rate, so their distances are
/// on one scale.
fn three(rng: &mut Rng, n: u32) -> [Chain; 3] {
    let systems = [0, 1, 2].map(|_| random_weighted(rng, n, n as usize * 2));
    let q = systems
        .iter()
        .map(|s| bisim_metric::common_rate(s, s))
        .fold(0.0, f64::max);
    systems.map(|s| uniformise(&s, q, parity))
}

#[test]
fn distance_is_symmetric_and_satisfies_the_triangle_inequality() {
    let mut rng = Rng(0x7121_A9C1);
    let opts = MetricOptions::default();
    for round in 0..100 {
        let [a, b, c] = three(&mut rng, 2 + (round % 5) as u32);
        let d = |x: &Chain, y: &Chain| {
            let (mut x, mut y) = (x.clone(), y.clone());
            // Parity is already in [0, 1]; normalising keeps it.
            normalise(&mut x, &mut y);
            distance(&x, &y, 0.9, opts).unwrap().value
        };
        let (ab, ba, bc, ac) = (d(&a, &b), d(&b, &a), d(&b, &c), d(&a, &c));
        assert!((ab - ba).abs() < 1e-5, "round {round}: {ab} vs {ba}");
        assert!(ac <= ab + bc + 1e-5, "round {round}: {ac} > {ab} + {bc}");
    }
}

#[test]
fn sinkhorn_value_iteration_bounds_the_exact_distance_from_above() {
    use bisim_metric::{SviOptions, distance_svi};
    let mut rng = Rng(0x5141_5EED);
    let mut worst: f64 = 0.0;
    for round in 0..60 {
        let n = 2 + (round % 5) as u32;
        let [mut a, mut b, _] = three(&mut rng, n);
        normalise(&mut a, &mut b);
        let exact = distance(&a, &b, 0.9, MetricOptions::default())
            .unwrap()
            .value;
        let svi = distance_svi(&a, &b, 0.9, SviOptions::default()).unwrap();
        // The cost of a genuine coupling: never below the distance.
        assert!(
            svi >= exact - 1e-6,
            "round {round}: svi {svi} < exact {exact}"
        );
        worst = worst.max(svi - exact);
    }
    println!("largest SVI excess over the exact distance: {worst:.4}");
    assert!(worst < 0.05, "SVI too loose: {worst}");
}
