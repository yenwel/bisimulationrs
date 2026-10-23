//! Benchmark suites for the roadmap workloads: dense finite LTSs,
//! sparse stochastic traces (SIR/garden family) and cyclic systems.
//!
//! Run with `cargo bench -p bisimulationrs`. The groups compare the naive
//! reference (`refine`) against the worklist backend (`refine_fast`),
//! which is what `minimise` and the API analyses use.

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::algebra::prism::parse_prism;
use bisimulationrs::core::{Lts, Semiring, StateId, WltsBuilder};
use bisimulationrs::refine::{GameBudget, game_bisimilar, minimise, refine, refine_fast};
use bisimulationrs::stochastic::{Ctmc, ensemble, integrate};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

/// xorshift64*: deterministic workload generation.
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

fn dense_lts(states: u32, edges: usize, labels: u8) -> Lts<char> {
    let mut rng = Rng(0xD395E);
    let mut b = WltsBuilder::new();
    b.ensure_states(states as usize);
    b.set_initial(StateId(0));
    for _ in 0..edges {
        b.add_transition(
            StateId(rng.below(states as u64) as u32),
            (b'a' + rng.below(labels as u64) as u8) as char,
            StateId(rng.below(states as u64) as u32),
            true,
        );
    }
    b.build()
}

const CYCLE: &str = "r = 1; P1 = (start, r).P2; P2 = (run, r).P3; P3 = (stop, r).P1;";

fn bench_refinement(c: &mut Criterion) {
    let mut group = c.benchmark_group("refinement/dense-lts");
    for (n, m) in [(100u32, 1000usize), (300, 3000)] {
        let sys = dense_lts(n, m, 4);
        group.bench_with_input(BenchmarkId::new("refine", n), &sys, |b, sys| {
            b.iter(|| refine(black_box(sys), None));
        });
        group.bench_with_input(BenchmarkId::new("refine_fast", n), &sys, |b, sys| {
            b.iter(|| refine_fast(black_box(sys), None));
        });
        group.bench_with_input(BenchmarkId::new("minimise", n), &sys, |b, sys| {
            b.iter(|| minimise(black_box(sys)));
        });
    }
    group.finish();
}

fn bench_structured(c: &mut Criterion) {
    // PEPA-derived models: large stable regions where the worklist
    // backend skips rehashing (its best case; dense graphs are worst).
    let mut group = c.benchmark_group("structured/pepa-models");
    let garden =
        include_str!("../../../examples/garden/tomatoes.pepa").replace("Tomato[24]", "Tomato[8]");
    let sir = sir_fun(9);
    for (name, src) in [("garden-8", garden.to_string()), ("sir-10", sir)] {
        let model = parse_pepa(&src).unwrap().derive().unwrap();
        group.bench_with_input(BenchmarkId::new("refine", name), &model, |b, model| {
            b.iter(|| refine(black_box(&model.lts), None));
        });
        group.bench_with_input(BenchmarkId::new("refine_fast", name), &model, |b, model| {
            b.iter(|| refine_fast(black_box(&model.lts), None));
        });
    }
    group.finish();

    // Deep chains: one split per round, the worklist's best case.
    let mut group = c.benchmark_group("structured/chains");
    for n in [200u32, 500] {
        let mut b = WltsBuilder::new();
        b.ensure_states(n as usize + 1);
        for i in 0..n {
            b.add_transition(StateId(i), 'a', StateId(i + 1), true);
        }
        let chain = b.build();
        group.bench_with_input(BenchmarkId::new("refine", n), &chain, |b, chain| {
            b.iter(|| refine(black_box(chain), None));
        });
        group.bench_with_input(BenchmarkId::new("refine_fast", n), &chain, |b, chain| {
            b.iter(|| refine_fast(black_box(chain), None));
        });
    }
    group.finish();
}

fn bench_cycles(c: &mut Criterion) {
    let mut group = c.benchmark_group("cycles");
    for copies in [4u32, 6, 8] {
        let src = format!("{CYCLE} P1[{copies}]");
        let model = parse_pepa(&src).unwrap().derive().unwrap();
        group.bench_with_input(BenchmarkId::new("minimise", copies), &model, |b, model| {
            b.iter(|| minimise(black_box(&model.lts)));
        });
        // The game runs on the untimed skeleton (boolean LTS).
        let untimed = model.lts.map_weights(|w| !w.is_zero());
        group.bench_with_input(BenchmarkId::new("game", copies), &untimed, |b, untimed| {
            b.iter(|| {
                game_bisimilar(
                    black_box(untimed),
                    StateId(0),
                    black_box(untimed),
                    StateId(1),
                    GameBudget::default(),
                )
            });
        });
    }
    group.finish();
}

fn sir_fun(n: u32) -> String {
    format!(
        "beta = 0.3; gamma = 0.1;
         S = (infect, beta * #I).I;
         I = (recover, gamma).R;
         R = 0;
         S[{n}] || I[1]"
    )
}

fn bench_stochastic(c: &mut Criterion) {
    let mut group = c.benchmark_group("stochastic/sir-transient");
    for n in [10u32, 20] {
        let model = parse_pepa(&sir_fun(n)).unwrap().derive().unwrap();
        let ctmc = Ctmc::from_wlts(&model.lts);
        let p0 = ctmc.initial_distribution();
        group.bench_with_input(
            BenchmarkId::new("transient", n),
            &(&ctmc, &p0),
            |b, (ctmc, p0)| {
                b.iter(|| ctmc.transient(black_box(p0), black_box(10.0), black_box(1e-9)));
            },
        );
        group.bench_with_input(BenchmarkId::new("ssa-ensemble", n), &model, |b, model| {
            b.iter(|| {
                ensemble(
                    black_box(&model.lts),
                    StateId(0),
                    black_box(10.0),
                    black_box(1_000),
                    black_box(50),
                    black_box(1),
                )
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("stochastic/fluid-no-enumeration");
    for n in [100u32, 1000] {
        // Fluid models never enumerate: thousands of copies stay cheap.
        let model = parse_pepa(&sir_fun(n)).unwrap().fluid_model().unwrap();
        let x0 = model.initial_counts();
        group.bench_with_input(
            BenchmarkId::new("rk4-30d", n),
            &(&model, &x0),
            |b, (model, x0)| {
                b.iter(|| {
                    integrate(
                        |x| model.fluid_drift(x),
                        black_box(x0),
                        black_box(&[30.0]),
                        black_box(0.05),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_prism(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism");
    let src = include_str!("../../../examples/sir/sir.prism");
    group.bench_function("sir-derive", |b| {
        b.iter(|| parse_prism(black_box(src)).unwrap().derive().unwrap());
    });
    let model = parse_prism(src).unwrap().derive().unwrap();
    let ctmc = Ctmc::from_wlts(&model.lts);
    let p0 = ctmc.initial_distribution();
    group.bench_function("sir-transient", |b| {
        b.iter(|| ctmc.transient(black_box(&p0), black_box(30.0), black_box(1e-9)));
    });
    group.finish();
}

/// One event of a revealed prefix.
#[derive(Clone, Copy)]
enum Reveal {
    Left((u32, char, u32)),
    Right((u32, char, u32)),
    /// Completes the state on both sides.
    Complete(u32),
}

/// Two identical rings revealed one transition at a time (alternating
/// sides, each state completed once its transitions are in). Every state
/// reaches the last one to complete, so nothing settles before the end.
fn ring_events(k: u32) -> Vec<Reveal> {
    let mut events = Vec::new();
    for i in 0..k {
        for e in [(i, 'a', (i + 1) % k), (i, 'b', (i * 7 + 3) % k)] {
            events.push(Reveal::Left(e));
            events.push(Reveal::Right(e));
        }
        events.push(Reveal::Complete(i));
    }
    events
}

/// A never-ending harvest: a spine `8i —a→ 8(i+1)` whose states each start
/// a finished episode (`b` then `c` or `d`), completed as soon as it is in.
/// The right system also offers a poorer `b` episode (`c` only), so
/// bisimilarity is violated while each system simulates the other, and the
/// poorer answers are refuted along the way. The spine stays open; the
/// episodes settle.
fn harvest_events(k: u32) -> Vec<Reveal> {
    let mut events = Vec::new();
    for i in 0..k {
        let (s, e, f, g, poor) = (8 * i, 8 * i + 1, 8 * i + 2, 8 * i + 3, 8 * i + 4);
        for t in [(s, 'a', s + 8), (s, 'b', e), (e, 'c', f), (e, 'd', g)] {
            events.push(Reveal::Left(t));
            events.push(Reveal::Right(t));
        }
        events.push(Reveal::Right((s, 'b', poor)));
        events.push(Reveal::Right((poor, 'c', f)));
        for state in [s, e, f, g, poor] {
            events.push(Reveal::Complete(state));
        }
    }
    events
}

#[derive(Clone, Copy, PartialEq)]
enum Solve {
    Incremental,
    /// Incremental, compacting whenever the live positions have doubled.
    Compact {
        minimise: bool,
    },
    Replay,
}

/// The size of a checker over a run.
#[derive(Default)]
struct Footprint {
    /// Largest over the run (estimated heap bytes are only tracked when
    /// measuring).
    peak_positions: usize,
    peak_bytes: usize,
    /// At the end, after one last compaction in the compacting modes.
    live_positions: usize,
    held_pairs: usize,
    heap_bytes: usize,
}

/// Reveals `events` with a verdict after each; returns how many verdicts
/// were bisimilarity violations and the footprint (heap bytes after every
/// event only when `measure`).
fn run_prefix(events: &[Reveal], solve: Solve, measure: bool) -> (usize, Footprint) {
    use bisimulationrs::refine::{GameBudget, PrefixChecker};
    let mut c = PrefixChecker::new(0u32, 0u32).with_budget(GameBudget {
        max_pairs: usize::MAX,
        max_depth: u32::MAX,
    });
    let mut violated = 0;
    let mut compacted_at = 1;
    let mut footprint = Footprint::default();
    for &event in events {
        match event {
            Reveal::Left(e) => c.observe_left([e]).expect("open state"),
            Reveal::Right(e) => c.observe_right([e]).expect("open state"),
            Reveal::Complete(s) => {
                c.complete_left(s);
                c.complete_right(s);
            }
        }
        if let Solve::Compact { minimise } = solve {
            if c.live_positions() >= 2 * compacted_at {
                if minimise {
                    c.compact_and_minimise();
                } else {
                    c.compact();
                }
                compacted_at = c.live_positions().max(1);
            }
        }
        let v = if solve == Solve::Replay {
            c.replay_verdict()
        } else {
            c.verdict()
        };
        violated += usize::from(v.bisimilar.is_violated());
        footprint.peak_positions = footprint.peak_positions.max(c.live_positions());
        if measure {
            footprint.peak_bytes = footprint.peak_bytes.max(c.heap_bytes());
        }
    }
    if let Solve::Compact { minimise } = solve {
        if minimise {
            c.compact_and_minimise();
        } else {
            c.compact();
        }
    }
    footprint.live_positions = c.live_positions();
    footprint.held_pairs = c.held_pairs();
    footprint.heap_bytes = c.heap_bytes();
    (violated, footprint)
}

fn bench_prefix(c: &mut Criterion) {
    type Workload = (&'static str, fn(u32) -> Vec<Reveal>, [u32; 2]);
    let workloads: [Workload; 2] = [
        ("ring", ring_events, [250, 1_000]),
        ("harvest", harvest_events, [100, 400]),
    ];
    let modes = [
        ("incremental", Solve::Incremental),
        ("compact", Solve::Compact { minimise: false }),
        ("minimise", Solve::Compact { minimise: true }),
    ];
    // Memory: live positions and estimated heap bytes (games and observed
    // systems) at their peak over the run, and at the end (after one last
    // compaction in the compacting modes, which otherwise compact when the
    // live positions have doubled).
    eprintln!("prefix_stream footprint:");
    eprintln!(
        "{:<8} {:>6} {:<12} {:>10} {:>11} {:>10} {:>8} {:>11}",
        "workload", "events", "mode", "peak pos", "peak bytes", "end pos", "held", "end bytes"
    );
    for (name, events, sizes) in workloads {
        for k in sizes {
            let events = events(k);
            for (mode, solve) in modes {
                let (_, f) = run_prefix(&events, solve, true);
                eprintln!(
                    "{name:<8} {:>6} {mode:<12} {:>10} {:>11} {:>10} {:>8} {:>11}",
                    events.len(),
                    f.peak_positions,
                    f.peak_bytes,
                    f.live_positions,
                    f.held_pairs,
                    f.heap_bytes
                );
            }
        }
    }

    let mut group = c.benchmark_group("prefix_stream");
    group.sample_size(10);
    for (name, events, sizes) in workloads {
        for k in sizes {
            let events = events(k);
            // Counted in observed transitions, as before completions
            // became events.
            let n = events
                .iter()
                .filter(|e| !matches!(e, Reveal::Complete(_)))
                .count() as u64;
            let mut runs = modes.to_vec();
            runs.push(("replay", Solve::Replay));
            for (mode, solve) in runs {
                if solve == Solve::Replay && k > sizes[0] && name == "harvest" {
                    continue; // quadratic: the small size makes the point
                }
                let id = BenchmarkId::new(format!("{name}/{mode}"), n);
                group.bench_with_input(id, &events, |b, events| {
                    b.iter(|| black_box(run_prefix(events, solve, false).0))
                });
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_prefix,
    bench_refinement,
    bench_structured,
    bench_cycles,
    bench_stochastic,
    bench_prism
);
criterion_main!(benches);
