//! A learned challenger for the bisimulation game, against the exact
//! solvers (experimental, ROADMAP §4e).
//!
//! A plan and a run that deviates from it: each bundled PEPA model (untimed
//! skeleton) is compared with a copy that lost one transition at a state some
//! distance from the start. Three solvers look for the difference:
//!
//! - `bisimilar`: partition refinement of both systems (exact);
//! - `game_bisimilar`: the on-the-fly game, breadth first (exact);
//! - the learned challenger: a value network trained by self-play orders a
//!   beam-limited depth-first search. Its witness is re-checked exactly; a
//!   failed search is `Unknown`, never "same". "untrained" is the same search
//!   ordered by an untrained network, to show what the training adds.
//!
//! Positions are expanded state pairs; the challenger's include training.
//!
//! Run: `cargo run --release -p bisimulationrs --features learned-challenger --example learned_challenger`

use std::collections::VecDeque;
use std::hash::Hash;
use std::time::{Duration, Instant};

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::core::{Formula, Lts, Semiring, Verdict, WltsBuilder};
use bisimulationrs::refine::challenger::{
    ChallengeBudget, ChallengerOptions, game_positions, train_challenger,
};
use bisimulationrs::refine::{GameBudget, bisimilar};

/// Bundled models with their population sizes reduced.
const MODELS: [(&str, &str, &str, &str); 7] = [
    (
        "sir",
        include_str!("../../../examples/sir/sir.pepa"),
        "S[9]",
        "S[9]",
    ),
    (
        "tomatoes",
        include_str!("../../../examples/garden/tomatoes.pepa"),
        "Tomato[24]",
        "Tomato[8]",
    ),
    (
        "guild",
        include_str!("../../../examples/permaculture/guild.pepa"),
        "Healthy[60] || Resting[12]",
        "Healthy[8] || Resting[2]",
    ),
    (
        "coral",
        include_str!("../../../examples/coral/coral.pepa"),
        "Healthy[40]",
        "Healthy[12]",
    ),
    (
        "credit",
        include_str!("../../../examples/credit/credit.pepa"),
        "Strong[100]",
        "Strong[20]",
    ),
    (
        "antibiotic",
        include_str!("../../../examples/antibiotic/antibiotic.pepa"),
        "Susceptible[200]",
        "Susceptible[20]",
    ),
    (
        "bass",
        include_str!("../../../examples/bass/bass.pepa"),
        "Potential[1000]",
        "Potential[40]",
    ),
];

/// Breadth-first distance of every state from the initial one.
fn distances<L: Eq + Hash + Clone>(sys: &Lts<L>) -> Vec<Option<usize>> {
    let mut dist = vec![None; sys.num_states()];
    let start = sys.initial().expect("initial state");
    dist[start.index()] = Some(0);
    let mut queue = VecDeque::from([start]);
    while let Some(s) = queue.pop_front() {
        for e in sys.successors(s) {
            if dist[e.target.index()].is_none() {
                dist[e.target.index()] = dist[s.index()].map(|d| d + 1);
                queue.push_back(e.target);
            }
        }
    }
    dist
}

/// `sys` without the first transition of the first state at `distance`.
fn without_transition<L: Eq + Hash + Clone>(sys: &Lts<L>, distance: usize) -> Lts<L> {
    let dist = distances(sys);
    let victim = sys
        .states()
        .find(|s| dist[s.index()] == Some(distance) && !sys.successors(*s).is_empty())
        .expect("a state at that distance");
    let mut b = WltsBuilder::new();
    b.ensure_states(sys.num_states());
    let mut skipped = false;
    for (s, l, t, w) in sys.transitions() {
        if s == victim && !skipped {
            skipped = true;
            continue;
        }
        b.add_transition(s, l.clone(), t, *w);
    }
    b.set_initial(sys.initial().expect("initial state"));
    b.build()
}

fn verdict<L>(v: &Verdict<Formula<L, bool>>) -> String {
    match v {
        Verdict::Violated { witness } => format!("differs (depth {})", witness.depth()),
        Verdict::Holds => "same".into(),
        Verdict::Unknown => "unknown".into(),
        _ => "other".into(),
    }
}

fn ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1e3)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "| model | states | removed at | refinement | game: positions, time | training: positions, time | learned search: positions, time | beam 4, depth 64: positions | untrained: positions |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let (mut found, mut found_wide, mut runs) = (0, 0, 0);
    for (name, src, from, to) in MODELS {
        let model = parse_pepa(&src.replace(from, to))?.derive()?;
        let plan = model.lts.map_weights(|w| !w.is_zero());
        let depth = distances(&plan)
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(0);
        for d in [depth / 4, depth / 2, 3 * depth / 4] {
            let run = without_transition(&plan, d);
            let (s, t) = (plan.initial().unwrap(), run.initial().unwrap());

            let clock = Instant::now();
            let exact = bisimilar(&plan, &run);
            let refine_time = clock.elapsed();

            let clock = Instant::now();
            let (game, game_pos) = game_positions(&plan, s, &run, t, GameBudget::default());
            let game_time = clock.elapsed();

            let clock = Instant::now();
            let mut ch = train_challenger(&plan, &run, &[], ChallengerOptions::default());
            let train_time = clock.elapsed();
            let clock = Instant::now();
            let (learned, stats) = ch.challenge(s, t, ChallengeBudget::default());
            let search_time = clock.elapsed();
            let wide_budget = ChallengeBudget {
                beam: 4,
                max_depth: 64,
                ..ChallengeBudget::default()
            };
            let (wide, wide_stats) = ch.challenge(s, t, wide_budget);

            let untrained_options = ChallengerOptions {
                train_positions: 0,
                ..ChallengerOptions::default()
            };
            let mut blind = train_challenger(&plan, &run, &[], untrained_options);
            let (untrained, blind_stats) = blind.challenge(s, t, ChallengeBudget::default());

            if exact.is_violated() {
                runs += 1;
                for (v, n) in [(&learned, &mut found), (&wide, &mut found_wide)] {
                    if v.is_violated() {
                        *n += 1;
                    }
                }
            }
            println!(
                "| {name} | {} | {d} | {} in {} | {} in {}, {} | {} in {} | {} in {}, {} | {}, {} | {}, {} |",
                plan.num_states(),
                verdict(&exact),
                ms(refine_time),
                game_pos,
                ms(game_time),
                verdict(&game),
                ch.training.positions,
                ms(train_time),
                stats.positions,
                ms(search_time),
                verdict(&learned),
                wide_stats.positions,
                verdict(&wide),
                blind_stats.positions,
                verdict(&untrained),
            );
        }
    }
    println!(
        "\nlearned challenger: {found} of {runs} differences found (beam 2, depth 32), \
         {found_wide} (beam 4, depth 64)"
    );
    Ok(())
}
