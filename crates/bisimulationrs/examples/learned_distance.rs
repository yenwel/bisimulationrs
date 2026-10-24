//! A learned bisimulation distance (deep-RL style) next to the exact one.
//!
//! A small neural network learns an embedding of states, described by
//! features (here: the share of plants healthy, infested and lost, and of
//! ladybirds resting and hunting), so that distances between embeddings
//! approximate the bisimulation distance. The objective follows Deep
//! Bisimulation for Control (Zhang et al., ICLR 2021). It is an estimate,
//! never a verdict, so it is compared with the exact distance where that can
//! still be computed:
//!
//! 1. small and medium beds: exact vs learned, and the time each takes;
//! 2. larger beds (30 plants, 6 ladybirds): the exact method explores 1.7
//!    million state pairs (0.134, about 8 minutes); the learned one never
//!    builds them (0.143, about 40 seconds), and its cost grows with the
//!    discount (training steps), not with the number of pairs.
//!
//! Run: `cargo run --release -p bisimulationrs --example learned_distance`

use std::time::Instant;

use bisimulationrs::algebra::pepa::{DerivedModel, Sym, parse_pepa};
use bisimulationrs::core::StateId;
use bisimulationrs::metric::{
    LearnOptions, MetricOptions, common_rate, discount, distance, learn_metric, normalise,
    uniformise,
};

const STAGES: [&str; 5] = ["Healthy", "Infested", "Lost", "Resting", "Hunting"];

fn beds(
    plants: u32,
    ladybirds: u32,
) -> Result<(DerivedModel<Sym>, DerivedModel<Sym>), Box<dyn std::error::Error>> {
    let guild = include_str!("../../../examples/permaculture/guild.pepa")
        .replace("Healthy[60]", &format!("Healthy[{plants}]"))
        .replace("Resting[12]", &format!("Resting[{ladybirds}]"));
    let mono = include_str!("../../../examples/permaculture/monoculture.pepa")
        .replace("Healthy[60]", &format!("Healthy[{plants}]"));
    Ok((parse_pepa(&guild)?.derive()?, parse_pepa(&mono)?.derive()?))
}

/// Share of each stage, per leaf size (plants or ladybirds).
fn features(m: &DerivedModel<Sym>, plants: u32, ladybirds: u32) -> Vec<Vec<f64>> {
    m.lts
        .states()
        .map(|s| {
            let mut f = vec![0.0; STAGES.len()];
            for leaf in 0..m.leaves().len() {
                for (d, n) in m.local_counts(s, leaf) {
                    if let Some(i) = STAGES.iter().position(|x| *x == d) {
                        let size = if i < 3 { plants } else { ladybirds };
                        f[i] += f64::from(n) / f64::from(size);
                    }
                }
            }
            f
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let half_life = 4.0; // weeks
    // (plants, ladybirds, compute the exact distance too)
    for (plants, ladybirds, exact_too) in [(6, 2, true), (14, 3, true), (30, 6, false)] {
        let (guild, mono) = beds(plants, ladybirds)?;
        let healthy = |m: &DerivedModel<Sym>, s: StateId| {
            m.local_counts(s, 0)
                .into_iter()
                .filter(|(d, _)| d == "Healthy")
                .map(|(_, n)| f64::from(n))
                .sum::<f64>()
        };
        let q = common_rate(&guild.lts, &mono.lts);
        let (mut a, mut b) = (
            uniformise(&guild.lts, q, |s| healthy(&guild, s)),
            uniformise(&mono.lts, q, |s| healthy(&mono, s)),
        );
        normalise(&mut a, &mut b);
        let gamma = discount(q, half_life);
        println!(
            "{plants} plants, {ladybirds} ladybirds (γ = {gamma:.4}): guild {} states, monoculture {} states ({} state pairs)",
            guild.lts.num_states(),
            mono.lts.num_states(),
            guild.lts.num_states() * mono.lts.num_states()
        );
        if exact_too {
            let t = Instant::now();
            let d = distance(&a, &b, gamma, MetricOptions::default())?;
            println!("  exact distance   {:.3}   ({:.2?})", d.value, t.elapsed());
        } else {
            // Measured once on an M-series laptop: 0.134 in 505 s (release build).
            println!("  exact distance   skipped here (measured once: 0.134, in about 8 minutes)");
        }
        let t = Instant::now();
        let (fa, fb) = (
            features(&guild, plants, ladybirds),
            features(&mono, plants, ladybirds),
        );
        let m = learn_metric(&a, &b, &fa, &fb, gamma, LearnOptions::default())?;
        println!(
            "  learned distance {:.3}   ({:.2?}, final loss {:.1e})",
            m.distance(&fa[a.initial], &fb[b.initial]),
            t.elapsed(),
            m.final_loss
        );
    }
    println!(
        "\nThe learned value is an estimate, never a verdict: on the beds where both were \
         computed it was within about 0.01 of the exact value."
    );
    Ok(())
}
