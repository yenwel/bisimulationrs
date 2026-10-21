//! Comparing continuous sensor series by their dynamics.
//!
//! Sensors in a bed report continuous signals: here the fluid (mean-field)
//! trajectories of the guild model in `examples/permaculture/`, sampled
//! every half week. Dynamical similarity analysis (Ostrow et al., NeurIPS
//! 2023) fits linear dynamics to each series and compares them up to a
//! change of coordinates. Not a bisimulation: it compares how signals
//! evolve, which suits measurements that are not discrete states.
//!
//! - The same bed seen through differently mixed sensors: about 0.
//! - A bed without the companion plants (no repelling): clearly different.
//!
//! Run: `cargo run --release -p bisimulationrs --example sensor_similarity`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::metric::{DsaOptions, dsa};
use bisimulationrs::stochastic::integrate;

/// Fluid trajectory of the guild with repelling factor `repel`, sampled
/// every half week for a season.
fn sensors(repel: f64) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let source = include_str!("../../../examples/permaculture/guild.pepa")
        .replace("repel = 0.40;", &format!("repel = {repel};"));
    let model = parse_pepa(&source)?.fluid_model()?;
    let times: Vec<f64> = (0..120).map(|k| k as f64 * 0.5).collect();
    Ok(integrate(
        |x| model.fluid_drift(x),
        &model.initial_counts(),
        &times,
        0.05,
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let guild = sensors(0.40)?;
    // Each "sensor" reads a fixed orthogonal mix of the five quantities
    // (healthy, infested, lost, resting and hunting ladybirds).
    let mixed: Vec<Vec<f64>> = guild
        .iter()
        .map(|x| {
            let (c, s) = (0.8, 0.6);
            vec![c * x[0] + s * x[1], -s * x[0] + c * x[1], x[2], x[4], -x[3]]
        })
        .collect();
    let opts = DsaOptions {
        delays: 2,
        rank: 4,
        ..DsaOptions::default()
    };
    let same = dsa(&guild, &mixed, opts)?;
    let other = dsa(&guild, &sensors(0.0)?, opts)?;
    println!(
        "same bed, mixed sensors:      distance {:.2e}, angle {:.2}°",
        same.distance,
        same.angle.to_degrees()
    );
    println!(
        "bed without companion plants: distance {:.2e}, angle {:.2}°",
        other.distance,
        other.angle.to_degrees()
    );
    Ok(())
}
