//! Comparing logged systems without fitting a model first.
//!
//! Each plot has 3 plants; aphids infest and get cleared, and a weekly
//! inspection records how many plants are infested (0–3). The distance is
//! estimated from the inspection logs alone (Calo et al., NeurIPS 2025):
//! every pair of consecutive inspections is one observed transition. The
//! logs here are simulated, from a "true" infestation rate and from three
//! candidate plans; the candidate closest to the truth should come out
//! closest.
//!
//! The estimator needs a lot of data: here 400 plots × 250 weeks are pooled
//! per candidate. A single plot's few years of weekly records would leave
//! the estimate far from settled (its error shrinks like 1/√transitions).
//!
//! Run: `cargo run --release -p bisimulationrs --example inspection_logs`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::core::StateId;
use bisimulationrs::metric::SampleDistance;
use bisimulationrs::stochastic::gillespie;

fn bed(infest: f64) -> String {
    format!(
        "infest = {infest}; clear = 0.5;
         Healthy = (infest, infest).Infested;
         Infested = (clear, clear).Healthy;
         Healthy[3]"
    )
}

/// Weekly inspections (number infested) of `plots` simulated plots.
fn logs(
    infest: f64,
    plots: u64,
    weeks: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let m = parse_pepa(&bed(infest))?.derive()?;
    let infested: Vec<usize> = m
        .lts
        .states()
        .map(|s| {
            m.local_counts(s, 0)
                .into_iter()
                .filter(|(d, _)| d == "Infested")
                .map(|(_, n)| n as usize)
                .sum()
        })
        .collect();
    let start = m.lts.initial().unwrap();
    Ok((0..plots)
        .map(|p| {
            let run = gillespie(&m.lts, start, weeks as f64, 1_000_000, seed * 1_000_003 + p);
            let mut state: StateId = start;
            let mut steps = run.steps.iter().peekable();
            (0..weeks)
                .map(|w| {
                    while let Some(step) = steps.next_if(|s| s.time <= w as f64) {
                        state = step.target;
                    }
                    infested[state.index()]
                })
                .collect()
        })
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (plots, weeks, gamma) = (400, 250, 0.8);
    let truth = logs(0.15, plots, weeks, 1)?;
    println!("true infestation rate 0.15 per week; distances from the inspection logs:");
    for (i, candidate) in [0.05, 0.15, 0.40].into_iter().enumerate() {
        let plan = logs(candidate, plots, weeks, 10 + i as u64)?;
        // Cost: difference in plants infested, as a fraction of the bed.
        let mut est = SampleDistance::new(4, 4, |x, y| x.abs_diff(y) as f64 / 3.0, gamma, (0, 0))?;
        for (a, b) in truth.iter().zip(&plan) {
            for w in 1..weeks {
                est.push((a[w - 1], a[w]), (b[w - 1], b[w]));
            }
        }
        println!(
            "  plan with rate {candidate:.2}: distance {:.3} ({} transitions)",
            est.estimate(),
            est.steps()
        );
    }
    println!(
        "The matching plan is closest but not at 0: the estimate is still settling (error \
         shrinking like 1/√transitions), so compare candidates on logs of equal size rather \
         than reading the value as exact."
    );
    Ok(())
}
