//! Biology, ecology and economics with the same tools.
//!
//! Four population models from `examples/` are solved with the fluid
//! (mean-field) approximation, which needs no enumeration of the state
//! space, and two of them are also checked exactly:
//!
//! - `coral`: a reef under heat waves (environment-dependent rates);
//! - `antibiotic`: drug cycling and the rise of resistance;
//! - `bass`: product adoption by word of mouth (a functional rate);
//! - `credit`: rating migration of a portfolio of firms.
//!
//! Run: `cargo run -p bisimulationrs --example domain_models`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::stochastic::integrate;

/// Local states as `(component, derivative)` and the expected count of each
/// at every requested time.
type Fluid = (Vec<(String, String)>, Vec<Vec<f64>>);

/// Expected population of every local state at the given times.
fn fluid(source: &str, times: &[f64]) -> Result<Fluid, Box<dyn std::error::Error>> {
    let model = parse_pepa(source)?.fluid_model()?;
    let snapshots = integrate(
        |x| model.fluid_drift(x),
        &model.initial_counts(),
        times,
        0.01,
    )?;
    Ok((model.fluid_slots(), snapshots))
}

fn print(
    title: &str,
    unit: &str,
    times: &[f64],
    slots: &[(String, String)],
    snaps: &[Vec<f64>],
    show: &[&str],
) {
    println!("\n{title} (expected counts, time in {unit})");
    print!("{:>8}", "time");
    for name in show {
        print!("{name:>12}");
    }
    println!();
    for (t, x) in times.iter().zip(snaps) {
        print!("{t:>8.0}");
        for name in show {
            let total: f64 = slots
                .iter()
                .zip(x)
                .filter(|((_, d), _)| d == name)
                .map(|(_, v)| v)
                .sum();
            print!("{total:>12.2}");
        }
        println!();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Coral: 40 corals; Calm/Heat are the environment's states.
    let times = [0.0, 26.0, 52.0, 104.0, 208.0];
    let (slots, snaps) = fluid(include_str!("../../../examples/coral/coral.pepa"), &times)?;
    print(
        "Coral reef",
        "weeks",
        &times,
        &slots,
        &snaps,
        &["Healthy", "Bleached", "Dead", "Heat"],
    );

    // Antibiotics: survivors and the resistant share.
    let times = [0.0, 10.0, 30.0, 60.0, 120.0];
    let (slots, snaps) = fluid(
        include_str!("../../../examples/antibiotic/antibiotic.pepa"),
        &times,
    )?;
    print(
        "Antibiotic cycling",
        "days",
        &times,
        &slots,
        &snaps,
        &["Susceptible", "Resistant", "Dead"],
    );

    // Bass: the S-shaped adoption curve against the closed form
    // F(t) = (1 - e^{-(p+q)t}) / (1 + (q/p) e^{-(p+q)t}).
    let times = [0.0, 6.0, 12.0, 18.0, 24.0];
    let (slots, snaps) = fluid(include_str!("../../../examples/bass/bass.pepa"), &times)?;
    print(
        "Bass diffusion",
        "months",
        &times,
        &slots,
        &snaps,
        &["Potential", "Adopter"],
    );
    let (p, q): (f64, f64) = (0.01, 0.4);
    print!("closed form, adopters at those times:");
    for t in times {
        let e = (-(p + q) * t).exp();
        print!("{:>12.2}", 1000.0 * (1.0 - e) / (1.0 + (q / p) * e));
    }
    println!("\n(the fluid limit solves the logistic equation behind this formula)");

    // Credit: fluid expectation for the portfolio against the exact
    // transient distribution of a single firm (firms are independent, so
    // the portfolio's expected defaults are 100 times the single firm's).
    let times = [0.0, 1.0, 5.0, 10.0, 20.0];
    let (slots, snaps) = fluid(include_str!("../../../examples/credit/credit.pepa"), &times)?;
    print(
        "Credit portfolio",
        "years",
        &times,
        &slots,
        &snaps,
        &["Strong", "Medium", "Weak", "Default"],
    );
    let one = parse_pepa(
        include_str!("../../../examples/credit/credit.pepa")
            .replace("Strong[100]", "Strong")
            .as_str(),
    )?
    .derive()?;
    let chain = bisimulationrs::stochastic::Ctmc::from_wlts(&one.lts);
    print!("exact single-firm chain x 100, defaults at those times:");
    for t in times {
        let p = chain.transient(&chain.initial_distribution(), t, 1e-12);
        let default: f64 = one
            .lts
            .states()
            .filter(|s| one.state_name(*s) == "Default")
            .map(|s| p[s.index()])
            .sum();
        print!("{:>12.2}", 100.0 * default);
    }
    println!();
    Ok(())
}
