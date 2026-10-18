//! Import a PRISM model and answer a question about it.
//!
//! A queue with capacity 3: jobs arrive at rate 2 and are served at rate 3.
//! How likely is the queue full in the long run, and how soon is it first
//! full? (Check the long-run answer against the closed form for M/M/1/3.)
//!
//! Run: `cargo run -p bisimulationrs --example prism_model`

use bisimulationrs::algebra::prism::parse_prism;
use bisimulationrs::stochastic::Ctmc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = parse_prism(
        "ctmc
         const double lambda = 2;
         const double mu = 3;
         module queue
           n : [0..3] init 0;
           [] n<3 -> lambda : (n'=n+1);
           [] n>0 -> mu : (n'=n-1);
         endmodule",
    )?;
    let derived = model.derive()?;
    let chain = Ctmc::from_wlts(&derived.lts);
    let full: Vec<bool> = derived
        .lts
        .states()
        .map(|s| derived.state_name(s).contains("n = 3"))
        .collect();

    let pi = chain.steady_state(1e-12, 100_000);
    let long_run: f64 = pi
        .iter()
        .zip(&full)
        .filter(|(_, f)| **f)
        .map(|(p, _)| p)
        .sum();
    // M/M/1/K: pi_n proportional to rho^n with rho = lambda / mu.
    let rho: f64 = 2.0 / 3.0;
    let exact = rho.powi(3) / (1.0 + rho + rho.powi(2) + rho.powi(3));
    println!("long-run P(full) = {long_run:.6} (exact {exact:.6})");

    let within_10 = chain.reach_bounded(&chain.initial_distribution(), &full, 10.0, 1e-10);
    println!("P(full within 10 time units) = {within_10:.6}");
    Ok(())
}
