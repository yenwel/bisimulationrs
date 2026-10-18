//! Long-run behaviour of a chain with very different time scales.
//!
//! A fast service (1000 per hour) with a rare failure (0.001) and a slow
//! repair (0.01). Power iteration stalls on such chains; the solver splits
//! the chain into its recurrent classes and solves each one exactly.
//!
//! Run: `cargo run -p bisimulationrs --example steady_state`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::stochastic::Ctmc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = parse_pepa(
        "Up = (work, 1000).Up2 + (fail, 0.001).Down;
         Up2 = (rest, 1000).Up;
         Down = (repair, 0.01).Up;
         Up",
    )?;
    let derived = model.derive()?;
    let chain = Ctmc::from_wlts(&derived.lts);
    let pi = chain.steady_state(1e-12, 1_000_000);
    for s in derived.lts.states() {
        println!("{:<6} {:.6}", derived.state_name(s), pi[s.index()]);
    }
    // Balance: `Down` is entered only from `Up` (0.001) and left at 0.01, and
    // `Up` and `Up2` swap at equal rates, so P(Down) = 0.1 / 2.1.
    println!("(expected P(Down) = {:.6})", 0.1 / 2.1);
    Ok(())
}
