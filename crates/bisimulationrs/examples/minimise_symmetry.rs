//! Lump a symmetric population: three identical animals share one vet.
//!
//! The full state space distinguishes which animal is sick; Markovian
//! bisimulation (lumpability) only needs to know how many are, and the
//! transient behaviour of the small chain is the same.
//!
//! Run: `cargo run -p bisimulationrs --example minimise_symmetry`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::refine::minimise;
use bisimulationrs::stochastic::Ctmc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = parse_pepa(
        "r_disease = 0.02;
         Healthy = (disease, r_disease).Sick;
         Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
         Dead = 0;
         Vet = (healing, infty).Vet;
         Healthy || Healthy || Healthy <healing> Vet",
    )?;
    let full = model.derive()?;
    let lumped = minimise(&full.lts);
    println!(
        "states: {} -> {} after lumping; transitions: {} -> {}",
        full.lts.num_states(),
        lumped.num_states(),
        full.lts.num_transitions(),
        lumped.num_transitions()
    );

    // Same behaviour over time: compare the probability of being absorbed
    // (all animals dead is the only deadlock) at day 30 in both chains.
    let absorbed = |sys: &bisimulationrs::core::Wlts<_, bisimulationrs::core::Real>| {
        let chain = Ctmc::from_wlts(sys);
        let p = chain.transient(&chain.initial_distribution(), 30.0, 1e-10);
        sys.states()
            .filter(|s| sys.successors(*s).is_empty())
            .map(|s| p[s.index()])
            .sum::<f64>()
    };
    println!(
        "P(absorbed by day 30): full {:.6}, lumped {:.6}",
        absorbed(&full.lts),
        absorbed(&lumped)
    );
    Ok(())
}
