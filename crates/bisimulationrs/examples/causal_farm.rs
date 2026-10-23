//! Same steps, same rates, different dependencies.
//!
//! An ordinary comparison looks at which actions can happen and how fast.
//! A comparison up to causality also asks which earlier steps each step
//! depends on (history-preserving bisimulation on the causal net semantics
//! of PEPA; rates are left out of the causal part).
//!
//! 1. Two workers, one watering and one weeding, against one farmer doing
//!    both: the Markov chains are the same, but for the farmer every task
//!    waits on the previous one. One person is a bottleneck and a single
//!    point of failure.
//! 2. A herd with one vet against the same herd with two vets: as sequences
//!    of actions nothing changes, but with one vet each healing depends on
//!    the previous healing (through the vet).
//!
//! Read a witness `⟨a ← 1⟩` as "an `a` that depends on the 1st step so far"
//! and `⟨a ← ∅⟩` as "an `a` that depends on none of the earlier steps".
//!
//! Run: `cargo run -p bisimulationrs --example causal_farm`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::core::Verdict;
use bisimulationrs::refine::{CausalBudget, bisimilar, hp_bisimilar};

fn report(title: &str, a: &str, b: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (ma, mb) = (parse_pepa(a)?, parse_pepa(b)?);
    let markovian = bisimilar(&ma.derive()?.lts, &mb.derive()?.lts);
    let causal = hp_bisimilar(
        &ma.causal_net()?,
        &mb.causal_net()?,
        CausalBudget::default(),
    )?;
    println!("{title}");
    println!(
        "  same Markov chain (Markovian bisimilar): {}",
        markovian.holds()
    );
    match causal {
        Verdict::Holds => println!("  same causal structure: yes"),
        Verdict::Violated { witness } => {
            println!("  same causal structure: no");
            println!("  witness (true of the first, false of the second): {witness}");
        }
        other => println!("  same causal structure: {other:?}"),
    }
    println!();
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    report(
        "Two workers vs one farmer",
        "Waterer = (water, 1).Waterer; Weeder = (weed, 1).Weeder; Waterer || Weeder",
        "Farmer = (water, 1).Farmer + (weed, 1).Farmer; Farmer",
    )?;
    let herd = |vets: &str| {
        format!(
            "Healthy = (disease, 0.02).Sick; Sick = (healing, 0.1).Healthy;
             Vet = (healing, infty).Vet; Healthy[2] <healing> ({vets})"
        )
    };
    report("Two vets vs one vet", &herd("Vet || Vet"), &herd("Vet"))?;
    Ok(())
}
