//! Coalgebraic process-algebra primitives for comparing streaming and
//! stochastic systems.
//!
//! This facade re-exports the workspace crates:
//!
//! - [`core`]: weighted transition systems, semirings, partitions, verdicts;
//! - [`refine`]: bisimilarity, simulation, distinguishing formulas,
//!   quasi-`(ε, δ)` checks with distances;
//! - [`algebra`]: PEPA models (builder, `.pepa` parser, [`pepa!`]) and SOS;
//! - [`stochastic`]: CTMC steady-state/transient analysis and Gillespie SSA;
//! - [`metric`]: behavioural distances (optimal-transport bisimulation
//!   metric, (ε, δ)-bisimulation, distances from samples, dynamical
//!   similarity);
//! - [`streaming`]: verdict iterators and `futures_core::Stream`s over
//!   observations, updated incrementally per event;
//! - [`legacy`]: the 0.1 API, kept for compatibility.
//!
//! ```
//! use bisimulationrs::{pepa, refine::minimise, stochastic::Ctmc};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let herd = pepa! {
//!     r_disease = 0.02;
//!     Healthy = (disease, r_disease).Sick;
//!     Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
//!     Dead = 0;
//!     Vet = (healing, infty).Vet;
//!     Healthy[3] <healing> Vet
//! }
//! .derive()?;
//!
//! let lumped = minimise(&herd.lts); // Markovian bisimulation quotient
//! let ctmc = Ctmc::from_wlts(&lumped);
//! let p = ctmc.transient(&ctmc.initial_distribution(), 30.0, 1e-9); // t = 30 s
//! assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-6);
//! # Ok(())
//! # }
//! ```

// The test harness refers to the deprecated `legacy` tests by path.
#![cfg_attr(test, allow(deprecated))]

extern crate self as bisimulationrs;

pub use bisim_algebra as algebra;
pub use bisim_core as core;
pub use bisim_metric as metric;
pub use bisim_pepa_macros::pepa;
pub use bisim_refine as refine;
pub use bisim_stochastic as stochastic;

pub mod streaming;

#[deprecated(
    since = "0.2.0",
    note = "use `bisimulationrs::refine` (bisimilarity, simulation) and `bisimulationrs::algebra::pepa` instead"
)]
pub mod legacy;
