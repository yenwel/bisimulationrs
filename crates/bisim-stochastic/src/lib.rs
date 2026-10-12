//! Stochastic analysis of rate-labelled systems.
//!
//! [`Ctmc`] turns a [`Wlts`](bisim_core::Wlts) whose weights are rates (s⁻¹),
//! such as a derived PEPA model, into a continuous-time Markov chain and
//! computes steady-state and transient distributions by uniformisation.
//! Markovian bisimulation (`bisim_refine::minimise`) yields an ordinarily
//! lumpable partition, so the minimised chain gives the same aggregated
//! distributions on a smaller state space.
//!
//! [`CountDistribution`] turns those probabilities into the distribution of
//! a count (how many copies are in some states), with ranges and p-values
//! for comparing a plan with observed counts; [`nelder_mead`] fits rates.

mod count;
mod ctmc;
mod optimize;

pub use count::CountDistribution;
pub use ctmc::Ctmc;
pub use optimize::nelder_mead;
