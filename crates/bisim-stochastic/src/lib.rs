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
//!
//! [`gillespie`] and [`ensemble`]
//! provide Gillespie SSA trajectories: the stochastic race semantics shared
//! by the Bio-PEPA, PRISM and Modest simulators, for Monte-Carlo estimates
//! when the full chain is too large to solve.
//!
//! [`reach`] answers CSL/PCTL-style reachability queries:
//! unbounded and step-bounded reachability on DTMCs by value iteration,
//! time-bounded reachability on CTMCs by uniformisation.

mod count;
mod ctmc;
pub mod fluid;
mod optimize;
pub mod reach;
mod simulate;

pub use count::CountDistribution;
pub use ctmc::Ctmc;
pub use fluid::{integrate, integrate_fixed};
pub use optimize::nelder_mead;
pub use reach::{MatrixError, bounded_steps, dtmc_matrix, unbounded_reach};
pub use simulate::{LearnedRate, LogEvent, SsaStep, Trajectory, ensemble, gillespie, learn_rates};
