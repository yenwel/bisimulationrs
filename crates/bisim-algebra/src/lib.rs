//! Process-algebra front ends and their operational semantics.
//!
//! Currently provides [`pepa`]: Hillston's Performance Evaluation Process
//! Algebra, with three equivalent ways to define a model (typed builder,
//! `.pepa` text, and the `pepa!` macro from the `bisimulationrs` facade) and
//! an SOS derivation into a rate-labelled transition system.

pub mod pepa;
