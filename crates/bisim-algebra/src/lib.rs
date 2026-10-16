//! Process-algebra front ends and their operational semantics.
//!
//! Currently provides two front ends with the same explicit-state,
//! rate-labelled output:
//!
//! - [`pepa`]: Hillston's Performance Evaluation Process Algebra, with
//!   three equivalent ways to define a model (typed builder, `.pepa`
//!   text, and the `pepa!` macro from the `bisimulationrs` facade) and
//!   an SOS derivation into a rate-labelled transition system.
//!   Bio-PEPA-style functional rates (`beta * #I`) are supported.
//! - [`prism`]: a subset of the PRISM guarded-command language (`dtmc`
//!   and `ctmc`, constants, formulas, synchronising modules), imported
//!   into the same labelled-transition-system world.

pub mod jani;
pub mod pepa;
pub mod prism;
