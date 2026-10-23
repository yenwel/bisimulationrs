//! PEPA: Performance Evaluation Process Algebra (Hillston, 1996).
//!
//! A model has two syntactic levels, mirrored by two AST types:
//!
//! ```text
//! S ::= (α, r).S | S + S | C_S | 0          sequential components   (Seq)
//! P ::= P ⋈_L P | P / L | C | C[n]          model components        (ModelTerm)
//! ```
//!
//! Rates are in s⁻¹; `infty` (⊤) marks a passive rate whose value is set by
//! the cooperation partner. `#{derivative}` is a species count for
//! Bio-PEPA-style functional rates (e.g. mass action `beta * #I`),
//! evaluated per global state during derivation. Three front ends produce
//! the same [`PepaModel`]:
//!
//! - [`PepaModel::builder`]: a typed builder (actions can be a Rust enum);
//! - [`parse_pepa`]: the textual syntax of the PEPA Eclipse plug-in;
//! - `bisimulationrs::pepa!`: the same syntax checked at compile time.
//!
//! [`PepaModel::derive`] explores the state space by structural operational
//! semantics and returns a [`Wlts`](bisim_core::Wlts) with rates as weights,
//! ready for Markovian bisimulation (lumping) and CTMC analysis.

mod ast;
mod builder;
mod derive;
mod net;
mod parse;
mod print;
mod rate;
mod validate;

pub use ast::{Action, ActionSet, ModelTerm, ParamId, PepaModel, RateExpr, Seq, SeqId, Sym};
pub use builder::{PepaBuilder, PrefixStart, act, count, passive, passive_weighted, stop};
pub use derive::{
    Act, DeriveOptions, DerivedModel, FluidModel, LeafInfo, LocalGraph, LocalTransition,
};
pub use net::MAX_CAUSAL_COMPONENTS;
pub use parse::{ParseError, parse_pepa};
pub use rate::{FunExpr, Rate};
pub use validate::PepaError;
