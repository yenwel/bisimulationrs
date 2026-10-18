//! PRISM guarded-command models: a subset of the PRISM language
//! (Kwiatkowska et al.) for DTMCs and CTMCs.
//!
//! ```text
//! ctmc
//! const double beta = 0.3;
//! formula spreading = s > 0 & i > 0;
//! module sir
//!   s : [0..10] init 9;
//!   i : [0..10] init 1;
//!   [infect] spreading -> beta*s*i : (s'=s-1) & (i'=i+1);
//!   [recover] i > 0 -> 0.1*i : (i'=i-1);
//! endmodule
//! ```
//!
//! Supported: `dtmc`/`ctmc` models, typed `const`s, `formula` macros,
//! one or more `module`s with bounded integer (or boolean) variables and
//! labelled commands with probabilistic/rate-weighted updates.
//! Multi-module commands synchronise CSP-style on shared labels (CTMC
//! rates multiply, as in PRISM); unlabelled commands interleave.
//!
//! Not supported (explicit errors): `mdp`, rewards, `init`/`system`
//! blocks, renaming, undefined-behaviour corners (out-of-bounds updates,
//! overlapping DTMC guards). `/` is real division (`1/2` is 0.5), as in
//! PRISM and JANI; `%` is a remainder, matching Rust rather than PRISM's
//! exact numerics — documented where it matters.
//!
//! [`PrismModel::derive`] enumerates the reachable valuations into a
//! [`Wlts`](bisim_core::Wlts) with probabilities/rates as weights, ready
//! for the same bisimulation and CTMC analysis as derived PEPA models.

mod ast;
mod derive;
mod parse;

pub use ast::{
    Assign, Cmp, Command, Expr, ModelKind, Module, PrismError, PrismModel, ProbQuery, Query,
    RewardBlock, RewardItem, SteadyQuery, SyncConfig, Until, Update, Value, Var, VarKind, VarType,
    eval,
};
pub use derive::{PrismDerived, VarInfo};
pub use parse::{parse_prism, parse_query};
