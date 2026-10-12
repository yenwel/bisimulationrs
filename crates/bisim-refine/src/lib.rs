//! Algorithms deciding behavioural equivalences and preorders on finite
//! coalgebras from `bisim-core`.
//!
//! - [`refine`]: signature-based partition refinement, generic over any
//!   [`Signature`](bisim_core::Signature). It iterates the final chain
//!   `∼₀ ⊇ ∼₁ ⊇ …` to its greatest fixpoint and keeps every round, so callers
//!   get depth-bounded answers (`∼ₙ`) as well as the full equivalence.
//! - [`distinguishing_formula`]: a quantitative Hennessy–Milner formula that
//!   separates two inequivalent states (a certificate a user can check).
//! - [`simulation_preorder`]: the greatest simulation on an LTS, with
//!   existential-HML witnesses when one state fails to simulate another.
//! - [`compare`] / [`bisimilar`]: the user-facing checks on two systems.

mod compare;
mod refine;
mod simulation;

pub use compare::{bisimilar, bisimilar_states, compare, minimise};
pub use refine::{Refinement, distinguishing_formula, refine};
pub use simulation::{Preorder, simulation_preorder, simulation_witness};
