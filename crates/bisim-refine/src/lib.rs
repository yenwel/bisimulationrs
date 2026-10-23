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
//! - [`refine_fast`]: worklist partition refinement with dirty
//!   flags; same final partition as [`refine`], no history.
//! - [`game_bisimilar`]: on-the-fly bisimulation game over lazily
//!   discovered pairs, with budgets (`HoldsUpTo`/`Unknown`) and HML
//!   witnesses from Spoiler's winning strategy.
//! - `approx`: quasi-Markovian bisimilarity up to `(ε, δ)` tolerances
//!   (absolute class-sum deviation + relative exit-rate deviation), in the
//!   spirit of approximate probabilistic bisimulation for CTMCs (Spork et
//!   al., CONCUR 2024 and CAV 2025; not the same definition, see
//!   [`Tolerance`]). Reports `Verdict::Approx` with a behavioural distance
//!   instead of forcing exact equality.
//! - [`PrefixChecker`]: the same comparisons over prefixes that grow, with
//!   verdicts that stay sound for what has not been observed yet.
//! - [`compare`] / [`bisimilar`]: the user-facing checks on two systems.

mod approx;
mod causal;
mod compare;
mod fast;
mod game;
mod live;
mod refine;
mod simulation;
mod stream;

pub use approx::{Tolerance, approx_bisimilar_states, approx_distance, lumping_error};
pub use causal::{CausalBudget, CausalFormula, hp_bisimilar};
pub use compare::{
    NoInitialState, bisimilar, bisimilar_states, compare, minimise, try_bisimilar, try_compare,
};
pub use fast::refine_fast;
pub use game::{GameBudget, game_bisimilar};
pub use refine::{Refinement, distinguishing_formula, refine};
pub use simulation::{Preorder, simulation_preorder, simulation_witness};
pub use stream::{PrefixChecker, PrefixError};
