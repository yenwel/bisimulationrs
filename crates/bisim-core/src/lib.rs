//! Core coalgebraic types for `bisimulationrs`.
//!
//! The central data model is the *weighted labelled transition system*
//! [`Wlts<L, W>`]: a coalgebra `X → FS(X, W)^L` that maps every state to a
//! finitely supported function from successor states to semiring weights, one
//! per label (Latella, Massink & de Vink, *FuTS*). Choosing the semiring picks
//! the behavioural equivalence:
//!
//! | weight `W`       | system                     | bisimilarity                         |
//! |------------------|----------------------------|--------------------------------------|
//! | `bool`           | LTS ([`Lts`])              | strong bisimilarity                  |
//! | `u64`            | multi-LTS                  | bisimilarity counting multiplicities |
//! | [`Real`] (rates) | labelled CTMC (e.g. PEPA)  | Markovian bisimulation / lumpability |
//! | [`Real`] (probs) | labelled DTMC              | probabilistic bisimulation           |
//!
//! Units: times are in seconds and rates in s⁻¹ throughout the workspace.

mod formula;
mod ids;
mod net;
mod partition;
mod semiring;
mod signature;
mod sparse;
mod verdict;
mod wlts;

pub use formula::Formula;
pub use ids::{Interner, LabelId, StateId};
pub use net::{Contact, Marking, NetTransition, ReachError, SafeNet};
pub use partition::Partition;
pub use semiring::{Real, RealKey, Semiring, SignatureKey};
pub use signature::Signature;
pub use sparse::SparseMap;
pub use verdict::{ComparisonVerdict, Verdict};
pub use wlts::{Edge, Lts, NamedBuilder, Wlts, WltsBuilder};
