//! Behavioural distances between systems, for comparisons that need more
//! than yes or no.
//!
//! - [`kantorovich`]: the optimal-transport bisimulation metric (Calo et
//!   al., NeurIPS 2024): a distance in `[0, 1]` between two models, and the
//!   state pairs that drive it.
//! - [`epsdelta`]: (ε, δ)-bisimulation for CTMCs (Spork et al., CAV 2025)
//!   with its bounds on time-bounded reachability.
//! - [`samples`]: the same distance estimated from observed transitions
//!   only (Calo et al., NeurIPS 2025), one step per observation.
//! - [`svi`]: Sinkhorn Value Iteration from the same paper, an upper bound
//!   for small dense chains.
//! - [`dsa`]: dynamical similarity analysis of continuous sensor series
//!   (Ostrow et al., NeurIPS 2023); not a bisimulation.
//! - [`linalg`], [`flow`]: the small dense linear algebra and network-flow
//!   solvers the methods share.

pub mod dsa;
pub mod epsdelta;
pub mod flow;
pub mod kantorovich;
pub mod linalg;
pub mod samples;
pub mod svi;

pub use dsa::{Dsa, DsaOptions, dsa, fit_operator, procrustes};
pub use epsdelta::{
    EpsDeltaRelation, MAX_EPSDELTA_STATES, epsdelta, epsdelta_bisimilar, min_epsilon,
    rate_only_bound, reach_bound,
};
pub use kantorovich::{
    Chain, Distance, MAX_METRIC_PAIRS, MetricError, MetricOptions, PairContribution,
    bisim_distance, common_rate, discount, distance, normalise, uniformise,
};
pub use samples::{MAX_SAMPLE_PAIRS, SampleDistance, StepSizes, occupancy_sample};
pub use svi::{MAX_SVI_PAIRS, SviOptions, distance_svi};
