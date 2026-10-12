use std::hash::Hash;

use crate::ids::StateId;

/// One step of the coalgebra, observed through a partition.
///
/// `signature(s, block_of)` is `H(π)(ξ(s))`: the structure map applied to `s`
/// with successor states replaced by their block. Two states are behaviourally
/// equivalent iff their signatures agree at the greatest fixpoint of the
/// refinement this induces (Latella, Massink & de Vink's class-sum lifting).
pub trait Signature {
    type Sig: Eq + Hash + Clone;

    fn num_states(&self) -> usize;

    fn signature(&self, s: StateId, block_of: &[u32]) -> Self::Sig;
}
