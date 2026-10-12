use std::collections::HashMap;
use std::hash::Hash;

use crate::ids::StateId;

/// A partition of `0..n` into numbered blocks.
///
/// Block numbers are canonical: blocks are numbered in order of their first
/// (lowest-indexed) state, so equal partitions compare equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    block_of: Vec<u32>,
    num_blocks: u32,
}

impl Partition {
    /// The partition with a single block.
    pub fn coarsest(n: usize) -> Self {
        Self {
            block_of: vec![0; n],
            num_blocks: u32::from(n > 0),
        }
    }

    /// Groups states with equal keys.
    pub fn from_keys<K: Eq + Hash>(keys: impl IntoIterator<Item = K>) -> Self {
        let mut ids: HashMap<K, u32> = HashMap::new();
        let block_of = keys
            .into_iter()
            .map(|k| {
                let next = ids.len() as u32;
                *ids.entry(k).or_insert(next)
            })
            .collect();
        Self {
            block_of,
            num_blocks: ids.len() as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.block_of.len()
    }

    pub fn is_empty(&self) -> bool {
        self.block_of.is_empty()
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks as usize
    }

    pub fn block_of(&self, s: StateId) -> u32 {
        self.block_of[s.index()]
    }

    pub fn same_block(&self, s: StateId, t: StateId) -> bool {
        self.block_of(s) == self.block_of(t)
    }

    /// Block index of every state, indexed by state.
    pub fn as_slice(&self) -> &[u32] {
        &self.block_of
    }

    /// The states of every block, in block order.
    pub fn blocks(&self) -> Vec<Vec<StateId>> {
        let mut blocks = vec![Vec::new(); self.num_blocks()];
        for (s, &b) in self.block_of.iter().enumerate() {
            blocks[b as usize].push(StateId(s as u32));
        }
        blocks
    }

    /// Restricts the partition to the states `range` and renumbers its blocks.
    pub fn restrict(&self, range: std::ops::Range<usize>) -> Partition {
        Partition::from_keys(self.block_of[range].iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_keys_is_canonical() {
        let p = Partition::from_keys(["b", "a", "b", "c"]);
        assert_eq!(p.as_slice(), &[0, 1, 0, 2]);
        assert_eq!(p.num_blocks(), 3);
        assert_eq!(p, Partition::from_keys([9, 3, 9, 1]));
        assert_eq!(p.restrict(1..4).as_slice(), &[0, 1, 2]);
    }
}
