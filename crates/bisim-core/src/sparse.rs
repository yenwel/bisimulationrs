use crate::semiring::Semiring;

/// Finitely supported function `K → W` (a FuTS continuation).
///
/// Entries are kept sorted by key and zero weights are never stored, so two
/// maps describing the same function have the same representation.
#[derive(Debug, Clone)]
pub struct SparseMap<K, W> {
    entries: Vec<(K, W)>,
}

impl<K, W> Default for SparseMap<K, W> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: Ord + Clone, W: Semiring> SparseMap<K, W> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `weight` to the value at `key` (accumulating multiplicities).
    ///
    /// For floating-point weights, repeated `add`s depend on their order in
    /// the last bits; collecting from an iterator does not (see
    /// [`Semiring::sum`]).
    pub fn add(&mut self, key: K, weight: W) {
        match self.entries.binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(i) => {
                let sum = self.entries[i].1.add(&weight);
                if sum.is_zero() {
                    self.entries.remove(i);
                } else {
                    self.entries[i].1 = sum;
                }
            }
            Err(i) => {
                if !weight.is_zero() {
                    self.entries.insert(i, (key, weight));
                }
            }
        }
    }

    pub fn get(&self, key: &K) -> W {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .map(|i| self.entries[i].1.clone())
            .unwrap_or_else(|_| W::zero())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &W)> {
        self.entries.iter().map(|(k, w)| (k, w))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Push-forward along `f`, summing the weights of keys that collide.
    pub fn map_sum<K2: Ord + Clone>(&self, mut f: impl FnMut(&K) -> K2) -> SparseMap<K2, W> {
        self.entries
            .iter()
            .map(|(k, w)| (f(k), w.clone()))
            .collect()
    }

    /// Canonical, hashable signature of this function.
    pub fn keys(&self) -> Vec<(K, W::Key)> {
        self.entries
            .iter()
            .map(|(k, w)| (k.clone(), w.key()))
            .collect()
    }
}

/// Weights of equal keys are added with [`Semiring::sum`], so the map does
/// not depend on the order of the items.
impl<K: Ord + Clone, W: Semiring> FromIterator<(K, W)> for SparseMap<K, W> {
    fn from_iter<I: IntoIterator<Item = (K, W)>>(iter: I) -> Self {
        let mut items: Vec<(K, W)> = iter.into_iter().collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        let entries = items
            .chunk_by(|a, b| a.0 == b.0)
            .filter_map(|run| {
                let sum = match run {
                    [(_, w)] => w.clone(),
                    _ => W::sum(run.iter().map(|(_, w)| w)),
                };
                (!sum.is_zero()).then(|| (run[0].0.clone(), sum))
            })
            .collect();
        SparseMap { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Real;

    #[test]
    fn accumulates_and_drops_zeros() {
        let map: SparseMap<u32, u64> = [(2, 1), (1, 3), (2, 4)].into_iter().collect();
        assert_eq!(map.keys(), vec![(1, 3), (2, 5)]);

        let pushed = map.map_sum(|_| 0u32);
        assert_eq!(pushed.keys(), vec![(0, 8)]);

        let mut reals = SparseMap::<u32, Real>::new();
        reals.add(7, Real(0.0));
        assert!(reals.is_empty());
    }
}
