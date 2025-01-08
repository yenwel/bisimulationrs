use std::collections::{HashMap, HashSet};

struct LabelledTransitionSystem {
    states: Vec<String>,
    transitions: HashMap<String, (String, String)>,
}

impl LabelledTransitionSystem {
    fn new(states: Vec<String>, transitions: HashMap<String, (String, String)>) -> Self {
        LabelledTransitionSystem { states, transitions }
    }

    fn bisimulates(&self, other: &LabelledTransitionSystem) -> bool {
        // Check if states and transitions are identical for a trivial case.
        if self.states.len() != other.states.len() || self.transitions.len() != other.transitions.len() {
            return false;
        }

        // Initialize partitions
        let mut partitions: Vec<HashSet<String>> = vec![self.states.iter().cloned().collect()];

        loop {
            let mut refined_partitions = vec![];

            for partition in &partitions {
                let mut sub_partitions: HashMap<Vec<(String, String)>, HashSet<String>> = HashMap::new();

                for state in partition {
                    // Compute the signature of the current state
                    let signature: Vec<(String, String)> = self
                        .transitions
                        .iter()
                        .filter_map(|(label, (src, tgt))| {
                            if src == state {
                                Some((label.clone(), partitions.iter().position(|p| p.contains(tgt)).unwrap_or(usize::MAX).to_string()))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Group states by their signatures
                    sub_partitions
                        .entry(signature)
                        .or_insert_with(HashSet::new)
                        .insert(state.clone());
                }

                // Add all sub-partitions to refined_partitions
                refined_partitions.extend(sub_partitions.values().cloned());
            }

            if partitions == refined_partitions {
                break;
            }

            partitions = refined_partitions;
        }

        // Check final partitions for equivalence
        let self_partition = partitions.clone();
        let other_partition = Self::compute_partitions(&other, &partitions);

        self_partition == other_partition
    }

    fn compute_partitions(lts: &LabelledTransitionSystem, partitions: &Vec<HashSet<String>>) -> Vec<HashSet<String>> {
        let mut new_partitions = vec![];

        for partition in partitions {
            let mut sub_partitions: HashMap<Vec<(String, String)>, HashSet<String>> = HashMap::new();

            for state in partition {
                // Compute the signature for the state in `lts`
                let signature: Vec<(String, String)> = lts
                    .transitions
                    .iter()
                    .filter_map(|(label, (src, tgt))| {
                        if src == state {
                            Some((label.clone(), partitions.iter().position(|p| p.contains(tgt)).unwrap_or(usize::MAX).to_string()))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Group states by their signatures
                sub_partitions
                    .entry(signature)
                    .or_insert_with(HashSet::new)
                    .insert(state.clone());
            }

            new_partitions.extend(sub_partitions.values().cloned());
        }

        new_partitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisimilarity() {
        let states = vec!["born".to_string(), "dead".to_string(), "sick".to_string(), "healthy".to_string()];
        let transitions: HashMap<String, (String, String)> = [
            ("disease".to_string(), ("born".to_string(), "sick".to_string())),
            ("death".to_string(), ("sick".to_string(), "dead".to_string())),
            ("healing".to_string(), ("sick".to_string(), "healthy".to_string())),
        ]
        .iter()
        .cloned()
        .collect();
        let lts_one = LabelledTransitionSystem::new(states.clone(), transitions.clone());
        let lts_two = LabelledTransitionSystem::new(states.clone(), transitions.clone());

        assert!(lts_one.bisimulates(&lts_two));

        let transitions_two: HashMap<String, (String, String)> = [
            ("disease".to_string(), ("born".to_string(), "sick".to_string())),
            ("death".to_string(), ("sick".to_string(), "dead".to_string())),
        ]
        .iter()
        .cloned()
        .collect();
        let lts_three = LabelledTransitionSystem::new(states, transitions_two);

        assert!(!lts_one.bisimulates(&lts_three));
    }
}
