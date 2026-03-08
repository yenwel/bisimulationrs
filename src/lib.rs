use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// A finite labelled transition system (LTS).
#[derive(Debug, Clone)]
pub struct LabelledTransitionSystem {
    states: Vec<String>,
    transitions: HashMap<String, (String, String)>,
}

impl LabelledTransitionSystem {
    pub fn new(states: Vec<String>, transitions: HashMap<String, (String, String)>) -> Self {
        LabelledTransitionSystem {
            states,
            transitions,
        }
    }

    /// Classic partition-refinement style bisimulation check for finite systems.
    pub fn bisimulates(&self, other: &LabelledTransitionSystem) -> bool {
        if self.states.len() != other.states.len()
            || self.transitions.len() != other.transitions.len()
        {
            return false;
        }

        let mut partitions: Vec<HashSet<String>> = vec![self.states.iter().cloned().collect()];

        loop {
            let mut refined_partitions = vec![];

            for partition in &partitions {
                let mut sub_partitions: HashMap<Vec<(String, String)>, HashSet<String>> =
                    HashMap::new();

                for state in partition {
                    let signature: Vec<(String, String)> = self
                        .transitions
                        .iter()
                        .filter_map(|(label, (src, tgt))| {
                            if src == state {
                                Some((
                                    label.clone(),
                                    partitions
                                        .iter()
                                        .position(|p| p.contains(tgt))
                                        .unwrap_or(usize::MAX)
                                        .to_string(),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect();

                    sub_partitions
                        .entry(signature)
                        .or_default()
                        .insert(state.clone());
                }

                refined_partitions.extend(sub_partitions.values().cloned());
            }

            if partitions == refined_partitions {
                break;
            }

            partitions = refined_partitions;
        }

        let self_partition = partitions.clone();
        let other_partition = Self::compute_partitions(other, &partitions);

        self_partition == other_partition
    }

    fn compute_partitions(
        lts: &LabelledTransitionSystem,
        partitions: &[HashSet<String>],
    ) -> Vec<HashSet<String>> {
        let mut new_partitions = vec![];

        for partition in partitions {
            let mut sub_partitions: HashMap<Vec<(String, String)>, HashSet<String>> =
                HashMap::new();

            for state in partition {
                let signature: Vec<(String, String)> = lts
                    .transitions
                    .iter()
                    .filter_map(|(label, (src, tgt))| {
                        if src == state {
                            Some((
                                label.clone(),
                                partitions
                                    .iter()
                                    .position(|p| p.contains(tgt))
                                    .unwrap_or(usize::MAX)
                                    .to_string(),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();

                sub_partitions
                    .entry(signature)
                    .or_default()
                    .insert(state.clone());
            }

            new_partitions.extend(sub_partitions.values().cloned());
        }

        new_partitions
    }
}

/// A single observed transition at a point in time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservedTransition<S, L> {
    pub source: S,
    pub label: L,
    pub target: S,
}

/// Incremental verdict for a single simulation direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixVerdict {
    HoldsSoFar,
    Violated,
}

/// Streaming output that mirrors your intended API:
/// - does right simulate left?
/// - are they mutually simulation-equivalent on seen prefixes?
/// - are they bisimilar on seen prefixes?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingVerdict {
    pub right_simulates_left: PrefixVerdict,
    pub are_bisimulate: PrefixVerdict,
    pub are_bisimilar: PrefixVerdict,
}

/// Stateful incremental checker over transition prefixes.
#[derive(Debug, Clone)]
pub struct PrefixBisimulationChecker<S, L>
where
    S: Eq + Hash + Clone,
    L: Eq + Hash + Clone,
{
    left_seen: HashSet<ObservedTransition<S, L>>,
    right_seen: HashSet<ObservedTransition<S, L>>,
}

impl<S, L> Default for PrefixBisimulationChecker<S, L>
where
    S: Eq + Hash + Clone,
    L: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self {
            left_seen: HashSet::new(),
            right_seen: HashSet::new(),
        }
    }
}

impl<S, L> PrefixBisimulationChecker<S, L>
where
    S: Eq + Hash + Clone,
    L: Eq + Hash + Clone,
{
    pub fn observe_left<I>(&mut self, transitions: I)
    where
        I: IntoIterator<Item = ObservedTransition<S, L>>,
    {
        self.left_seen.extend(transitions);
    }

    pub fn observe_right<I>(&mut self, transitions: I)
    where
        I: IntoIterator<Item = ObservedTransition<S, L>>,
    {
        self.right_seen.extend(transitions);
    }

    pub fn verdict(&self) -> StreamingVerdict {
        let right_simulates_left = if self.left_seen.is_subset(&self.right_seen) {
            PrefixVerdict::HoldsSoFar
        } else {
            PrefixVerdict::Violated
        };

        let left_simulates_right = if self.right_seen.is_subset(&self.left_seen) {
            PrefixVerdict::HoldsSoFar
        } else {
            PrefixVerdict::Violated
        };

        let are_bisimulate = if right_simulates_left == PrefixVerdict::HoldsSoFar
            && left_simulates_right == PrefixVerdict::HoldsSoFar
        {
            PrefixVerdict::HoldsSoFar
        } else {
            PrefixVerdict::Violated
        };

        StreamingVerdict {
            right_simulates_left,
            are_bisimulate,
            are_bisimilar: are_bisimulate,
        }
    }

    pub fn left_prefix(&self) -> &HashSet<ObservedTransition<S, L>> {
        &self.left_seen
    }

    pub fn right_prefix(&self) -> &HashSet<ObservedTransition<S, L>> {
        &self.right_seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisimilarity() {
        let states = vec![
            "born".to_string(),
            "dead".to_string(),
            "sick".to_string(),
            "healthy".to_string(),
        ];
        let transitions: HashMap<String, (String, String)> = [
            (
                "disease".to_string(),
                ("born".to_string(), "sick".to_string()),
            ),
            (
                "death".to_string(),
                ("sick".to_string(), "dead".to_string()),
            ),
            (
                "healing".to_string(),
                ("sick".to_string(), "healthy".to_string()),
            ),
        ]
        .iter()
        .cloned()
        .collect();
        let lts_one = LabelledTransitionSystem::new(states.clone(), transitions.clone());
        let lts_two = LabelledTransitionSystem::new(states.clone(), transitions.clone());

        assert!(lts_one.bisimulates(&lts_two));

        let transitions_two: HashMap<String, (String, String)> = [
            (
                "disease".to_string(),
                ("born".to_string(), "sick".to_string()),
            ),
            (
                "death".to_string(),
                ("sick".to_string(), "dead".to_string()),
            ),
        ]
        .iter()
        .cloned()
        .collect();
        let lts_three = LabelledTransitionSystem::new(states, transitions_two);

        assert!(!lts_one.bisimulates(&lts_three));
    }

    #[test]
    fn test_streaming_prefix_verdicts() {
        let mut checker = PrefixBisimulationChecker::<&str, &str>::default();

        checker.observe_left([ObservedTransition {
            source: "born",
            label: "disease",
            target: "sick",
        }]);

        // Right has not yet caught up with left observations.
        assert_eq!(
            checker.verdict().right_simulates_left,
            PrefixVerdict::Violated
        );

        checker.observe_right([ObservedTransition {
            source: "born",
            label: "disease",
            target: "sick",
        }]);

        assert_eq!(checker.verdict().are_bisimilar, PrefixVerdict::HoldsSoFar);

        checker.observe_right([ObservedTransition {
            source: "sick",
            label: "death",
            target: "dead",
        }]);

        // Right now has a behavior not present in left prefix.
        assert_eq!(checker.verdict().are_bisimulate, PrefixVerdict::Violated);
    }
}
