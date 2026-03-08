use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::{Duration, Instant};

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

/// A PEPA transition from one derivative to another.
#[derive(Debug, Clone, PartialEq)]
pub struct PepaTransition {
    pub action: String,
    pub rate: f64,
    pub target: String,
}

/// A PEPA sequential component and its outgoing transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct PepaComponent {
    pub name: String,
    pub transitions: Vec<PepaTransition>,
}

/// A lightweight PEPA process runtime for deterministic benchmarking workloads.
#[derive(Debug, Clone)]
pub struct PepaProcess {
    initial_component: String,
    current_component: String,
    components: HashMap<String, PepaComponent>,
}

impl PepaProcess {
    pub fn new(initial_component: impl Into<String>, components: Vec<PepaComponent>) -> Self {
        let initial_component = initial_component.into();
        let component_map = components
            .into_iter()
            .map(|c| (c.name.clone(), c))
            .collect::<HashMap<_, _>>();

        Self {
            initial_component: initial_component.clone(),
            current_component: initial_component,
            components: component_map,
        }
    }

    pub fn current_component(&self) -> &str {
        &self.current_component
    }
}

/// Grouped-PEPA population state over PEPA derivatives.
#[derive(Debug, Clone)]
pub struct GroupedPepaProcess {
    initial_population: HashMap<String, usize>,
    population: HashMap<String, usize>,
    components: HashMap<String, PepaComponent>,
}

impl GroupedPepaProcess {
    pub fn new(components: Vec<PepaComponent>, initial_population: HashMap<String, usize>) -> Self {
        let component_map = components
            .into_iter()
            .map(|c| (c.name.clone(), c))
            .collect::<HashMap<_, _>>();

        Self {
            initial_population: initial_population.clone(),
            population: initial_population,
            components: component_map,
        }
    }

    pub fn population(&self) -> &HashMap<String, usize> {
        &self.population
    }
}

/// Minimal runtime abstraction used by the benchmark harness.
pub trait ProcessRuntime {
    fn reset(&mut self);
    fn enabled_actions(&self) -> Vec<String>;
    fn fire_action(&mut self, action: &str) -> bool;
    fn state_cardinality(&self) -> usize;
}

impl ProcessRuntime for PepaProcess {
    fn reset(&mut self) {
        self.current_component = self.initial_component.clone();
    }

    fn enabled_actions(&self) -> Vec<String> {
        let mut actions = self
            .components
            .get(&self.current_component)
            .map(|component| {
                component
                    .transitions
                    .iter()
                    .map(|transition| transition.action.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        actions.sort();
        actions
    }

    fn fire_action(&mut self, action: &str) -> bool {
        let next_target = self
            .components
            .get(&self.current_component)
            .and_then(|component| {
                component
                    .transitions
                    .iter()
                    .find(|transition| transition.action == action)
            })
            .map(|transition| transition.target.clone());

        if let Some(target) = next_target {
            self.current_component = target;
            true
        } else {
            false
        }
    }

    fn state_cardinality(&self) -> usize {
        1
    }
}

impl ProcessRuntime for GroupedPepaProcess {
    fn reset(&mut self) {
        self.population = self.initial_population.clone();
    }

    fn enabled_actions(&self) -> Vec<String> {
        let mut actions = HashSet::new();

        for (component_name, count) in &self.population {
            if *count == 0 {
                continue;
            }

            if let Some(component) = self.components.get(component_name) {
                for transition in &component.transitions {
                    actions.insert(transition.action.clone());
                }
            }
        }

        let mut sorted = actions.into_iter().collect::<Vec<_>>();
        sorted.sort();
        sorted
    }

    fn fire_action(&mut self, action: &str) -> bool {
        let mut component_names = self.population.keys().cloned().collect::<Vec<_>>();
        component_names.sort();

        for component_name in component_names {
            let Some(count) = self.population.get(&component_name).copied() else {
                continue;
            };

            if count == 0 {
                continue;
            }

            let target = self
                .components
                .get(&component_name)
                .and_then(|component| {
                    component
                        .transitions
                        .iter()
                        .find(|transition| transition.action == action)
                })
                .map(|transition| transition.target.clone());

            if let Some(target_component) = target {
                if let Some(source_count) = self.population.get_mut(&component_name) {
                    *source_count -= 1;
                }
                *self.population.entry(target_component).or_insert(0) += 1;
                return true;
            }
        }

        false
    }

    fn state_cardinality(&self) -> usize {
        self.population.values().copied().sum()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BenchmarkConfig {
    pub warmup_steps: usize,
    pub measured_steps: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 0,
            measured_steps: 10_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub configured_steps: usize,
    pub executed_steps: usize,
    pub elapsed: Duration,
    pub steps_per_second: f64,
    pub action_counts: HashMap<String, usize>,
    pub final_state_cardinality: usize,
}

/// Deterministic process benchmark harness for PEPA and grouped-PEPA models.
#[derive(Debug, Default)]
pub struct ProcessBenchmark;

impl ProcessBenchmark {
    pub fn run<M>(model: &mut M, config: BenchmarkConfig) -> BenchmarkReport
    where
        M: ProcessRuntime,
    {
        model.reset();

        for _ in 0..config.warmup_steps {
            let actions = model.enabled_actions();
            let Some(action) = actions.first() else {
                break;
            };
            let _ = model.fire_action(action);
        }

        let start = Instant::now();
        let mut executed_steps = 0;
        let mut action_counts = HashMap::new();

        for _ in 0..config.measured_steps {
            let actions = model.enabled_actions();
            let Some(action) = actions.first().cloned() else {
                break;
            };

            if !model.fire_action(&action) {
                break;
            }

            *action_counts.entry(action).or_insert(0) += 1;
            executed_steps += 1;
        }

        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let steps_per_second = if elapsed_secs > 0.0 {
            executed_steps as f64 / elapsed_secs
        } else {
            0.0
        };

        BenchmarkReport {
            configured_steps: config.measured_steps,
            executed_steps,
            elapsed,
            steps_per_second,
            action_counts,
            final_state_cardinality: model.state_cardinality(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toggling_components() -> Vec<PepaComponent> {
        vec![
            PepaComponent {
                name: "P0".to_string(),
                transitions: vec![PepaTransition {
                    action: "tick".to_string(),
                    rate: 1.0,
                    target: "P1".to_string(),
                }],
            },
            PepaComponent {
                name: "P1".to_string(),
                transitions: vec![PepaTransition {
                    action: "tick".to_string(),
                    rate: 1.0,
                    target: "P0".to_string(),
                }],
            },
        ]
    }

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

    #[test]
    fn benchmark_infrastructure_runs_for_pepa() {
        let mut process = PepaProcess::new("P0", toggling_components());
        let report = ProcessBenchmark::run(
            &mut process,
            BenchmarkConfig {
                warmup_steps: 2,
                measured_steps: 20,
            },
        );

        assert_eq!(report.configured_steps, 20);
        assert_eq!(report.executed_steps, 20);
        assert_eq!(report.final_state_cardinality, 1);
        assert_eq!(report.action_counts.get("tick"), Some(&20));
    }

    #[test]
    fn benchmark_infrastructure_runs_for_grouped_pepa() {
        let mut initial_population = HashMap::new();
        initial_population.insert("P0".to_string(), 3);
        initial_population.insert("P1".to_string(), 0);

        let mut grouped = GroupedPepaProcess::new(toggling_components(), initial_population);
        let report = ProcessBenchmark::run(
            &mut grouped,
            BenchmarkConfig {
                warmup_steps: 0,
                measured_steps: 12,
            },
        );

        assert_eq!(report.executed_steps, 12);
        assert_eq!(report.final_state_cardinality, 3);
        assert_eq!(report.action_counts.get("tick"), Some(&12));
    }
}
