//! A small, stable API over `bisimulationrs` for language bindings.
//!
//! Everything here takes PEPA source text and plain numbers, and returns
//! plain data (tables and graphs) that serialises to JSON. The web editor,
//! the Python package and the R package are thin wrappers over these
//! functions, so they all give the same answers.
//!
//! Rates are "per unit of time"; the analysis functions use that unit
//! directly (seconds unless you say otherwise). The observation functions
//! take a start date and a [`TimeUnit`] to turn calendar dates (dd/mm/yyyy)
//! into model time.

use bisim_algebra::pepa::{Act, DeriveOptions, DerivedModel, PepaError, Rate, Sym, parse_pepa};
use bisim_algebra::prism::{
    ModelKind, PrismDerived, PrismError, PrismModel, Query, Until, parse_prism, parse_query,
};
use bisim_core::{Formula, Lts, Partition, Real, Semiring, StateId, Verdict, Wlts};
use bisim_refine::{compare as compare_lts, refine_fast};
use bisim_stochastic::{Ctmc, bounded_steps, dtmc_matrix, unbounded_reach};
use serde::Serialize;

/// Most steps of size `dt` a fluid request may need.
const MAX_FLUID_STEPS: f64 = 5e6;

/// Largest step bound accepted in `F<=k` queries (a linear-time loop).
const MAX_STEP_BOUND: f64 = 1e8;

mod engine;
mod observe;

pub use observe::{
    FitReport, FittedParam, Forecast, Observation, ObservationReport, ObservationResult, Status,
    TimeUnit, Timeline, check_observations, fit_parameters, forecast, format_date, parse_date,
};

#[cfg(feature = "json")]
mod json;
#[cfg(feature = "json")]
pub use json::{PROTOCOL_VERSION, handle_json};

/// An error or warning in a model, with a 1-based position when known
/// (`line = 0` means the problem is not tied to one place).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub line: usize,
    pub column: usize,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

fn diagnostic(e: &PepaError) -> Diagnostic {
    match e {
        PepaError::Parse(p) => Diagnostic {
            severity: Severity::Error,
            line: p.line,
            column: p.column,
            message: p.message.clone(),
        },
        other => Diagnostic {
            severity: Severity::Error,
            line: 0,
            column: 0,
            message: other.to_string(),
        },
    }
}

fn prism_diagnostic(e: &PrismError) -> Diagnostic {
    match e {
        PrismError::Parse(p) => Diagnostic {
            severity: Severity::Error,
            line: p.line,
            column: p.column,
            message: p.message.clone(),
        },
        other => Diagnostic {
            severity: Severity::Error,
            line: 0,
            column: 0,
            message: other.to_string(),
        },
    }
}

/// Checks a model without deriving it: syntax errors, validation errors
/// and warnings. An empty list means the model is fine.
pub fn check(source: &str) -> Vec<Diagnostic> {
    match parse_pepa(source) {
        Err(e) => vec![diagnostic(&e)],
        Ok(model) => model
            .warnings()
            .into_iter()
            .map(|message| Diagnostic {
                severity: Severity::Warning,
                line: 0,
                column: 0,
                message,
            })
            .collect(),
    }
}

/// Options shared by the analysis functions.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Stop with an error when the state space grows beyond this.
    pub max_states: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { max_states: 20_000 }
    }
}

/// A state of a derived (or lumped) state space.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StateView {
    pub id: usize,
    pub name: String,
    pub initial: bool,
    /// Block of the Markovian-bisimulation partition this state belongs to
    /// (for the lumped graph: its own id).
    pub block: usize,
    /// For the lumped graph: how many original states it stands for.
    pub size: usize,
}

/// A transition with its total rate in s⁻¹.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TransitionView {
    pub source: usize,
    pub target: usize,
    pub action: String,
    pub rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphView {
    pub states: Vec<StateView>,
    pub transitions: Vec<TransitionView>,
}

/// Local derivative graph of one sequential component.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComponentView {
    pub component: String,
    pub states: Vec<String>,
    pub transitions: Vec<ComponentTransitionView>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComponentTransitionView {
    pub source: usize,
    pub target: usize,
    pub action: String,
    /// Rate in s⁻¹, or the weight of a passive rate (`0` for functional rates).
    pub rate: f64,
    pub passive: bool,
    /// The functional rate expression (e.g. `0.3 * #I`) when the rate
    /// depends on species counts; `None` for constant and passive rates.
    pub functional: Option<String>,
}

/// Everything the visualiser shows for one model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Analysis {
    pub warnings: Vec<Diagnostic>,
    /// Components of the system equation, left to right.
    pub leaves: Vec<String>,
    pub components: Vec<ComponentView>,
    /// The full derived state space.
    pub full: GraphView,
    /// The quotient by Markovian bisimulation (ordinary lumpability).
    pub lumped: GraphView,
    /// Steady-state probability of every full state (by power iteration).
    pub steady_state: Vec<f64>,
    /// Set when the full state space is too large to list; `full`, `lumped`
    /// and `steady_state` are then empty, but behaviour over time and
    /// observations still work for independent components.
    pub note: Option<String>,
}

struct Derived {
    model: DerivedModel<Sym>,
    partition: Partition,
}

fn derive(source: &str, options: Options) -> Result<Derived, PepaError> {
    let model = parse_pepa(source)?;
    let derived = model.derive_with(DeriveOptions {
        max_states: options.max_states,
    })?;
    let partition = refine_fast(&derived.lts, None);
    Ok(Derived {
        model: derived,
        partition,
    })
}

fn graph(
    lts: &Wlts<Act<Sym>, Real>,
    name: impl Fn(StateId) -> String,
    block: impl Fn(StateId) -> usize,
    size: impl Fn(StateId) -> usize,
) -> GraphView {
    GraphView {
        states: lts
            .states()
            .map(|s| StateView {
                id: s.index(),
                name: name(s),
                initial: lts.initial() == Some(s),
                block: block(s),
                size: size(s),
            })
            .collect(),
        transitions: lts
            .transitions()
            .map(|(s, a, t, w)| TransitionView {
                source: s.index(),
                target: t.index(),
                action: a.to_string(),
                rate: w.value(),
            })
            .collect(),
    }
}

/// Parses, derives and lumps a model, and computes its steady state.
pub fn analyse(source: &str, options: Options) -> Result<Analysis, Vec<Diagnostic>> {
    let warnings = check(source);
    let pepa = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
    let components = pepa
        .component_graphs()
        .map_err(|e| vec![diagnostic(&e)])?
        .into_iter()
        .map(|g| ComponentView {
            component: g.component,
            states: g.states,
            transitions: g
                .transitions
                .into_iter()
                .map(|t| ComponentTransitionView {
                    source: t.source,
                    target: t.target,
                    action: t.action,
                    rate: match t.rate {
                        Rate::Active(r) | Rate::Passive(r) => r,
                        Rate::Functional(_) => 0.0,
                    },
                    passive: t.rate.is_passive(),
                    functional: match &t.rate {
                        Rate::Functional(f) => Some(f.to_string()),
                        _ => None,
                    },
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let leaves: Vec<String> = pepa.leaf_info().into_iter().map(|l| l.name).collect();

    let Derived { model, partition } = match derive(source, options) {
        Ok(d) => d,
        Err(PepaError::StateSpaceTooLarge { limit }) => {
            let empty = || GraphView {
                states: Vec::new(),
                transitions: Vec::new(),
            };
            return Ok(Analysis {
                warnings,
                leaves,
                components,
                full: empty(),
                lumped: empty(),
                steady_state: Vec::new(),
                note: Some(format!(
                    "The full state space has more than {} states, too many to list. \
                     Behaviour over time and observations still work for components \
                     that evolve independently.",
                    limit
                )),
            });
        }
        Err(e) => return Err(vec![diagnostic(&e)]),
    };
    let lts = &model.lts;
    let full = graph(
        lts,
        |s| model.state_name(s),
        |s| partition.block_of(s) as usize,
        |_| 1,
    );
    let blocks = partition.blocks();
    let quotient = lts.quotient(&partition);
    let lumped = graph(
        &quotient,
        |b| {
            let members = &blocks[b.index()];
            let rep = model.state_name(members[0]);
            match members.len() {
                1 => rep,
                n => format!("{rep} (+{} equivalent)", n - 1),
            }
        },
        |b| b.index(),
        |b| blocks[b.index()].len(),
    );
    let steady_state = Ctmc::from_wlts(lts).steady_state(1e-12, 200_000);
    Ok(Analysis {
        warnings,
        leaves,
        components,
        full,
        lumped,
        steady_state,
        note: None,
    })
}

/// Checks a PRISM model without deriving it: syntax and resolution
/// errors. An empty list means the model is fine.
pub fn check_prism(source: &str) -> Vec<Diagnostic> {
    match parse_prism(source) {
        Ok(_) => Vec::new(),
        Err(e) => vec![prism_diagnostic(&e)],
    }
}

/// Checks a JANI model without deriving it. An empty list means it is valid.
pub fn check_jani(source: &str) -> Vec<Diagnostic> {
    match bisim_algebra::jani::parse_jani(source) {
        Ok(_) => Vec::new(),
        Err(e) => vec![prism_diagnostic(&e)],
    }
}

/// Everything the visualiser shows for one PRISM model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PrismAnalysis {
    /// Variable names in state-vector order.
    pub vars: Vec<String>,
    /// The full derived state space.
    pub full: GraphView,
    /// The quotient by Markovian/probabilistic bisimulation.
    pub lumped: GraphView,
    /// Steady-state probability of every full state (by power iteration).
    pub steady_state: Vec<f64>,
    /// Expected rewards per block: state rewards per time unit (per step
    /// for DTMCs) and transition rewards per time unit, in steady state.
    pub rewards: Vec<RewardResult>,
    /// Set when the full state space is too large to list.
    pub note: Option<String>,
}

/// Expected rewards of one `rewards` block in steady state.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RewardResult {
    pub block: String,
    /// Expected state reward: Σ π(s)·value(s) over matching states.
    pub state: f64,
    /// Expected transition reward: Σ π(s)·rate(s→)·value(s) over
    /// matching transitions (probabilities for DTMCs).
    pub transition: f64,
}

/// Parses, derives and lumps a PRISM (`dtmc`/`ctmc` subset) model, and
/// computes its steady state. Lumpability and analysis work exactly as
/// for PEPA models, since both derive the same kind of labelled system.
/// The steady state is valid for DTMC and CTMC alike (uniformisation
/// preserves the stationary distribution); per-time transient analysis
/// for DTMCs is not offered (their clock is steps, not seconds).
pub fn analyse_prism(source: &str, options: Options) -> Result<PrismAnalysis, Vec<Diagnostic>> {
    let model = parse_prism(source).map_err(|e| vec![prism_diagnostic(&e)])?;
    analyse_prism_model(&model, options).map_err(|e| vec![prism_diagnostic(&e)])
}

/// Analyses a JANI model (see `bisim_algebra::jani` for the subset) exactly
/// like [`analyse_prism`]: the interchange format of Storm and Modest.
pub fn analyse_jani(source: &str, options: Options) -> Result<PrismAnalysis, Vec<Diagnostic>> {
    let model = bisim_algebra::jani::parse_jani(source).map_err(|e| vec![prism_diagnostic(&e)])?;
    analyse_prism_model(&model, options).map_err(|e| vec![prism_diagnostic(&e)])
}

fn analyse_prism_model(model: &PrismModel, options: Options) -> Result<PrismAnalysis, PrismError> {
    let vars: Vec<String> = model.vars.iter().map(|v| v.name.clone()).collect();
    let empty = || GraphView {
        states: Vec::new(),
        transitions: Vec::new(),
    };
    let derived = match model.derive_with(DeriveOptions {
        max_states: options.max_states,
    }) {
        Ok(d) => d,
        Err(PrismError::StateSpaceTooLarge { limit }) => {
            return Ok(PrismAnalysis {
                vars,
                full: empty(),
                lumped: empty(),
                steady_state: Vec::new(),
                rewards: Vec::new(),
                note: Some(format!(
                    "The full state space has more than {limit} states, too many to list."
                )),
            });
        }
        Err(e) => return Err(e),
    };
    let lts = &derived.lts;
    let partition = refine_fast(lts, None);
    let full = graph(
        lts,
        |s| derived.state_name(s),
        |s| partition.block_of(s) as usize,
        |_| 1,
    );
    let blocks = partition.blocks();
    let quotient = lts.quotient(&partition);
    let lumped = graph(
        &quotient,
        |b| {
            let members = &blocks[b.index()];
            let rep = derived.state_name(members[0]);
            match members.len() {
                1 => rep,
                n => format!("{rep} (+{} equivalent)", n - 1),
            }
        },
        |b| b.index(),
        |b| blocks[b.index()].len(),
    );
    let steady_state = Ctmc::from_wlts(lts).steady_state(1e-12, 200_000);
    let rewards = eval_rewards(model, &derived, &steady_state)?;
    Ok(PrismAnalysis {
        vars,
        full,
        lumped,
        steady_state,
        rewards,
        note: None,
    })
}

/// Expected rewards of every `rewards` block in steady state.
fn eval_rewards(
    model: &PrismModel,
    derived: &PrismDerived,
    steady: &[f64],
) -> Result<Vec<RewardResult>, PrismError> {
    use bisim_algebra::prism::RewardItem;

    model
        .rewards
        .iter()
        .map(|b| {
            let (mut state, mut transition) = (0.0, 0.0);
            for (i, valuation) in derived.states.iter().enumerate() {
                let pi = steady[i];
                if pi == 0.0 {
                    continue;
                }
                for item in &b.items {
                    match item {
                        RewardItem::State { guard, value } => {
                            if model.eval_pred(guard, valuation)? {
                                let v = model.eval_num(value, valuation)?;
                                if !v.is_finite() {
                                    return Err(PrismError::Arithmetic(format!(
                                        "state reward of `{}` is not finite",
                                        b.name
                                    )));
                                }
                                state += pi * v;
                            }
                        }
                        RewardItem::Trans {
                            label,
                            guard,
                            value,
                        } => {
                            if !model.eval_pred(guard, valuation)? {
                                continue;
                            }
                            let v = model.eval_num(value, valuation)?;
                            if !v.is_finite() {
                                return Err(PrismError::Arithmetic(format!(
                                    "transition reward of `{}` is not finite",
                                    b.name
                                )));
                            }
                            for e in derived.lts.successors(StateId(i as u32)) {
                                if matches!(derived.lts.label(e.label), Act::Act(l) if l == label) {
                                    transition += pi * e.weight.value() * v;
                                }
                            }
                        }
                    }
                }
            }
            Ok(RewardResult {
                block: b.name.clone(),
                state,
                transition,
            })
        })
        .collect()
}

/// The answer to one CSL/PCTL-style query (see [`query_prism`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryResult {
    /// Probability from the initial state (`P`) or in the long run (`S`).
    pub probability: f64,
    /// Whether the threshold holds (`None` for quantitative `=?` queries).
    pub holds: Option<bool>,
    pub explanation: String,
}

/// Answers a query (`P=? [F pred]`, `P>=p [F<=t pred]`, `S=? [pred]`, …)
/// on a PRISM model, from its initial state.
///
/// - `F` on DTMCs is solved by value iteration (unbounded) or exact
///   step-bounded iteration (`F<=k`, integral `k`); on CTMCs, unbounded
///   `F` runs on the embedded jump chain and `F<=t` by uniformisation
///   with absorbing targets.
/// - `S` sums the steady state over `pred` (valid for DTMC and CTMC).
pub fn query_prism(
    source: &str,
    query: &str,
    options: Options,
) -> Result<QueryResult, Vec<Diagnostic>> {
    let model = parse_prism(source).map_err(|e| vec![prism_diagnostic(&e)])?;
    query_model(&model, query, options)
}

/// Answers a query on a JANI model (see [`query_prism`]).
pub fn query_jani(
    source: &str,
    query: &str,
    options: Options,
) -> Result<QueryResult, Vec<Diagnostic>> {
    let model = bisim_algebra::jani::parse_jani(source).map_err(|e| vec![prism_diagnostic(&e)])?;
    query_model(&model, query, options)
}

fn query_model(
    model: &PrismModel,
    query: &str,
    options: Options,
) -> Result<QueryResult, Vec<Diagnostic>> {
    let query = parse_query(model, query).map_err(|e| vec![prism_diagnostic(&e)])?;
    let derived = model
        .derive_with(DeriveOptions {
            max_states: options.max_states,
        })
        .map_err(|e| vec![prism_diagnostic(&e)])?;
    let lts = &derived.lts;
    let targets: Vec<bool> = lts
        .states()
        .map(|s| {
            model
                .eval_pred(
                    match &query {
                        Query::Prob(q) => &q.pred,
                        Query::Steady(q) => &q.pred,
                    },
                    &derived.states[s.index()],
                )
                .map_err(|e| vec![prism_diagnostic(&e)])
        })
        .collect::<Result<_, _>>()?;
    let init = lts.initial().expect("derived models have an initial state");
    let ctmc = Ctmc::from_wlts(lts);
    let probability = match &query {
        Query::Steady(_) => ctmc
            .steady_state(1e-12, 200_000)
            .into_iter()
            .enumerate()
            .map(|(i, p)| if targets[i] { p } else { 0.0 })
            .sum(),
        Query::Prob(q) => match (&model.kind, &q.until) {
            (ModelKind::Dtmc, Until::Eventually) => {
                let p = dtmc_matrix(lts).map_err(|e| {
                    vec![Diagnostic {
                        severity: Severity::Error,
                        line: 0,
                        column: 0,
                        message: e.to_string(),
                    }]
                })?;
                unbounded_reach(&p, &targets, 1e-12, 1_000_000)[init.index()]
            }
            (ModelKind::Dtmc, Until::Bounded(k)) => {
                if k.fract() != 0.0 || *k < 0.0 || *k > MAX_STEP_BOUND {
                    return Err(vec![Diagnostic {
                        severity: Severity::Error,
                        line: 0,
                        column: 0,
                        message: format!(
                            "DTMC step bounds must be whole steps up to {MAX_STEP_BOUND}, got {k}"
                        ),
                    }]);
                }
                let p = dtmc_matrix(lts).map_err(|e| {
                    vec![Diagnostic {
                        severity: Severity::Error,
                        line: 0,
                        column: 0,
                        message: e.to_string(),
                    }]
                })?;
                bounded_steps(&p, &targets, *k as u64)[init.index()]
            }
            (_, Until::Eventually) => {
                let p = ctmc.embedded();
                unbounded_reach(&p, &targets, 1e-12, 1_000_000)[init.index()]
            }
            (_, Until::Bounded(t)) => {
                if !t.is_finite() || *t < 0.0 {
                    return Err(vec![Diagnostic {
                        severity: Severity::Error,
                        line: 0,
                        column: 0,
                        message: format!("time bounds must be finite and non-negative, got {t}"),
                    }]);
                }
                ctmc.reach_bounded(&ctmc.initial_distribution(), &targets, *t, 1e-10)
            }
        },
    };
    let (threshold, what) = match &query {
        Query::Prob(q) => (q.threshold, "reachability probability"),
        Query::Steady(q) => (q.threshold, "steady-state probability"),
    };
    let holds = threshold.map(|(cmp, bound)| cmp.holds(probability, bound));
    let explanation = match threshold {
        None => format!("The {what} from the initial state is {probability:.6}."),
        Some((cmp, bound)) => {
            let verdict = if holds.expect("threshold") {
                "holds"
            } else {
                "fails"
            };
            format!(
                "The {what} from the initial state is {probability:.6}, {} {} {bound} {verdict}.",
                match &query {
                    Query::Prob(_) => "P",
                    Query::Steady(_) => "S",
                },
                cmp.symbol()
            )
        }
    };
    Ok(QueryResult {
        probability,
        holds,
        explanation,
    })
}

/// Mean-field fluid approximation over time, by RK4 integration of the
/// SOS drift (see `DerivedModel::fluid_drift`).
///
/// Returns the same [`Transient`] shape as [`transient`], but computed
/// from expected-count ODEs instead of the Markov chain: cheap for large
/// populations, where enumeration is infeasible. Only array populations
/// (`P[n]`) are supported; single components are an error, as are
/// non-positive or non-finite `dt` values.
///
/// `options` is accepted for symmetry with the other analyses; the state
/// space is never enumerated, so its limits do not apply here.
pub fn fluid(
    source: &str,
    times: &[f64],
    dt: f64,
    _options: Options,
) -> Result<Transient, Vec<Diagnostic>> {
    if let Some(t) = times.iter().find(|t| !t.is_finite() || **t < 0.0) {
        return Err(vec![Diagnostic {
            severity: Severity::Error,
            line: 0,
            column: 0,
            message: format!("times must be non-negative, got {t}"),
        }]);
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(vec![Diagnostic {
            severity: Severity::Error,
            line: 0,
            column: 0,
            message: format!("dt must be positive and finite, got {dt}"),
        }]);
    }
    let horizon = times.iter().copied().fold(0.0, f64::max);
    if horizon / dt > MAX_FLUID_STEPS {
        return Err(vec![Diagnostic {
            severity: Severity::Error,
            line: 0,
            column: 0,
            message: format!(
                "dt = {dt} would take more than {MAX_FLUID_STEPS} steps to reach {horizon}; use a larger dt"
            ),
        }]);
    }
    let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
    let fluid = model.fluid_model().map_err(|e| vec![diagnostic(&e)])?;
    let slots = fluid.fluid_slots();
    let snaps =
        bisim_stochastic::integrate(|x| fluid.fluid_drift(x), &fluid.initial_counts(), times, dt)
            .map_err(|e| vec![diagnostic(&e)])?;
    let mut by_series: Vec<Vec<f64>> = vec![Vec::new(); slots.len()];
    for snap in &snaps {
        for (values, x) in by_series.iter_mut().zip(snap) {
            values.push(*x);
        }
    }
    Ok(Transient {
        times: times.to_vec(),
        occupancy: slots
            .into_iter()
            .zip(by_series)
            .map(|((component, derivative), values)| Series {
                component,
                derivative,
                values,
            })
            .collect(),
    })
}

/// One time series of a transient table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Series {
    /// Component of the system equation (e.g. `Healthy[3]`).
    pub component: String,
    /// Local derivative (e.g. `Sick`).
    pub derivative: String,
    /// Expected number of copies of `component` in `derivative`, per time.
    pub values: Vec<f64>,
}

/// Behaviour over time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Transient {
    /// Times, in the model's time unit.
    pub times: Vec<f64>,
    /// Expected occupancy of every local derivative of every component.
    pub occupancy: Vec<Series>,
}

/// Expected occupancy over time, starting from the initial state.
///
/// Components that evolve independently are computed from one copy's chain,
/// so large populations stay cheap; others use the full Markov chain.
pub fn transient(
    source: &str,
    times: &[f64],
    options: Options,
) -> Result<Transient, Vec<Diagnostic>> {
    if let Some(t) = times.iter().find(|t| !t.is_finite() || **t < 0.0) {
        return Err(vec![Diagnostic {
            severity: Severity::Error,
            line: 0,
            column: 0,
            message: format!("times must be non-negative, got {t}"),
        }]);
    }
    let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
    let eval = engine::Evaluator::new(&model, options)?;
    let snaps = eval.snapshots(times);
    Ok(Transient {
        times: times.to_vec(),
        occupancy: eval.occupancy(&snaps, times.len()),
    })
}

/// The outcome of one check, with a readable explanation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerdictView {
    pub holds: bool,
    /// `false` when the check was skipped (`holds` is then meaningless; see
    /// `explanation`).
    pub checked: bool,
    /// A formula true in A's initial state and false in B's (or the
    /// reverse when `formula_holds_in` says so).
    pub formula: Option<String>,
    pub formula_holds_in: Option<String>,
    /// States of A and B that satisfy the formula (for highlighting).
    pub satisfied_in_a: Vec<usize>,
    pub satisfied_in_b: Vec<usize>,
    pub explanation: String,
}

/// Comparison of two models.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    /// Same behaviour including rates (Markovian bisimilarity).
    pub markovian_bisimilar: VerdictView,
    /// Ignoring rates: can B do everything A can (step by step)?
    pub b_simulates_a: VerdictView,
    /// Ignoring rates: each can do everything the other can.
    pub mutually_similar: VerdictView,
    /// Ignoring rates: same branching behaviour.
    pub bisimilar: VerdictView,
    /// Set when the models were too large and one copy of each component
    /// was compared instead.
    pub note: Option<String>,
}

/// Errors in either model of a comparison.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompareError {
    pub a: Vec<Diagnostic>,
    pub b: Vec<Diagnostic>,
}

/// Most bisimulation classes (both models together) for which simulation is
/// checked: the preorder is a table with one entry per pair of classes.
const SIMULATION_LIMIT: usize = 2_000;

fn skipped(what: &str, states: usize) -> VerdictView {
    VerdictView {
        holds: false,
        checked: false,
        formula: None,
        formula_holds_in: None,
        satisfied_in_a: Vec::new(),
        satisfied_in_b: Vec::new(),
        explanation: format!(
            "Not checked: {what} needs a table of every pair of states, and the two models \
             have {states} distinct behaviours together (the limit is {SIMULATION_LIMIT})."
        ),
    }
}

const READING_GUIDE: &str = "Read ⟨a⟩φ as \"can do a, then satisfy φ\", ⟨a⟩=r φ as \
     \"does a at total rate r per second into states satisfying φ\", ¬ as \"not\" and tt as \"true\".";

fn verdict_view<L, W>(
    verdict: &Verdict<Formula<L, W::Key>>,
    a: &Wlts<L, W>,
    b: &Wlts<L, W>,
    holds_in_b_side: bool,
    what: &str,
) -> VerdictView
where
    L: Clone + Eq + std::hash::Hash + std::fmt::Display,
    W: Semiring,
{
    match verdict {
        Verdict::Violated { witness } => {
            let (yes, no) = if holds_in_b_side {
                ("B", "A")
            } else {
                ("A", "B")
            };
            let sat = |sys: &Wlts<L, W>| {
                witness
                    .satisfying(sys)
                    .into_iter()
                    .map(StateId::index)
                    .collect()
            };
            VerdictView {
                holds: false,
                checked: true,
                formula: Some(witness.to_string()),
                formula_holds_in: Some(yes.to_owned()),
                satisfied_in_a: sat(a),
                satisfied_in_b: sat(b),
                explanation: format!(
                    "Not {what}: the initial state of {yes} satisfies {witness}, the initial state of {no} does not. {READING_GUIDE}"
                ),
            }
        }
        _ => VerdictView {
            holds: true,
            checked: true,
            formula: None,
            formula_holds_in: None,
            satisfied_in_a: Vec::new(),
            satisfied_in_b: Vec::new(),
            explanation: format!("The models are {what}."),
        },
    }
}

/// Compares the initial states of two models.
pub fn compare(
    source_a: &str,
    source_b: &str,
    options: Options,
) -> Result<Comparison, CompareError> {
    let derive_lts = |source: &str, single: bool| -> Result<Wlts<Act<Sym>, Real>, PepaError> {
        let model = parse_pepa(source)?;
        let model = if single { model.single_copies() } else { model };
        Ok(model
            .derive_with(DeriveOptions {
                max_states: options.max_states,
            })?
            .lts)
    };
    let too_large =
        |r: &Result<_, PepaError>| matches!(r, Err(PepaError::StateSpaceTooLarge { .. }));
    let (mut a, mut b) = (derive_lts(source_a, false), derive_lts(source_b, false));
    let mut note = None;
    if too_large(&a) || too_large(&b) {
        (a, b) = (derive_lts(source_a, true), derive_lts(source_b, true));
        note = Some(format!(
            "The full models have more than {} states, so one copy of each component was \
             compared. This is exact for components that evolve independently and have the \
             same number of copies in both models.",
            options.max_states
        ));
    }
    let (a, b) = match (a, b) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            return Err(CompareError {
                a: a.err().map(|e| vec![diagnostic(&e)]).unwrap_or_default(),
                b: b.err().map(|e| vec![diagnostic(&e)]).unwrap_or_default(),
            });
        }
    };

    // Markovian bisimilarity of the initial states.
    let markovian = bisim_refine::bisimilar(&a, &b);

    // Untimed comparisons forget rates.
    let untimed =
        |sys: &Wlts<Act<Sym>, Real>| -> Lts<Act<Sym>> { sys.map_weights(|w| !w.is_zero()) };
    let (ua, ub) = (untimed(&a), untimed(&b));
    // Simulation is tabled per bisimulation class, so count those.
    let states = refine_fast(&ua.disjoint_union(&ub), None).num_blocks();
    if states > SIMULATION_LIMIT {
        // Bisimilarity scales (worklist refinement); simulation does not.
        let bisimilar = bisim_refine::bisimilar(&ua, &ub);
        return Ok(Comparison {
            markovian_bisimilar: verdict_view(
                &markovian,
                &a,
                &b,
                false,
                "Markovian bisimilar (same behaviour and rates)",
            ),
            b_simulates_a: skipped("simulation", states),
            mutually_similar: skipped("mutual similarity", states),
            bisimilar: verdict_view(
                &bisimilar,
                &ua,
                &ub,
                false,
                "bisimilar when rates are ignored",
            ),
            note,
        });
    }
    let v = compare_lts(&ua, &ub);
    let reverse = matches!(
        (&v.right_simulates_left, &v.mutually_similar),
        (Verdict::Holds, Verdict::Violated { .. })
    );
    Ok(Comparison {
        markovian_bisimilar: verdict_view(
            &markovian,
            &a,
            &b,
            false,
            "Markovian bisimilar (same behaviour and rates)",
        ),
        b_simulates_a: verdict_view(
            &v.right_simulates_left,
            &ua,
            &ub,
            false,
            "in a simulation relation (B cannot match every step of A)",
        ),
        mutually_similar: verdict_view(&v.mutually_similar, &ua, &ub, reverse, "mutually similar"),
        bisimilar: verdict_view(
            &v.bisimilar,
            &ua,
            &ub,
            false,
            "bisimilar when rates are ignored",
        ),
        note,
    })
}
