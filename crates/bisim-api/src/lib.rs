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
use bisim_core::{Formula, Lts, Partition, Real, Semiring, StateId, Verdict, Wlts};
use bisim_refine::{compare as compare_lts, distinguishing_formula, refine};
use bisim_stochastic::Ctmc;
use serde::Serialize;

mod engine;
mod observe;

pub use observe::{
    FitReport, FittedParam, Forecast, Observation, ObservationReport, ObservationResult, Status,
    TimeUnit, Timeline, check_observations, fit_parameters, forecast, format_date, parse_date,
};

#[cfg(feature = "json")]
mod json;
#[cfg(feature = "json")]
pub use json::handle_json;

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
    /// Rate in s⁻¹, or the weight of a passive rate.
    pub rate: f64,
    pub passive: bool,
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
    let partition = refine(&derived.lts, None).partition().clone();
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
                    },
                    passive: t.rate.is_passive(),
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
                sys.states()
                    .filter(|s| witness.holds(sys, *s))
                    .map(StateId::index)
                    .collect()
            };
            VerdictView {
                holds: false,
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

    // Markovian bisimilarity on the disjoint union.
    let union = a.disjoint_union(&b);
    let t = StateId(
        b.initial().expect("derived models have an initial state").0 + a.num_states() as u32,
    );
    let s = a.initial().expect("derived models have an initial state");
    let r = refine(&union, None);
    let markovian = match distinguishing_formula(&union, &r, s, t) {
        None => Verdict::Holds,
        Some(witness) => Verdict::Violated { witness },
    };

    // Untimed comparisons forget rates.
    let untimed =
        |sys: &Wlts<Act<Sym>, Real>| -> Lts<Act<Sym>> { sys.map_weights(|w| !w.is_zero()) };
    let (ua, ub) = (untimed(&a), untimed(&b));
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
