//! Stateful sessions for the JSON protocol: open one, push observations as
//! they arrive, and get the updated answer each time without resending or
//! recomputing what came before.
//!
//! - `prefix`: two systems observed transition by transition
//!   ([`PrefixChecker`]: incremental games, final violations);
//! - `observe`: dated counts against a plan ([`ObserveSession`]: each
//!   count advances from the previous date);
//! - `stream_distance`: the sample-based distance ([`SampleDistance`]: one
//!   step per observed pair of transitions).
//!
//! Sessions live in this process until closed. Their number and the size of
//! each push are capped; going over is an error, nothing is evicted
//! silently.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use bisim_core::{ComparisonVerdict, Formula, Verdict};
use bisim_metric::SampleDistance;
use bisim_refine::{GameBudget, PrefixChecker};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::observe::{Observation, ObserveSession, Timeline, error};
use crate::{Diagnostic, Options};

/// Most sessions open at once.
pub const MAX_SESSIONS: usize = 64;
/// Most events in one push.
pub const MAX_EVENTS_PER_PUSH: usize = 100_000;
/// Largest game a prefix session may build.
pub const MAX_SESSION_PAIRS: usize = 1_000_000;

/// What to open.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum OpenSpec {
    Prefix {
        left: String,
        right: String,
        max_pairs: Option<usize>,
    },
    Observe {
        source: String,
        timeline: Timeline,
        max_states: Option<usize>,
    },
    StreamDistance {
        /// The states each log can be in, with their observation values.
        states_a: Vec<String>,
        states_b: Vec<String>,
        values: HashMap<String, f64>,
        gamma: f64,
    },
}

/// One prefix event: `{"left": [s, a, t]}`, `{"complete_left": s}`, ...
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum PrefixEvent {
    Left(String, String, String),
    Right(String, String, String),
    CompleteLeft(String),
    CompleteRight(String),
}

/// One step of both logs: a transition `[from, to]` of each.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamStep {
    a: (String, String),
    b: (String, String),
}

enum Session {
    Prefix(Box<PrefixChecker<String, String>>),
    Observe(Box<ObserveSession>),
    Stream {
        estimator: Box<SampleDistance>,
        states_a: Vec<String>,
        states_b: Vec<String>,
    },
}

struct Registry {
    next: u64,
    open: HashMap<u64, Session>,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(Registry {
            next: 1,
            open: HashMap::new(),
        })
    })
}

fn with_registry<T>(f: impl FnOnce(&mut Registry) -> T) -> T {
    // A panic while holding the lock leaves a usable registry behind.
    let mut guard = registry().lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

/// Opens a session; returns its id.
pub(crate) fn open(
    spec: OpenSpec,
    options: impl Fn(Option<usize>) -> Options,
) -> Result<Value, Vec<Diagnostic>> {
    let session = match spec {
        OpenSpec::Prefix {
            left,
            right,
            max_pairs,
        } => {
            let budget = GameBudget {
                max_pairs: max_pairs.unwrap_or(100_000).min(MAX_SESSION_PAIRS),
                max_depth: u32::MAX,
            };
            Session::Prefix(Box::new(
                PrefixChecker::new(left, right).with_budget(budget),
            ))
        }
        OpenSpec::Observe {
            source,
            timeline,
            max_states,
        } => Session::Observe(Box::new(ObserveSession::new(
            &source,
            &timeline,
            options(max_states),
        )?)),
        OpenSpec::StreamDistance {
            states_a,
            states_b,
            values,
            gamma,
        } => {
            let value = |s: &String| {
                values
                    .get(s)
                    .copied()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| error(format!("no finite value for state `{s}`")))
            };
            let va: Vec<f64> = states_a.iter().map(value).collect::<Result<_, _>>()?;
            let vb: Vec<f64> = states_b.iter().map(value).collect::<Result<_, _>>()?;
            let lo = va.iter().chain(&vb).copied().fold(f64::INFINITY, f64::min);
            let hi = va
                .iter()
                .chain(&vb)
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let scale = if hi > lo { hi - lo } else { 1.0 };
            let estimator = SampleDistance::new(
                states_a.len(),
                states_b.len(),
                |x, y| (va[x] - vb[y]).abs() / scale,
                gamma,
                (0, 0),
            )
            .map_err(|e| error(e.to_string()))?;
            Session::Stream {
                estimator: Box::new(estimator),
                states_a,
                states_b,
            }
        }
    };
    with_registry(|r| {
        if r.open.len() >= MAX_SESSIONS {
            return Err(error(format!(
                "{MAX_SESSIONS} sessions are open; close one first"
            )));
        }
        let id = r.next;
        r.next += 1;
        r.open.insert(id, session);
        Ok(json!({ "id": id }))
    })
}

/// Pushes events into session `id`; returns its updated answer.
pub(crate) fn push(id: u64, events: Value) -> Result<Value, Vec<Diagnostic>> {
    let count = events.as_array().map_or(0, Vec::len);
    if count > MAX_EVENTS_PER_PUSH {
        return Err(error(format!(
            "{count} events in one push; at most {MAX_EVENTS_PER_PUSH}"
        )));
    }
    with_registry(|r| {
        let session = r
            .open
            .get_mut(&id)
            .ok_or_else(|| error(format!("no open session {id}")))?;
        let parse = |what: &str, e: serde_json::Error| error(format!("invalid {what}: {e}"));
        match session {
            Session::Prefix(checker) => {
                let events: Vec<PrefixEvent> =
                    serde_json::from_value(events).map_err(|e| parse("prefix events", e))?;
                for (i, event) in events.into_iter().enumerate() {
                    let applied = match event {
                        PrefixEvent::Left(s, a, t) => checker.observe_left([(s, a, t)]),
                        PrefixEvent::Right(s, a, t) => checker.observe_right([(s, a, t)]),
                        PrefixEvent::CompleteLeft(s) => {
                            checker.complete_left(s);
                            Ok(())
                        }
                        PrefixEvent::CompleteRight(s) => {
                            checker.complete_right(s);
                            Ok(())
                        }
                    };
                    if let Err(e) = applied {
                        return Err(error(format!(
                            "event {}: {e} (the events before it were applied)",
                            i + 1
                        )));
                    }
                }
                let changed = checker.poll_change().is_some();
                Ok(json!({ "changed": changed, "verdict": prefix_view(&checker.verdict()) }))
            }
            Session::Observe(s) => {
                let obs: Vec<Observation> =
                    serde_json::from_value(events).map_err(|e| parse("observations", e))?;
                serde_json::to_value(s.push(&obs)?).map_err(|e| error(e.to_string()))
            }
            Session::Stream {
                estimator,
                states_a,
                states_b,
            } => {
                let steps: Vec<StreamStep> =
                    serde_json::from_value(events).map_err(|e| parse("stream steps", e))?;
                let index = |states: &[String], s: &str| {
                    states
                        .iter()
                        .position(|x| x == s)
                        .ok_or_else(|| error(format!("unknown state `{s}`")))
                };
                // Resolve everything first, so a bad step adds nothing.
                let resolved = steps
                    .iter()
                    .map(|st| {
                        Ok((
                            (index(states_a, &st.a.0)?, index(states_a, &st.a.1)?),
                            (index(states_b, &st.b.0)?, index(states_b, &st.b.1)?),
                        ))
                    })
                    .collect::<Result<Vec<_>, Vec<Diagnostic>>>()?;
                for (a, b) in resolved {
                    estimator.push(a, b);
                }
                let matching: Vec<(String, String)> = estimator
                    .matching()
                    .into_iter()
                    .enumerate()
                    .map(|(x, y)| (states_a[x].clone(), states_b[y].clone()))
                    .collect();
                Ok(json!({
                    "estimate": estimator.estimate(),
                    "steps": estimator.steps(),
                    "matching": matching,
                }))
            }
        }
    })
}

/// Closes session `id`.
pub(crate) fn close(id: u64) -> Result<Value, Vec<Diagnostic>> {
    with_registry(|r| match r.open.remove(&id) {
        Some(_) => Ok(json!({ "closed": id })),
        None => Err(error(format!("no open session {id}"))),
    })
}

/// A verdict for JSON: its kind and, for a violation, the witness.
#[derive(Debug, Serialize)]
struct VerdictKind {
    verdict: &'static str,
    witness: Option<String>,
}

fn kind(v: &Verdict<Formula<String, bool>>) -> VerdictKind {
    let (verdict, witness) = match v {
        Verdict::Holds => ("holds", None),
        Verdict::HoldsSoFar => ("holds_so_far", None),
        Verdict::Violated { witness } => ("violated", Some(witness.to_string())),
        Verdict::Unknown => ("unknown", None),
        _ => ("other", None),
    };
    VerdictKind { verdict, witness }
}

fn prefix_view(v: &ComparisonVerdict<Formula<String, bool>>) -> Value {
    json!({
        "right_simulates_left": kind(&v.right_simulates_left),
        "mutually_similar": kind(&v.mutually_similar),
        "bisimilar": kind(&v.bisimilar),
    })
}
