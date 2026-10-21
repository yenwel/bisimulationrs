//! Behavioural distances between PEPA models and between logged data, for
//! the bindings: how far apart two models are (`distance`), whether they
//! are within a tolerance with a guaranteed bound on outcomes (`epsdelta`),
//! a distance estimated from two logged state sequences
//! (`stream_distance`), and a comparison of continuous sensor series
//! (`dsa`). The methods and their sources are in `bisim-metric`.

use std::collections::HashMap;

use bisim_algebra::pepa::{DeriveOptions, DerivedModel, Sym, parse_pepa};
use bisim_core::StateId;
use bisim_metric::{
    DsaOptions, MetricError, MetricOptions, SampleDistance, bisim_distance, common_rate,
    dsa as dsa_metric, epsdelta as epsdelta_relation, min_epsilon, rate_only_bound, reach_bound,
};
use serde::{Deserialize, Serialize};

use crate::observe::error;
use crate::{Diagnostic, Options, diagnostic};

/// A local state of a component, counted in every global state: `Plant`
/// in `Fruiting` counts the copies of `Plant` whose state is `Fruiting` (or
/// `Fruiting_<n>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Atom {
    /// The component; may be left out when the model has one.
    pub component: Option<String>,
    pub state: String,
}

fn metric_error(e: MetricError) -> Vec<Diagnostic> {
    error(e.to_string())
}

/// A derived model and, per atom, the count of that atom in each state.
struct Counted {
    model: DerivedModel<Sym>,
    counts: Vec<Vec<f64>>,
}

fn derive_counted(
    which: &str,
    source: &str,
    atoms: &[Atom],
    options: Options,
) -> Result<Counted, Vec<Diagnostic>> {
    let at = |d: Vec<Diagnostic>| {
        d.into_iter()
            .map(|mut d| {
                d.message = format!("model {which}: {}", d.message);
                d
            })
            .collect::<Vec<_>>()
    };
    let pepa = parse_pepa(source).map_err(|e| at(vec![diagnostic(&e)]))?;
    let leaves = pepa.leaf_info();
    let model = pepa
        .derive_with(DeriveOptions {
            max_states: options.max_states,
        })
        .map_err(|e| at(vec![diagnostic(&e)]))?;
    let mut counts = Vec::new();
    for atom in atoms {
        let leaf = match &atom.component {
            None if leaves.len() == 1 => 0,
            None => {
                let names: Vec<&str> = leaves.iter().map(|l| l.name.as_str()).collect();
                return Err(at(error(format!(
                    "say which component `{}` belongs to: {}",
                    atom.state,
                    names.join(", ")
                ))));
            }
            Some(c) => {
                let hits: Vec<usize> = (0..leaves.len())
                    .filter(|&k| leaves[k].name == *c || leaves[k].component == *c)
                    .collect();
                match hits.as_slice() {
                    [k] => *k,
                    [] => return Err(at(error(format!("no component `{c}`")))),
                    _ => {
                        return Err(at(error(format!(
                            "`{c}` occurs more than once; use its position-specific name"
                        ))));
                    }
                }
            }
        };
        let prefix = format!("{}_", atom.state);
        let mut seen = false;
        let column: Vec<f64> = model
            .lts
            .states()
            .map(|s| {
                model
                    .local_counts(s, leaf)
                    .into_iter()
                    .filter(|(d, _)| *d == atom.state || d.starts_with(&prefix))
                    .map(|(_, n)| {
                        seen = true;
                        f64::from(n)
                    })
                    .sum()
            })
            .collect();
        if !seen {
            return Err(at(error(format!(
                "`{}` never appears in a reachable state of `{}`",
                atom.state, leaves[leaf].name
            ))));
        }
        counts.push(column);
    }
    Ok(Counted { model, counts })
}

/// A state pair contributing to a distance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairView {
    pub a: String,
    pub b: String,
    /// Difference of the (normalised) observation in the pair.
    pub gap: f64,
    /// Discounted share of time the closest coupling spends in the pair.
    pub occupancy: f64,
    pub contribution: f64,
}

/// The bisimulation distance between two models.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DistanceReport {
    /// In `[0, 1]`: 0 when the models cannot be told apart by the
    /// observation, 1 when it always differs maximally.
    pub distance: f64,
    /// The true distance is within this of `distance`.
    pub error_bound: f64,
    pub half_life: f64,
    /// The observation's range across both models (what 1 corresponds to).
    pub scale: f64,
    pub pairs: Vec<PairView>,
    pub explanation: String,
}

/// How differently two models behave, seen through the count of `observe`,
/// with differences `half_life` time units ahead counting half (Calo et
/// al., NeurIPS 2024).
pub fn distance(
    source_a: &str,
    source_b: &str,
    observe: &Atom,
    half_life: f64,
    options: Options,
) -> Result<DistanceReport, Vec<Diagnostic>> {
    if !(half_life > 0.0 && half_life.is_finite()) {
        return Err(error("half_life must be a positive number of time units"));
    }
    let a = derive_counted("a", source_a, std::slice::from_ref(observe), options)?;
    let b = derive_counted("b", source_b, std::slice::from_ref(observe), options)?;
    let (ca, cb) = (&a.counts[0], &b.counts[0]);
    let lo = ca.iter().chain(cb).copied().fold(f64::INFINITY, f64::min);
    let hi = ca
        .iter()
        .chain(cb)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let (sa, sb) = (
        a.model
            .lts
            .initial()
            .ok_or_else(|| error("model a has no initial state"))?,
        b.model
            .lts
            .initial()
            .ok_or_else(|| error("model b has no initial state"))?,
    );
    let d = bisim_distance(
        (&a.model.lts, sa),
        (&b.model.lts, sb),
        |s| ca[s.index()],
        |s| cb[s.index()],
        half_life,
        MetricOptions::default(),
    )
    .map_err(metric_error)?;
    let pairs = d
        .pairs
        .iter()
        .map(|p| PairView {
            a: a.model.state_name(StateId(p.left as u32)),
            b: b.model.state_name(StateId(p.right as u32)),
            gap: p.gap,
            occupancy: p.occupancy,
            contribution: p.contribution,
        })
        .collect();
    let what = match &observe.component {
        Some(c) => format!("{c} in {}", observe.state),
        None => observe.state.clone(),
    };
    let explanation = format!(
        "Running both models side by side in the way that makes them look most alike, the \
         count of {what} differs on average by {:.1}% of its range ({:.3} on a scale of {:.0}), \
         weighting time so that differences {half_life} time units ahead count half. \
         0 means the models cannot be told apart through this count.",
        100.0 * d.value,
        d.value * (hi - lo),
        hi - lo
    );
    Ok(DistanceReport {
        distance: d.value,
        error_bound: d.error_bound,
        half_life,
        scale: hi - lo,
        pairs,
        explanation,
    })
}

/// The (ε, δ)-bisimulation check between two models.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EpsDeltaReport {
    pub delta: f64,
    /// The smallest ε at which the initial states are related, if any.
    pub min_epsilon: Option<f64>,
    /// Whether they are related at the requested ε.
    pub within: Option<bool>,
    /// The uniformisation rate `q` used for the bound.
    pub rate: f64,
    /// Proposition 3's bound on time-bounded reachability differences, at
    /// `horizon`, using the requested ε (or the smallest one).
    pub reach_bound: Option<f64>,
    /// The sharper bound when only rates differ (ε = 0, equal exit rates
    /// assumed at `rate`); for orientation.
    pub rate_only_bound: Option<f64>,
    pub explanation: String,
}

/// Are the models within `epsilon` on jump probabilities and a factor
/// `e^delta` on rates, for state labels given by the counts of `labels`
/// (Spork et al., CAV 2025)? With `horizon`, also bounds how much the
/// probability of reaching any labelled situation by then can differ.
pub fn epsdelta(
    source_a: &str,
    source_b: &str,
    labels: &[Atom],
    epsilon: Option<f64>,
    delta: f64,
    horizon: Option<f64>,
    options: Options,
) -> Result<EpsDeltaReport, Vec<Diagnostic>> {
    if !(delta >= 0.0 && delta.is_finite()) {
        return Err(error("delta must be a non-negative number"));
    }
    if let Some(e) = epsilon
        && !(0.0..=1.0).contains(&e)
    {
        return Err(error("epsilon must be between 0 and 1"));
    }
    let a = derive_counted("a", source_a, labels, options)?;
    let b = derive_counted("b", source_b, labels, options)?;
    let label = |c: &Counted| {
        let counts = c.counts.clone();
        move |s: StateId| {
            counts
                .iter()
                .map(|col| col[s.index()] as u64)
                .collect::<Vec<u64>>()
        }
    };
    let (la, lb) = (label(&a), label(&b));
    let min = min_epsilon(&a.model.lts, &b.model.lts, &la, &lb, delta).map_err(metric_error)?;
    let within = match epsilon {
        Some(e) => {
            let rel = epsdelta_relation(&a.model.lts, &b.model.lts, &la, &lb, e, delta)
                .map_err(metric_error)?;
            let (sa, sb) = (a.model.lts.initial(), b.model.lts.initial());
            Some(matches!((sa, sb), (Some(s), Some(t)) if rel.related(s, t)))
        }
        None => None,
    };
    let q = common_rate(&a.model.lts, &b.model.lts);
    let used = epsilon.or(min);
    let bound = match (horizon, used) {
        (Some(t), Some(e)) => Some(reach_bound(e, delta, q, t)),
        _ => None,
    };
    let rate_only = horizon.map(|t| rate_only_bound(delta, q, t));
    let explanation = match (min, within, bound) {
        (None, _, _) => format!(
            "The models are not (ε, δ)-bisimilar for any ε at δ = {delta}: some reachable \
             situation has a different label, or exit rates differ by more than a factor \
             {:.3}.",
            delta.exp()
        ),
        (Some(e), w, b) => {
            let mut s = format!(
                "The smallest tolerance on jump probabilities is ε = {e:.4} at rates within a \
                 factor {:.3} (δ = {delta}).",
                delta.exp()
            );
            if let (Some(w), Some(eps)) = (w, epsilon) {
                let verdict = if w {
                    "the models are within tolerance"
                } else {
                    "they are not"
                };
                s += &format!(" At ε = {eps} {verdict}.");
            }
            if let (Some(b), Some(t)) = (b, horizon) {
                s += &format!(
                    " Guarantee: the probability of reaching any labelled situation within \
                     {t} time units differs by at most {b:.4} (Spork et al., Proposition 3); \
                     the bound grows with time and is loose for long horizons."
                );
            }
            s
        }
    };
    Ok(EpsDeltaReport {
        delta,
        min_epsilon: min,
        within,
        rate: q,
        reach_bound: bound,
        rate_only_bound: rate_only,
        explanation,
    })
}

/// A distance estimated from two logged state sequences.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamReport {
    pub estimate: f64,
    pub steps: usize,
    /// For each state of `a`, the state of `b` it is matched with.
    pub matching: Vec<(String, String)>,
    pub explanation: String,
}

/// Estimates the bisimulation distance from two sequences of observed
/// states (consecutive entries are transitions), with `values` giving each
/// state's observation (Calo et al., NeurIPS 2025). Uses consecutive pairs
/// of each log, the paper's practical variant without its guarantee; the
/// estimate needs very many transitions to settle (see `bisim-metric`).
pub fn stream_distance(
    a: &[String],
    b: &[String],
    values: &HashMap<String, f64>,
    gamma: f64,
) -> Result<StreamReport, Vec<Diagnostic>> {
    if a.len() < 2 || b.len() < 2 {
        return Err(error("each log needs at least two observed states"));
    }
    let intern = |log: &[String]| {
        let mut names: Vec<String> = Vec::new();
        let ids: Vec<usize> = log
            .iter()
            .map(|s| match names.iter().position(|n| n == s) {
                Some(i) => i,
                None => {
                    names.push(s.clone());
                    names.len() - 1
                }
            })
            .collect();
        (names, ids)
    };
    let ((na, ia), (nb, ib)) = (intern(a), intern(b));
    let value = |n: &String| {
        values
            .get(n)
            .copied()
            .filter(|v| v.is_finite())
            .ok_or_else(|| error(format!("no finite value for state `{n}`")))
    };
    let va: Vec<f64> = na.iter().map(value).collect::<Result<_, _>>()?;
    let vb: Vec<f64> = nb.iter().map(value).collect::<Result<_, _>>()?;
    let lo = va.iter().chain(&vb).copied().fold(f64::INFINITY, f64::min);
    let hi = va
        .iter()
        .chain(&vb)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let scale = if hi > lo { hi - lo } else { 1.0 };
    let cost = |x: usize, y: usize| (va[x] - vb[y]).abs() / scale;
    let mut est = SampleDistance::new(na.len(), nb.len(), cost, gamma, (ia[0], ib[0]))
        .map_err(metric_error)?;
    let steps = (ia.len() - 1).max(ib.len() - 1);
    for k in 0..steps {
        let (x, y) = (k % (ia.len() - 1), k % (ib.len() - 1));
        est.push((ia[x], ia[x + 1]), (ib[y], ib[y + 1]));
    }
    let matching = est
        .matching()
        .into_iter()
        .enumerate()
        .map(|(x, y)| (na[x].clone(), nb[y].clone()))
        .collect();
    Ok(StreamReport {
        estimate: est.estimate(),
        steps,
        explanation: format!(
            "Estimated from {steps} observed transitions per log, without a model. This \
             estimator converges slowly (error shrinking like 1/√steps): treat values from \
             short logs as rough, and compare candidates on logs of equal length."
        ),
        matching,
    })
}

/// A comparison of two continuous series.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DsaReport {
    pub distance: f64,
    /// Angle between the fitted dynamics, in `[0, π]` (scale-free).
    pub angle: f64,
    pub explanation: String,
}

/// Dynamical similarity analysis of two multichannel series
/// (`series[t][channel]`, Ostrow et al., NeurIPS 2023).
pub fn dsa(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    delays: Option<usize>,
    rank: Option<usize>,
) -> Result<DsaReport, Vec<Diagnostic>> {
    let mut options = DsaOptions::default();
    if let Some(d) = delays {
        options.delays = d.clamp(1, 64);
    }
    if let Some(r) = rank {
        options.rank = r.clamp(1, 32);
    }
    let r = dsa_metric(x, y, options).map_err(metric_error)?;
    Ok(DsaReport {
        distance: r.distance,
        angle: r.angle,
        explanation: format!(
            "Linear dynamics of rank {} were fitted to each series ({} delays) and compared \
             up to a rotation of coordinates: distance {:.4}, angle {:.1}°. This compares \
             how the signals evolve, not their levels; it is not a bisimulation.",
            options.rank,
            options.delays,
            r.distance,
            r.angle.to_degrees()
        ),
    })
}
