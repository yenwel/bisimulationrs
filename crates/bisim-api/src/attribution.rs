//! What-if analysis and attribution of a gap to rates, on a PEPA model.
//!
//! Two questions that go beyond "do the plan and the observations agree?":
//!
//! - [`what_if`]: what happens to the expected count of a state if one rate
//!   is *set* to a value (an intervention, `do(rate = x)`), compared with the
//!   baseline model?
//! - [`attribute`]: the plan and the fitted model differ in several rates, and
//!   so does the outcome. How much of the gap does each rate account for?
//!   The answer is a Shapley decomposition: every rate is switched from its
//!   plan value to its fitted value in every possible order, and its
//!   contribution is averaged over the orders. The parts add up to the gap
//!   exactly (efficiency), however the rates interact.
//!
//! Both are **model-based**: they describe how the model behaves when a rate
//! changes, not what causes what on the farm. Setting a rate in the model is
//! an intervention on the model; the result carries over to the farm only if
//! the model's structure is right, and a handful of dated counts rarely tells
//! competing mechanisms apart. A control plot or a matched comparison is the
//! stronger evidence.
//!
//! The Shapley split is a convention with a uniqueness theorem behind it
//! (efficiency, symmetry, dummy, additivity), not the only reasonable split:
//! it averages over all orders of switching the rates as if each order were
//! equally likely, and it depends on starting from the plan. The effect of a
//! rate alone is reported next to it for that reason.
//!
//! Causality between *events* (which event enabled which) is a different
//! notion and is not covered here.

use bisim_algebra::pepa::{ParamId, PepaModel, Sym, parse_pepa};
use serde::{Deserialize, Serialize};

use crate::observe::{Forecast, Timeline, error, forecast_model};
use crate::{Diagnostic, Options, diagnostic};

/// Most rates [`attribute`] decomposes: it evaluates the model once for
/// every subset of them (`2^n` evaluations).
pub const MAX_ATTRIBUTION_PARAMS: usize = 8;

/// Sets one rate parameter of the model to a value (an intervention on the
/// model: `param := value`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intervention {
    pub param: String,
    pub value: f64,
}

/// The baseline and the intervened forecast of one state.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WhatIf {
    pub baseline: Forecast,
    pub intervened: Forecast,
    /// `intervened.expected − baseline.expected` at every time.
    pub effect: Vec<f64>,
    pub explanation: String,
}

fn param_id(model: &PepaModel<Sym>, name: &str) -> Result<ParamId, Vec<Diagnostic>> {
    model.param_id(name).ok_or_else(|| {
        let known: Vec<&str> = model.params().map(|(_, n, _)| n).collect();
        error(format!(
            "no parameter `{name}`; the model has {}",
            known.join(", ")
        ))
    })
}

/// Compares the baseline model with the model after `interventions`.
///
/// A rate that appears in a static rate must stay positive, so an
/// intervention cannot switch a mechanism off exactly; use a very small
/// value instead.
pub fn what_if(
    source: &str,
    interventions: &[Intervention],
    component: Option<&str>,
    state: &str,
    elapsed: &[f64],
    timeline: &Timeline,
    options: Options,
) -> Result<WhatIf, Vec<Diagnostic>> {
    if interventions.is_empty() {
        return Err(error("give at least one intervention"));
    }
    let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
    let mut changed = model.clone();
    for iv in interventions {
        if !iv.value.is_finite() || iv.value < 0.0 {
            return Err(error(format!(
                "`{}` must be set to a finite, non-negative value, got {}",
                iv.param, iv.value
            )));
        }
        changed = changed.with_param(param_id(&model, &iv.param)?, iv.value);
    }
    let baseline = forecast_model(&model, component, state, elapsed, timeline, options)?;
    let intervened = forecast_model(&changed, component, state, elapsed, timeline, options)?;
    let effect: Vec<f64> = intervened
        .expected
        .iter()
        .zip(&baseline.expected)
        .map(|(a, b)| a - b)
        .collect();
    let settings: Vec<String> = interventions
        .iter()
        .map(|iv| format!("{} = {}", iv.param, iv.value))
        .collect();
    let last = effect.last().copied().unwrap_or(0.0);
    let explanation = format!(
        "In the model, setting {} changes the expected number of `{state}` by {last:+.2} at the \
         last time ({:.2} → {:.2}). This describes how the model behaves when the rate changes; \
         it holds for the farm only if the model's structure is right.",
        settings.join(", "),
        baseline.expected.last().copied().unwrap_or(0.0),
        intervened.expected.last().copied().unwrap_or(0.0),
    );
    Ok(WhatIf {
        baseline,
        intervened,
        effect,
        explanation,
    })
}

/// What one rate contributes to the gap between plan and actual.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Contribution {
    pub name: String,
    pub plan: f64,
    pub actual: f64,
    /// Shapley value: the average effect on the outcome of switching this
    /// rate from plan to actual, over every order of switching the rates.
    pub shapley: f64,
    /// `shapley / gap`, absent when the gap is (numerically) zero.
    pub share: Option<f64>,
    /// Effect of switching only this rate, all others at plan.
    pub alone: f64,
    /// Effect of switching this rate last, all others already at actual.
    pub without: f64,
}

/// The gap between two models' expected count, split by rate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Attribution {
    pub component: String,
    pub state: String,
    /// Time since the start, in the timeline's unit.
    pub at: f64,
    pub plan: f64,
    pub actual: f64,
    pub gap: f64,
    /// Rates that differ, largest absolute Shapley value first.
    pub contributions: Vec<Contribution>,
    pub explanation: String,
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

/// Splits the difference in the expected count of `state` at time `at`
/// between `plan_source` and `actual_source` (typically the fitted model)
/// over the rates in which they differ.
///
/// The two models must be identical except for the values of their
/// parameters; this is checked, not assumed.
pub fn attribute(
    plan_source: &str,
    actual_source: &str,
    component: Option<&str>,
    state: &str,
    at: f64,
    timeline: &Timeline,
    options: Options,
) -> Result<Attribution, Vec<Diagnostic>> {
    let plan = parse_pepa(plan_source).map_err(|e| vec![diagnostic(&e)])?;
    let actual = parse_pepa(actual_source).map_err(|e| vec![diagnostic(&e)])?;
    let plan_params: Vec<(ParamId, String, f64)> = plan
        .params()
        .map(|(id, name, v)| (id, name.to_owned(), v))
        .collect();
    let actual_params: Vec<(String, f64)> = actual
        .params()
        .map(|(_, name, v)| (name.to_owned(), v))
        .collect();
    if plan_params.len() != actual_params.len()
        || plan_params
            .iter()
            .zip(&actual_params)
            .any(|((_, a, _), (b, _))| a != b)
    {
        return Err(error(
            "the two models must have the same parameters; they differ in more than rate values",
        ));
    }
    // The rates that differ, with their plan and actual values.
    let differing: Vec<(ParamId, String, f64, f64)> = plan_params
        .iter()
        .zip(&actual_params)
        .filter(|((_, _, p), (_, a))| !close(*p, *a))
        .map(|((id, name, p), (_, a))| (*id, name.clone(), *p, *a))
        .collect();
    let n = differing.len();
    if n > MAX_ATTRIBUTION_PARAMS {
        return Err(error(format!(
            "{n} rates differ; at most {MAX_ATTRIBUTION_PARAMS} can be attributed (the cost is 2^n)"
        )));
    }
    let at_times = [at];
    // Outcome with the rates in `mask` at their actual values, the rest at plan.
    let outcome = |mask: usize| -> Result<(f64, String), Vec<Diagnostic>> {
        let model = differing
            .iter()
            .enumerate()
            .filter(|(i, _)| mask >> i & 1 == 1)
            .fold(plan.clone(), |m, (_, (id, _, _, a))| m.with_param(*id, *a));
        let f = forecast_model(&model, component, state, &at_times, timeline, options)?;
        Ok((f.expected[0], f.component))
    };
    let mut values = Vec::with_capacity(1 << n);
    let mut component_name = String::new();
    for mask in 0..1usize << n {
        let (v, c) = outcome(mask)?;
        values.push(v);
        component_name = c;
    }
    let (plan_value, actual_value) = (values[0], values[(1 << n) - 1]);
    // The decomposition assumes the actual model *is* the plan with those
    // rates changed. Computed parameters and structural edits break that.
    let direct = forecast_model(&actual, component, state, &at_times, timeline, options)?;
    if !close(direct.expected[0], actual_value) {
        return Err(error(
            "the two models differ in more than the values of their parameters (structure or \
             computed parameters), so the gap cannot be attributed to rates alone",
        ));
    }
    let gap = actual_value - plan_value;

    let factorial = |k: usize| (1..=k).map(|x| x as f64).product::<f64>();
    let mut contributions: Vec<Contribution> = differing
        .iter()
        .enumerate()
        .map(|(i, (_, name, p, a))| {
            let bit = 1usize << i;
            let mut shapley = 0.0;
            for mask in 0..1usize << n {
                if mask & bit != 0 {
                    continue;
                }
                let k = mask.count_ones() as usize;
                let weight = factorial(k) * factorial(n - k - 1) / factorial(n);
                shapley += weight * (values[mask | bit] - values[mask]);
            }
            Contribution {
                name: name.clone(),
                plan: *p,
                actual: *a,
                shapley,
                share: (gap.abs() > 1e-12).then(|| shapley / gap),
                alone: values[bit] - values[0],
                without: values[(1 << n) - 1] - values[(1 << n) - 1 - bit],
            }
        })
        .collect();
    contributions.sort_by(|a, b| b.shapley.abs().total_cmp(&a.shapley.abs()));

    let mut explanation = if n == 0 {
        "The models have identical parameters, so there is nothing to attribute.".to_owned()
    } else {
        let parts: Vec<String> = contributions
            .iter()
            .map(|c| match c.share {
                Some(s) => format!("{} {:+.2} ({:.0} %)", c.name, c.shapley, s * 100.0),
                None => format!("{} {:+.2}", c.name, c.shapley),
            })
            .collect();
        format!(
            "The expected number of `{state}` at time {at} is {actual_value:.2} instead of \
             {plan_value:.2} (gap {gap:+.2}). Switching the rates from plan to actual, averaged \
             over every order: {}.",
            parts.join(", ")
        )
    };
    explanation.push_str(
        " This is an attribution within the model, not proof of what causes the gap: it holds \
         for the farm only if the model's structure is right, and rates interact, so a rate's \
         share depends on the others and on starting from the plan.",
    );
    Ok(Attribution {
        component: component_name,
        state: state.to_owned(),
        at,
        plan: plan_value,
        actual: actual_value,
        gap,
        contributions,
        explanation,
    })
}
