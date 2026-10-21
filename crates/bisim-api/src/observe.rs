//! Comparing a plan (a PEPA model) with dated counts.
//!
//! Each observation says how many copies of a component were in some state
//! on a date, e.g. "on 12/06/2026, 9 of the 24 tomato plants were flowering".
//! The model gives the exact distribution of that count on that date, hence
//! an expected value, a range that holds with the chosen probability (95 %
//! by default) and a p-value. The overall verdict adjusts for the number of
//! counts checked (Bonferroni), which stays valid when the same plants are
//! counted repeatedly.
//!
//! State names match exactly or by prefix before `_`, so a stage modelled as
//! `Flowering_1`, `Flowering_2` (to make its duration less random) is
//! observed simply as `Flowering`.

use bisim_algebra::pepa::{PepaModel, Sym, parse_pepa};
use bisim_stochastic::nelder_mead;
use serde::{Deserialize, Serialize};

use crate::engine::Evaluator;
use crate::{Diagnostic, Options, Severity, diagnostic};

/// A NaN or infinite probability would read as "no probability mass" and so
/// as a count far off plan; report it as the model error it is.
fn ensure_finite(snaps: &crate::engine::Snapshots) -> Result<(), Vec<Diagnostic>> {
    if snaps.is_finite() {
        Ok(())
    } else {
        Err(error(
            "the model produced non-finite probabilities; check its rates",
        ))
    }
}

pub(crate) fn error(message: impl Into<String>) -> Vec<Diagnostic> {
    vec![Diagnostic {
        severity: Severity::Error,
        line: 0,
        column: 0,
        message: message.into(),
    }]
}

// ---------- Dates (Western European dd/mm/yyyy; ISO yyyy-mm-dd also accepted) ----------

/// Days since 01/01/1970 for a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// Parses `dd/mm/yyyy` (also `-` or `.` separators, one-digit day or month)
/// or ISO `yyyy-mm-dd`. Returns days since 01/01/1970.
pub fn parse_date(text: &str) -> Result<i64, String> {
    let bad = || format!("`{text}` is not a date; write it as dd/mm/yyyy, e.g. 15/05/2026");
    let parts: Vec<&str> = text.trim().split(['/', '-', '.']).collect();
    if parts.len() != 3 {
        return Err(bad());
    }
    // Plain digits only (no sign), at most four per part, and a four-digit
    // year: `15/05/26` is not the year 26 AD.
    if parts
        .iter()
        .any(|p| p.is_empty() || p.len() > 4 || !p.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err(bad());
    }
    let nums: Vec<i64> = parts
        .iter()
        .map(|p| p.parse::<i64>().map_err(|_| bad()))
        .collect::<Result<_, _>>()?;
    let (y, m, d) = if parts[0].len() == 4 {
        (nums[0], nums[1], nums[2])
    } else {
        (nums[2], nums[1], nums[0])
    };
    let year_digits = if parts[0].len() == 4 {
        parts[0].len()
    } else {
        parts[2].len()
    };
    if year_digits != 4 || !(1..=12).contains(&m) || !(1583..=9999).contains(&y) {
        return Err(bad());
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days_in_month = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if d < 1 || d > days_in_month[(m - 1) as usize] {
        return Err(bad());
    }
    Ok(days_from_civil(y, m, d))
}

/// Formats days since 01/01/1970 as `dd/mm/yyyy`.
pub fn format_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{d:02}/{m:02}/{y:04}")
}

/// The time unit the model's rates use (rates are "per unit").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TimeUnit {
    #[serde(alias = "s", alias = "sec", alias = "seconds")]
    Second,
    #[serde(alias = "min", alias = "minutes")]
    Minute,
    #[serde(alias = "h", alias = "hours")]
    Hour,
    #[default]
    #[serde(alias = "d", alias = "days")]
    Day,
    #[serde(alias = "w", alias = "weeks")]
    Week,
}

impl TimeUnit {
    /// Model time units per day.
    fn per_day(self) -> f64 {
        match self {
            TimeUnit::Second => 86_400.0,
            TimeUnit::Minute => 1_440.0,
            TimeUnit::Hour => 24.0,
            TimeUnit::Day => 1.0,
            TimeUnit::Week => 1.0 / 7.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TimeUnit::Second => "s",
            TimeUnit::Minute => "min",
            TimeUnit::Hour => "h",
            TimeUnit::Day => "days",
            TimeUnit::Week => "weeks",
        }
    }
}

impl std::str::FromStr for TimeUnit {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "s" | "sec" | "second" | "seconds" => TimeUnit::Second,
            "min" | "minute" | "minutes" => TimeUnit::Minute,
            "h" | "hour" | "hours" => TimeUnit::Hour,
            "d" | "day" | "days" => TimeUnit::Day,
            "w" | "week" | "weeks" => TimeUnit::Week,
            other => {
                return Err(format!(
                    "unknown time unit `{other}`; use second, minute, hour, day or week"
                ));
            }
        })
    }
}

// ---------- Inputs and results ----------

/// One dated count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// `dd/mm/yyyy`.
    pub date: String,
    /// Component as in the system equation (`Plant[24]` or `Plant`); may be
    /// omitted when the model has one component.
    #[serde(default)]
    pub component: Option<String>,
    /// State (or stage prefix, see the module docs).
    pub state: String,
    pub count: u32,
}

/// When the plan starts and what its rates are measured in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timeline {
    /// Date of the model's initial state, `dd/mm/yyyy`.
    pub start: String,
    #[serde(default)]
    pub unit: TimeUnit,
    /// Probability covered by the expected ranges (default 0.95).
    #[serde(default = "default_level")]
    pub level: f64,
}

fn default_level() -> f64 {
    0.95
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    OnTrack,
    AbovePlan,
    BelowPlan,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ObservationResult {
    pub date: String,
    /// Time since the start, in the model's unit.
    pub elapsed: f64,
    pub component: String,
    pub state: String,
    pub observed: u32,
    pub expected: f64,
    /// Range that holds with probability `level` under the plan.
    pub low: u32,
    pub high: u32,
    /// Two-sided p-value: how surprising the count is under the plan.
    pub p_value: f64,
    pub status: Status,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ObservationReport {
    pub start: String,
    pub unit: TimeUnit,
    pub level: f64,
    pub rows: Vec<ObservationResult>,
    /// No count is too far off once the number of checks is accounted for.
    pub on_track: bool,
    pub outside_range: usize,
    /// p-values below this mean "off plan" (Bonferroni: (1 − level) / checks).
    pub threshold: f64,
    pub explanation: String,
    /// How counts were computed for each component.
    pub methods: Vec<String>,
}

/// A resolved observation: leaf index, matched derivatives, model time.
struct Resolved {
    leaf: usize,
    leaf_name: String,
    matched: Vec<String>,
    elapsed: f64,
    date: String,
}

fn stage_names(derivatives: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for d in derivatives {
        let stage = match d.rsplit_once('_') {
            Some((head, tail)) if !head.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => {
                head
            }
            _ => d.as_str(),
        };
        if !out.iter().any(|s| s == stage) {
            out.push(stage.to_owned());
        }
    }
    out
}

fn resolve(
    eval: &Evaluator,
    obs: &[Observation],
    timeline: &Timeline,
) -> Result<Vec<Resolved>, Vec<Diagnostic>> {
    let start = parse_date(&timeline.start).map_err(|e| error(format!("start date: {e}")))?;
    let leaves = eval.leaves();
    obs.iter()
        .enumerate()
        .map(|(i, o)| {
            let at = |msg: String| {
                error(format!(
                    "observation {} ({}, {}): {msg}",
                    i + 1,
                    o.date,
                    o.state
                ))
            };
            let day = parse_date(&o.date).map_err(at)?;
            if day < start {
                return Err(at(format!(
                    "the date is before the start date {}",
                    format_date(start)
                )));
            }
            let leaf = match &o.component {
                None if leaves.len() == 1 => 0,
                None => {
                    let names: Vec<&str> = leaves.iter().map(|l| l.name.as_str()).collect();
                    return Err(at(format!(
                        "say which component was counted: {}",
                        names.join(", ")
                    )));
                }
                Some(c) => {
                    let hits: Vec<usize> = (0..leaves.len())
                        .filter(|&k| leaves[k].name == *c || leaves[k].component == *c)
                        .collect();
                    match hits.as_slice() {
                        [k] => *k,
                        [] => {
                            let names: Vec<&str> = leaves.iter().map(|l| l.name.as_str()).collect();
                            return Err(at(format!(
                                "no component `{c}`; the model has {}",
                                names.join(", ")
                            )));
                        }
                        _ => {
                            return Err(at(format!(
                                "`{c}` occurs more than once; use its position-specific name"
                            )));
                        }
                    }
                }
            };
            let derivatives = eval.derivatives(leaf);
            let prefix = format!("{}_", o.state);
            let matched: Vec<String> = derivatives
                .iter()
                .filter(|d| **d == o.state || d.starts_with(&prefix))
                .cloned()
                .collect();
            if matched.is_empty() {
                return Err(at(format!(
                    "`{}` has no state `{}`; its states are {}",
                    leaves[leaf].name,
                    o.state,
                    stage_names(&derivatives).join(", ")
                )));
            }
            if o.count > leaves[leaf].copies {
                return Err(at(format!(
                    "counted {} but `{}` has only {} copies",
                    o.count, leaves[leaf].name, leaves[leaf].copies
                )));
            }
            Ok(Resolved {
                leaf,
                leaf_name: leaves[leaf].name.clone(),
                matched,
                elapsed: (day - start) as f64 * timeline.unit.per_day(),
                date: format_date(day),
            })
        })
        .collect()
}

fn fmt_p(p: f64) -> String {
    if p < 0.0001 {
        "< 0.0001".to_owned()
    } else {
        format!("{p:.4}")
    }
}

fn evaluate(
    model: &PepaModel<Sym>,
    obs: &[Observation],
    timeline: &Timeline,
    options: Options,
) -> Result<(ObservationReport, f64), Vec<Diagnostic>> {
    let mut session = ObserveSession::from_model(model, timeline, options)?;
    let report = session.push(obs)?;
    Ok((report, session.log_likelihood))
}

/// Dated counts checked against a plan as they come in. Each count advances
/// the model from the previous date instead of recomputing from the start
/// (a count dated earlier than the latest one is computed from the start),
/// and the verdict covers every count so far, exactly as
/// [`check_observations`] on all of them.
pub struct ObserveSession {
    eval: Evaluator,
    timeline: Timeline,
    cursor: crate::engine::Cursor,
    rows: Vec<ObservationResult>,
    log_likelihood: f64,
}

impl std::fmt::Debug for ObserveSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObserveSession")
            .field("rows", &self.rows.len())
            .finish_non_exhaustive()
    }
}

impl ObserveSession {
    /// A session for the plan in `source`.
    pub fn new(
        source: &str,
        timeline: &Timeline,
        options: Options,
    ) -> Result<Self, Vec<Diagnostic>> {
        let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
        Self::from_model(&model, timeline, options)
    }

    fn from_model(
        model: &PepaModel<Sym>,
        timeline: &Timeline,
        options: Options,
    ) -> Result<Self, Vec<Diagnostic>> {
        if !(0.5..1.0).contains(&timeline.level) {
            return Err(error(
                "the range level must be between 0.5 and 1 (e.g. 0.95)",
            ));
        }
        parse_date(&timeline.start).map_err(|e| error(format!("start date: {e}")))?;
        let eval = Evaluator::new(model, options)?;
        let cursor = eval.cursor();
        Ok(Self {
            eval,
            timeline: timeline.clone(),
            cursor,
            rows: Vec::new(),
            log_likelihood: 0.0,
        })
    }

    /// Adds counts (all or none: a count that cannot be resolved adds
    /// nothing) and returns the report on every count so far.
    pub fn push(&mut self, obs: &[Observation]) -> Result<ObservationReport, Vec<Diagnostic>> {
        if obs.is_empty() && self.rows.is_empty() {
            return Err(error("add at least one observation"));
        }
        let resolved = resolve(&self.eval, obs, &self.timeline)?;
        // In time order, so each count advances from the previous one.
        let mut order: Vec<usize> = (0..obs.len()).collect();
        order.sort_by(|&a, &b| resolved[a].elapsed.total_cmp(&resolved[b].elapsed));
        let mut new_rows: Vec<Option<ObservationResult>> = vec![None; obs.len()];
        let mut ll = 0.0;
        for i in order {
            let (r, o) = (&resolved[i], &obs[i]);
            let snaps = self.eval.snapshot_at(&mut self.cursor, r.elapsed);
            ensure_finite(&snaps)?;
            let dist = self.eval.count_distribution(&snaps, 0, r.leaf, &r.matched);
            ll += dist.prob(o.count).max(1e-300).ln();
            let (low, high) = dist.central_range(self.timeline.level);
            let status = if o.count < low {
                Status::BelowPlan
            } else if o.count > high {
                Status::AbovePlan
            } else {
                Status::OnTrack
            };
            new_rows[i] = Some(ObservationResult {
                date: r.date.clone(),
                elapsed: r.elapsed,
                component: r.leaf_name.clone(),
                state: o.state.clone(),
                observed: o.count,
                expected: dist.mean(),
                low,
                high,
                p_value: dist.p_value(o.count),
                status,
            });
        }
        self.rows.extend(new_rows.into_iter().flatten());
        self.log_likelihood += ll;
        self.report()
    }

    /// The report on every count so far.
    pub fn report(&self) -> Result<ObservationReport, Vec<Diagnostic>> {
        let (rows, timeline, eval) = (&self.rows, &self.timeline, &self.eval);
        let checks = rows.len();
        if checks == 0 {
            return Err(error("add at least one observation"));
        }
        let threshold = (1.0 - timeline.level) / checks as f64;
        let outside = rows.iter().filter(|r| r.status != Status::OnTrack).count();
        let worst = rows
            .iter()
            .min_by(|a, b| a.p_value.total_cmp(&b.p_value))
            .expect("at least one row");
        let on_track = worst.p_value >= threshold;
        let pct = format!("{:.0} %", timeline.level * 100.0);
        let explanation = if outside == 0 {
            format!("On track: all {checks} counts are within their {pct} ranges.")
        } else if on_track {
            format!(
                "Still consistent with the plan: {outside} of {checks} counts fall outside their {pct} range, \
                 but with {checks} checks that can happen by chance (smallest p-value {}, \
                 off plan would need below {}).",
                fmt_p(worst.p_value),
                fmt_p(threshold)
            )
        } else {
            format!(
                "Off plan: {outside} of {checks} counts fall outside their {pct} range. \
                 {} {} on {} ({} counted, {:.1} expected, range {}–{}) is too far off to be chance \
                 (p = {}, below {}).",
                worst.component,
                worst.state,
                worst.date,
                worst.observed,
                worst.expected,
                worst.low,
                worst.high,
                fmt_p(worst.p_value),
                fmt_p(threshold)
            )
        };
        let methods = eval
            .leaves()
            .iter()
            .enumerate()
            .map(|(k, l)| {
                if eval.is_independent(k) {
                    format!("{}: binomial over {} independent copies", l.name, l.copies)
                } else {
                    format!("{}: exact distribution from the full Markov chain", l.name)
                }
            })
            .collect();
        Ok(ObservationReport {
            start: format_date(parse_date(&timeline.start).map_err(error)?),
            unit: timeline.unit,
            level: timeline.level,
            rows: rows.clone(),
            on_track,
            outside_range: outside,
            threshold,
            explanation,
            methods,
        })
    }
}

/// Checks dated counts against the plan.
pub fn check_observations(
    source: &str,
    observations: &[Observation],
    timeline: &Timeline,
    options: Options,
) -> Result<ObservationReport, Vec<Diagnostic>> {
    let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
    evaluate(&model, observations, timeline, options).map(|(r, _)| r)
}

/// Expected count and range of one state over time, for a forecast chart.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Forecast {
    pub component: String,
    pub state: String,
    pub unit: TimeUnit,
    /// Time since the start, in the model's unit.
    pub elapsed: Vec<f64>,
    /// The matching calendar dates (`dd/mm/yyyy`, rounded to whole days).
    pub dates: Vec<String>,
    pub expected: Vec<f64>,
    pub low: Vec<u32>,
    pub high: Vec<u32>,
    pub copies: u32,
}

/// Expected count and `level` range of `state` at each elapsed time.
pub fn forecast(
    source: &str,
    component: Option<&str>,
    state: &str,
    elapsed: &[f64],
    timeline: &Timeline,
    options: Options,
) -> Result<Forecast, Vec<Diagnostic>> {
    let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
    forecast_model(&model, component, state, elapsed, timeline, options)
}

/// [`forecast`] for a model that is already parsed (and perhaps modified:
/// see [`crate::attribution`]).
pub(crate) fn forecast_model(
    model: &PepaModel<Sym>,
    component: Option<&str>,
    state: &str,
    elapsed: &[f64],
    timeline: &Timeline,
    options: Options,
) -> Result<Forecast, Vec<Diagnostic>> {
    if let Some(t) = elapsed.iter().find(|t| !t.is_finite() || **t < 0.0) {
        return Err(error(format!("times must be non-negative, got {t}")));
    }
    if !(0.5..1.0).contains(&timeline.level) {
        return Err(error(
            "the range level must be between 0.5 and 1 (e.g. 0.95)",
        ));
    }
    let eval = Evaluator::new(model, options)?;
    let start = parse_date(&timeline.start).map_err(|e| error(format!("start date: {e}")))?;
    // Resolve component and state through a dummy observation at the start.
    let probe = Observation {
        date: format_date(start),
        component: component.map(str::to_owned),
        state: state.to_owned(),
        count: 0,
    };
    let r = resolve(&eval, std::slice::from_ref(&probe), timeline)?.remove(0);
    let snaps = eval.snapshots(elapsed);
    ensure_finite(&snaps)?;
    let (mut expected, mut low, mut high) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..elapsed.len() {
        let d = eval.count_distribution(&snaps, i, r.leaf, &r.matched);
        let (lo, hi) = d.central_range(timeline.level);
        expected.push(d.mean());
        low.push(lo);
        high.push(hi);
    }
    Ok(Forecast {
        component: r.leaf_name,
        state: state.to_owned(),
        unit: timeline.unit,
        elapsed: elapsed.to_vec(),
        dates: elapsed
            .iter()
            .map(|t| format_date(start + (t / timeline.unit.per_day()).round() as i64))
            .collect(),
        expected,
        low,
        high,
        copies: eval.leaves()[r.leaf].copies,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FittedParam {
    pub name: String,
    pub planned: f64,
    pub fitted: f64,
    /// `fitted / planned`.
    pub ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FitReport {
    pub params: Vec<FittedParam>,
    /// Log-likelihood of the counts under the plan and the fitted model
    /// (treating counts as independent; used to rank, not for inference).
    pub planned_log_likelihood: f64,
    pub fitted_log_likelihood: f64,
    /// The model text with fitted parameter values.
    pub fitted_source: String,
    /// The observations checked against the fitted model.
    pub fitted_check: ObservationReport,
    pub explanation: String,
}

/// Finds parameter values that make the observed counts most likely.
///
/// The parameters must be rate parameters of the model (`name = value;`),
/// used in prefix rates; they are kept positive. Counts on different dates
/// of the same copies are treated as independent, so the fit is a point
/// estimate: use it to see *which* parameter differs from the plan, and
/// compare the fitted model with the plan to see the effect.
pub fn fit_parameters(
    source: &str,
    observations: &[Observation],
    params: &[String],
    timeline: &Timeline,
    options: Options,
) -> Result<FitReport, Vec<Diagnostic>> {
    fit_parameters_from(source, observations, params, None, timeline, options)
}

/// [`fit_parameters`] starting the search from `start` (one positive value
/// per parameter, e.g. the previous fit when new counts arrive) instead of
/// the planned values. The report still compares with the plan.
pub fn fit_parameters_from(
    source: &str,
    observations: &[Observation],
    params: &[String],
    start: Option<&[f64]>,
    timeline: &Timeline,
    options: Options,
) -> Result<FitReport, Vec<Diagnostic>> {
    let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
    if let Some(start) = start
        && (start.len() != params.len() || start.iter().any(|v| !(v.is_finite() && *v > 0.0)))
    {
        return Err(error("start needs one positive value per fitted parameter"));
    }
    if params.is_empty() {
        return Err(error("choose at least one parameter to fit"));
    }
    let ids = params
        .iter()
        .map(|p| {
            model.param_id(p).ok_or_else(|| {
                let known: Vec<&str> = model.params().map(|(_, n, _)| n).collect();
                error(format!(
                    "no parameter `{p}`; the model has {}",
                    known.join(", ")
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let planned: Vec<f64> = ids
        .iter()
        .map(|id| {
            model
                .params()
                .find(|(i, _, _)| i == id)
                .expect("known id")
                .2
        })
        .collect();
    let (_, planned_ll) = evaluate(&model, observations, timeline, options)?;

    let with = |x: &[f64]| {
        ids.iter()
            .zip(x)
            .fold(model.clone(), |m, (id, v)| m.with_param(*id, v.exp()))
    };
    let x0: Vec<f64> = start.unwrap_or(&planned).iter().map(|v| v.ln()).collect();
    let (best, _) = nelder_mead(
        |x| match evaluate(&with(x), observations, timeline, options) {
            Ok((_, ll)) => -ll,
            Err(_) => f64::INFINITY,
        },
        &x0,
        0.3,
        150 * (ids.len() + 1),
        1e-7,
    );
    let fitted_model = with(&best);
    let (fitted_check, fitted_ll) = evaluate(&fitted_model, observations, timeline, options)?;
    let params: Vec<FittedParam> = params
        .iter()
        .zip(&planned)
        .zip(&best)
        .map(|((name, &p), &x)| FittedParam {
            name: name.clone(),
            planned: p,
            fitted: x.exp(),
            ratio: x.exp() / p,
        })
        .collect();
    let changes: Vec<String> = params
        .iter()
        .map(|p| {
            format!(
                "{} {:.3} → {:.3} (×{:.2})",
                p.name, p.planned, p.fitted, p.ratio
            )
        })
        .collect();
    let ratio = (fitted_ll - planned_ll).exp();
    let likelihood = if ratio > 1000.0 {
        "The counts are more than 1000× more likely with the fitted values than with the plan \
         (an approximation: every dated count is treated as independent, which overstates the evidence)."
            .to_owned()
    } else {
        format!(
            "The counts are {ratio:.1}× more likely with the fitted values than with the plan \
             (an approximation: every dated count is treated as independent, which overstates the evidence)."
        )
    };
    let mut explanation = format!(
        "Best fit to the counts: {}. {likelihood}",
        changes.join(", ")
    );
    if observations.len() < 5 * params.len() {
        explanation.push_str(
            " There are few counts per fitted parameter, so these values are rough; \
             fit one parameter at a time to see which stage is off.",
        );
    }
    Ok(FitReport {
        params,
        planned_log_likelihood: planned_ll,
        fitted_log_likelihood: fitted_ll,
        fitted_source: fitted_model.to_string(),
        fitted_check,
        explanation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip_in_western_european_format() {
        let d = parse_date("15/05/2026").unwrap();
        assert_eq!(format_date(d), "15/05/2026");
        assert_eq!(parse_date("2026-05-15").unwrap(), d);
        assert_eq!(parse_date("1/6/2026").unwrap() - d, 17);
        assert_eq!(
            parse_date("01/03/2024").unwrap() - parse_date("28/02/2024").unwrap(),
            2
        );
        assert!(parse_date("31/04/2026").is_err());
        assert!(parse_date("29/02/2026").is_err());
        assert!(parse_date("12/13/2026").unwrap_err().contains("dd/mm/yyyy"));
        assert_eq!(format_date(0), "01/01/1970");
    }

    #[test]
    fn implausible_dates_are_rejected() {
        // Two-digit years are not years 26 AD; signs, huge years (which used
        // to overflow the day count) and empty parts are errors.
        for bad in [
            "15/05/26",
            "15/05/0026",
            "01/01/99999999999999999",
            "+15/+05/+2026",
            "15//2026",
            "15/05/10000",
            "1500-01-01",
        ] {
            assert!(parse_date(bad).is_err(), "{bad}");
        }
        assert!(parse_date("01/01/1583").is_ok());
        assert!(parse_date("31/12/9999").is_ok());
    }

    #[test]
    fn stage_prefixes() {
        let names = ["Veg_1", "Veg_2", "Flowering", "Lost"].map(String::from);
        assert_eq!(stage_names(&names), vec!["Veg", "Flowering", "Lost"]);
    }
}
