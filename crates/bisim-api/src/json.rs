//! A single JSON request/response entry point, used by the WebAssembly
//! module and the R package.
//!
//! Requests are objects with an `op` field:
//!
//! ```json
//! {"op": "check",     "source": "…"}
//! {"op": "analyse",   "source": "…", "max_states": 20000}
//! {"op": "analyse_prism", "source": "…", "max_states": 20000}
//! {"op": "analyse_jani", "source": "…", "max_states": 20000}
//! {"op": "transient", "source": "…", "times": [0, 1, 2]}
//! {"op": "compare",   "a": "…", "b": "…"}
//! {"op": "observe",   "source": "…", "timeline": {"start": "15/05/2026", "unit": "day"},
//!                     "observations": [{"date": "12/06/2026", "state": "Flowering", "count": 9}]}
//! {"op": "forecast",  "source": "…", "timeline": {…}, "state": "Flowering", "times": [0, 7, 14]}
//! {"op": "fit",       "source": "…", "timeline": {…}, "observations": […], "params": ["veg_days"]}
//! {"op": "query",     "source": "…", "query": "P=? [F<=10 s=0]"}
//! {"op": "query_jani", "source": "…", "query": "P=? [F<=10 s=0]"}
//! {"op": "fluid",     "source": "…", "times": [0, 7, 14], "dt": 0.01}
//! {"op": "whatif",    "source": "…", "timeline": {…}, "state": "Ripe", "times": [30],
//!                     "interventions": [{"param": "veg_days", "value": 21}]}
//! {"op": "attribute", "plan": "…", "actual": "…", "timeline": {…}, "state": "Ripe", "at": 60}
//! ```
//!
//! Responses are `{"ok": …}` or `{"error": …}`, where an error is a list of
//! diagnostics (for `compare`: `{"a": […], "b": […]}`).

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Atom, Intervention, Observation, Options, Timeline, analyse, analyse_jani, analyse_prism,
    attribute, check, check_jani, check_observations, check_prism, compare, distance, dsa,
    epsdelta, fit_parameters_from, fluid, forecast, query_jani, query_prism, stream_distance,
    transient, what_if,
};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase", deny_unknown_fields)]
enum Request {
    Check {
        source: String,
    },
    #[serde(rename = "check_prism")]
    CheckPrism {
        source: String,
    },
    #[serde(rename = "check_jani")]
    CheckJani {
        source: String,
    },
    Analyse {
        source: String,
        max_states: Option<usize>,
    },
    #[serde(rename = "analyse_prism")]
    AnalysePrism {
        source: String,
        max_states: Option<usize>,
    },
    #[serde(rename = "analyse_jani")]
    AnalyseJani {
        source: String,
        max_states: Option<usize>,
    },
    Transient {
        source: String,
        times: Vec<f64>,
        max_states: Option<usize>,
    },
    Compare {
        a: String,
        b: String,
        max_states: Option<usize>,
    },
    Observe {
        source: String,
        timeline: Timeline,
        observations: Vec<Observation>,
        max_states: Option<usize>,
    },
    Forecast {
        source: String,
        timeline: Timeline,
        component: Option<String>,
        state: String,
        times: Vec<f64>,
        max_states: Option<usize>,
    },
    Fit {
        source: String,
        timeline: Timeline,
        observations: Vec<Observation>,
        params: Vec<String>,
        /// Start the search here (e.g. the previous fit) instead of the plan.
        start: Option<Vec<f64>>,
        max_states: Option<usize>,
    },
    Query {
        source: String,
        query: String,
        max_states: Option<usize>,
    },
    #[serde(rename = "query_jani")]
    QueryJani {
        source: String,
        query: String,
        max_states: Option<usize>,
    },
    Fluid {
        source: String,
        times: Vec<f64>,
        dt: Option<f64>,
        max_states: Option<usize>,
    },
    WhatIf {
        source: String,
        timeline: Timeline,
        component: Option<String>,
        state: String,
        times: Vec<f64>,
        interventions: Vec<Intervention>,
        max_states: Option<usize>,
    },
    #[serde(rename = "session_open")]
    SessionOpen {
        session: crate::session::OpenSpec,
    },
    #[serde(rename = "session_push")]
    SessionPush {
        id: u64,
        events: Value,
    },
    #[serde(rename = "session_close")]
    SessionClose {
        id: u64,
    },
    Distance {
        a: String,
        b: String,
        observe: Atom,
        half_life: f64,
        max_states: Option<usize>,
    },
    Epsdelta {
        a: String,
        b: String,
        labels: Vec<Atom>,
        epsilon: Option<f64>,
        delta: f64,
        horizon: Option<f64>,
        max_states: Option<usize>,
    },
    #[serde(rename = "stream_distance")]
    StreamDistance {
        a: Vec<String>,
        b: Vec<String>,
        values: std::collections::HashMap<String, f64>,
        gamma: f64,
    },
    Dsa {
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        delays: Option<usize>,
        rank: Option<usize>,
    },
    Attribute {
        plan: String,
        actual: String,
        timeline: Timeline,
        component: Option<String>,
        state: String,
        at: f64,
        max_states: Option<usize>,
    },
}

/// Largest `max_states` a JSON request may ask for. Running out of memory
/// aborts the host process (a browser tab, a Python or R session).
pub const MAX_JSON_STATES: usize = 1_000_000;

fn options(max_states: Option<usize>) -> Options {
    let mut o = Options::default();
    if let Some(m) = max_states {
        o.max_states = m.min(MAX_JSON_STATES);
    }
    o
}

fn respond<T: serde::Serialize, E: serde::Serialize>(r: Result<T, E>) -> Value {
    match r {
        Ok(v) => json!({ "ok": v }),
        Err(e) => json!({ "error": e }),
    }
}

/// Version of the request/response protocol. Requests may state the
/// version they were written for (`"version": 1`); a different one is
/// refused rather than guessed at. Responses carry the version too.
pub const PROTOCOL_VERSION: u64 = 1;

/// Handles one JSON request and returns the JSON response.
pub fn handle_json(request: &str) -> String {
    let mut response = handle(request);
    if let Value::Object(map) = &mut response {
        map.insert("version".to_owned(), json!(PROTOCOL_VERSION));
    }
    response.to_string()
}

fn handle(request: &str) -> Value {
    let parsed = serde_json::from_str::<Value>(request).and_then(|mut v| {
        if let Some(Value::Number(n)) = v.as_object_mut().and_then(|o| o.remove("version")) {
            if n.as_u64() != Some(PROTOCOL_VERSION) {
                return Err(serde::de::Error::custom(format!(
                    "protocol version {n} is not supported (this is version {PROTOCOL_VERSION})"
                )));
            }
        }
        serde_json::from_value::<Request>(v)
    });
    match parsed {
        Err(e) => json!({ "error": [{ "severity": "error", "line": 0, "column": 0,
                                     "message": format!("invalid request: {e}") }] }),
        Ok(Request::Check { source }) => json!({ "ok": check(&source) }),
        Ok(Request::CheckPrism { source }) => json!({ "ok": check_prism(&source) }),
        Ok(Request::CheckJani { source }) => json!({ "ok": check_jani(&source) }),
        Ok(Request::Analyse { source, max_states }) => {
            respond(analyse(&source, options(max_states)))
        }
        Ok(Request::AnalysePrism { source, max_states }) => {
            respond(analyse_prism(&source, options(max_states)))
        }
        Ok(Request::AnalyseJani { source, max_states }) => {
            respond(analyse_jani(&source, options(max_states)))
        }
        Ok(Request::Transient {
            source,
            times,
            max_states,
        }) => respond(transient(&source, &times, options(max_states))),
        Ok(Request::Compare { a, b, max_states }) => respond(compare(&a, &b, options(max_states))),
        Ok(Request::Observe {
            source,
            timeline,
            observations,
            max_states,
        }) => respond(check_observations(
            &source,
            &observations,
            &timeline,
            options(max_states),
        )),
        Ok(Request::Forecast {
            source,
            timeline,
            component,
            state,
            times,
            max_states,
        }) => respond(forecast(
            &source,
            component.as_deref(),
            &state,
            &times,
            &timeline,
            options(max_states),
        )),
        Ok(Request::Fit {
            source,
            timeline,
            observations,
            params,
            start,
            max_states,
        }) => respond(fit_parameters_from(
            &source,
            &observations,
            &params,
            start.as_deref(),
            &timeline,
            options(max_states),
        )),
        Ok(Request::Query {
            source,
            query,
            max_states,
        }) => respond(query_prism(&source, &query, options(max_states))),
        Ok(Request::QueryJani {
            source,
            query,
            max_states,
        }) => respond(query_jani(&source, &query, options(max_states))),
        Ok(Request::SessionOpen { session }) => respond(crate::session::open(session, options)),
        Ok(Request::SessionPush { id, events }) => respond(crate::session::push(id, events)),
        Ok(Request::SessionClose { id }) => respond(crate::session::close(id)),
        Ok(Request::Distance {
            a,
            b,
            observe,
            half_life,
            max_states,
        }) => respond(distance(&a, &b, &observe, half_life, options(max_states))),
        Ok(Request::Epsdelta {
            a,
            b,
            labels,
            epsilon,
            delta,
            horizon,
            max_states,
        }) => respond(epsdelta(
            &a,
            &b,
            &labels,
            epsilon,
            delta,
            horizon,
            options(max_states),
        )),
        Ok(Request::StreamDistance {
            a,
            b,
            values,
            gamma,
        }) => respond(stream_distance(&a, &b, &values, gamma)),
        Ok(Request::Dsa { x, y, delays, rank }) => respond(dsa(&x, &y, delays, rank)),
        Ok(Request::WhatIf {
            source,
            timeline,
            component,
            state,
            times,
            interventions,
            max_states,
        }) => respond(what_if(
            &source,
            &interventions,
            component.as_deref(),
            &state,
            &times,
            &timeline,
            options(max_states),
        )),
        Ok(Request::Attribute {
            plan,
            actual,
            timeline,
            component,
            state,
            at,
            max_states,
        }) => respond(attribute(
            &plan,
            &actual,
            component.as_deref(),
            &state,
            at,
            &timeline,
            options(max_states),
        )),
        Ok(Request::Fluid {
            source,
            times,
            dt,
            max_states,
        }) => respond(fluid(
            &source,
            &times,
            dt.unwrap_or(0.01),
            options(max_states),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let r: Value = serde_json::from_str(&handle_json(
            r#"{"op":"analyse","source":"P = (a, 1).Q; Q = (b, 2).P; P"}"#,
        ))
        .unwrap();
        assert_eq!(r["ok"]["full"]["states"].as_array().unwrap().len(), 2);

        let r: Value = serde_json::from_str(&handle_json(
            r#"{"op":"analyse_prism","source":"dtmc module m x : [0..1] init 0; [] x=0 -> 1 : (x'=1); [] x=1 -> 1 : (x'=0); endmodule"}"#,
        ))
        .unwrap();
        assert_eq!(r["ok"]["full"]["states"].as_array().unwrap().len(), 2);
        assert_eq!(r["ok"]["vars"], serde_json::json!(["x"]));

        let r: Value =
            serde_json::from_str(&handle_json(r#"{"op":"check","source":"P = (a, r).P; P"}"#))
                .unwrap();
        assert_eq!(r["ok"][0]["column"], 9);

        let r: Value = serde_json::from_str(&handle_json(
            r#"{"op":"check_prism","source":"dtmc module m x : [0..1] init 0; endmodule"}"#,
        ))
        .unwrap();
        assert_eq!(r["ok"], serde_json::json!([]));

        let r: Value = serde_json::from_str(&handle_json(r#"{"op":"nope"}"#)).unwrap();
        assert!(
            r["error"][0]["message"]
                .as_str()
                .unwrap()
                .contains("invalid request")
        );
    }
}
