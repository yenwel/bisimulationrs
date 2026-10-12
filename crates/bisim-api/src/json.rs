//! A single JSON request/response entry point, used by the WebAssembly
//! module and the R package.
//!
//! Requests are objects with an `op` field:
//!
//! ```json
//! {"op": "check",     "source": "…"}
//! {"op": "analyse",   "source": "…", "max_states": 20000}
//! {"op": "transient", "source": "…", "times": [0, 1, 2]}
//! {"op": "compare",   "a": "…", "b": "…"}
//! {"op": "observe",   "source": "…", "timeline": {"start": "15/05/2026", "unit": "day"},
//!                     "observations": [{"date": "12/06/2026", "state": "Flowering", "count": 9}]}
//! {"op": "forecast",  "source": "…", "timeline": {…}, "state": "Flowering", "times": [0, 7, 14]}
//! {"op": "fit",       "source": "…", "timeline": {…}, "observations": […], "params": ["veg_days"]}
//! ```
//!
//! Responses are `{"ok": …}` or `{"error": …}`, where an error is a list of
//! diagnostics (for `compare`: `{"a": […], "b": […]}`).

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Observation, Options, Timeline, analyse, check, check_observations, compare, fit_parameters,
    forecast, transient,
};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Request {
    Check {
        source: String,
    },
    Analyse {
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
        max_states: Option<usize>,
    },
}

fn options(max_states: Option<usize>) -> Options {
    let mut o = Options::default();
    if let Some(m) = max_states {
        o.max_states = m;
    }
    o
}

fn respond<T: serde::Serialize, E: serde::Serialize>(r: Result<T, E>) -> Value {
    match r {
        Ok(v) => json!({ "ok": v }),
        Err(e) => json!({ "error": e }),
    }
}

/// Handles one JSON request and returns the JSON response.
pub fn handle_json(request: &str) -> String {
    let response = match serde_json::from_str::<Request>(request) {
        Err(e) => json!({ "error": [{ "severity": "error", "line": 0, "column": 0,
                                     "message": format!("invalid request: {e}") }] }),
        Ok(Request::Check { source }) => json!({ "ok": check(&source) }),
        Ok(Request::Analyse { source, max_states }) => {
            respond(analyse(&source, options(max_states)))
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
            max_states,
        }) => respond(fit_parameters(
            &source,
            &observations,
            &params,
            &timeline,
            options(max_states),
        )),
    };
    response.to_string()
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

        let r: Value =
            serde_json::from_str(&handle_json(r#"{"op":"check","source":"P = (a, r).P; P"}"#))
                .unwrap();
        assert_eq!(r["ok"][0]["column"], 9);

        let r: Value = serde_json::from_str(&handle_json(r#"{"op":"nope"}"#)).unwrap();
        assert!(
            r["error"][0]["message"]
                .as_str()
                .unwrap()
                .contains("invalid request")
        );
    }
}
