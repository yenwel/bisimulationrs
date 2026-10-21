//! Behavioural distances through the API and the JSON protocol.

use std::collections::HashMap;

use bisim_api::{Atom, Options, distance, dsa, epsdelta, handle_json, stream_distance};

fn herd(infect: f64, recover: f64) -> String {
    format!(
        "infect = {infect}; recover = {recover};
         Healthy = (infect, infect).Sick;
         Sick = (recover, recover).Healthy;
         Healthy[4]"
    )
}

fn sick() -> Atom {
    Atom {
        component: None,
        state: "Sick".to_owned(),
    }
}

#[test]
fn distance_is_zero_for_the_same_model_and_grows_with_the_change() {
    let o = Options::default();
    let same = distance(&herd(0.1, 0.5), &herd(0.1, 0.5), &sick(), 5.0, o).unwrap();
    assert!(same.distance < 1e-6, "{same:?}");
    let near = distance(&herd(0.1, 0.5), &herd(0.12, 0.5), &sick(), 5.0, o).unwrap();
    let far = distance(&herd(0.1, 0.5), &herd(0.3, 0.5), &sick(), 5.0, o).unwrap();
    assert!(
        0.0 < near.distance && near.distance < far.distance,
        "{} {}",
        near.distance,
        far.distance
    );
    assert!(!far.pairs.is_empty() && far.explanation.contains("half"));
    assert_eq!(far.scale, 4.0);
}

#[test]
fn epsdelta_tolerates_a_uniform_speed_up_within_delta() {
    let o = Options::default();
    // All rates 5 % faster: same jump probabilities, exit rates within e^δ.
    let r = epsdelta(
        &herd(0.1, 0.5),
        &herd(0.105, 0.525),
        &[sick()],
        Some(0.0),
        0.1,
        Some(10.0),
        o,
    )
    .unwrap();
    assert_eq!(r.min_epsilon, Some(0.0));
    assert_eq!(r.within, Some(true));
    let bound = r.reach_bound.unwrap();
    assert!(bound > 0.0 && bound < 1.0, "{r:?}");
    // A change in the balance of rates needs ε > 0.
    let r = epsdelta(
        &herd(0.1, 0.5),
        &herd(0.2, 0.5),
        &[sick()],
        Some(0.01),
        0.7,
        None,
        o,
    )
    .unwrap();
    assert!(r.min_epsilon.unwrap() > 0.01, "{r:?}");
    assert_eq!(r.within, Some(false));
}

#[test]
fn stream_distance_and_dsa_through_json() {
    let log = |n: usize| -> Vec<String> {
        (0..n)
            .map(|i| if (i / 3) % 2 == 0 { "ok" } else { "sick" }.to_owned())
            .collect()
    };
    let values = HashMap::from([("ok".to_owned(), 0.0), ("sick".to_owned(), 1.0)]);
    let r = stream_distance(&log(200), &log(200), &values, 0.8).unwrap();
    assert_eq!(r.steps, 199);
    assert!(
        r.matching.contains(&("ok".to_owned(), "ok".to_owned())),
        "{r:?}"
    );

    let wave: Vec<Vec<f64>> = (0..200)
        .map(|t| {
            let t = t as f64 * 0.1;
            vec![t.sin(), t.cos()]
        })
        .collect();
    let d = dsa(&wave, &wave, Some(2), Some(2)).unwrap();
    assert!(d.distance < 1e-6, "{d:?}");

    let req = serde_json::json!({
        "op": "distance", "a": herd(0.1, 0.5), "b": herd(0.3, 0.5),
        "observe": {"state": "Sick"}, "half_life": 5.0
    });
    let resp: serde_json::Value = serde_json::from_str(&handle_json(&req.to_string())).unwrap();
    assert!(resp["ok"]["distance"].as_f64().unwrap() > 0.0, "{resp}");

    let req = serde_json::json!({
        "op": "epsdelta", "a": herd(0.1, 0.5), "b": herd(0.105, 0.525),
        "labels": [{"state": "Sick"}], "delta": 0.1, "horizon": 10.0
    });
    let resp: serde_json::Value = serde_json::from_str(&handle_json(&req.to_string())).unwrap();
    assert_eq!(resp["ok"]["min_epsilon"].as_f64(), Some(0.0), "{resp}");

    let req = serde_json::json!({
        "op": "stream_distance", "a": log(50), "b": log(50),
        "values": {"ok": 0.0, "sick": 1.0}, "gamma": 0.8
    });
    let resp: serde_json::Value = serde_json::from_str(&handle_json(&req.to_string())).unwrap();
    assert!(resp["ok"]["estimate"].is_number(), "{resp}");

    let req = serde_json::json!({ "op": "dsa", "x": wave, "y": wave, "delays": 2, "rank": 2 });
    let resp: serde_json::Value = serde_json::from_str(&handle_json(&req.to_string())).unwrap();
    assert!(resp["ok"]["distance"].as_f64().unwrap() < 1e-6, "{resp}");

    // Errors come back as diagnostics.
    let req = serde_json::json!({
        "op": "distance", "a": herd(0.1, 0.5), "b": herd(0.1, 0.5),
        "observe": {"state": "Nope"}, "half_life": 5.0
    });
    let resp: serde_json::Value = serde_json::from_str(&handle_json(&req.to_string())).unwrap();
    assert!(
        resp["error"][0]["message"]
            .as_str()
            .unwrap()
            .contains("Nope"),
        "{resp}"
    );
}
