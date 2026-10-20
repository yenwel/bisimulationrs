use bisim_api::{
    Intervention, MAX_ATTRIBUTION_PARAMS, Options, TimeUnit, Timeline, attribute, what_if,
};

/// Ten copies of `P -> Q -> R` with rates `ra` and `rb` (per day), plus a
/// parameter `idle` that no rate uses.
fn chain(ra: f64, rb: f64, idle: f64) -> String {
    format!(
        "ra = {ra}; rb = {rb}; idle = {idle};
         P = (a, ra).Q; Q = (b, rb).R; R = 0;
         P[10]"
    )
}

fn timeline() -> Timeline {
    Timeline {
        start: "01/01/2026".to_owned(),
        unit: TimeUnit::Day,
        level: 0.95,
    }
}

/// Expected number of the ten copies in `R` at time `t`.
fn expected_r(ra: f64, rb: f64, t: f64) -> f64 {
    10.0 * (1.0 - (rb * (-ra * t).exp() - ra * (-rb * t).exp()) / (rb - ra))
}

#[test]
fn a_single_changed_rate_gets_the_whole_gap() {
    let t = 6.0;
    let a = attribute(
        &chain(0.2, 0.5, 1.0),
        &chain(0.1, 0.5, 1.0),
        None,
        "R",
        t,
        &timeline(),
        Options::default(),
    )
    .unwrap();
    assert!((a.plan - expected_r(0.2, 0.5, t)).abs() < 1e-6, "{a:?}");
    assert!((a.actual - expected_r(0.1, 0.5, t)).abs() < 1e-6, "{a:?}");
    assert_eq!(a.contributions.len(), 1);
    let c = &a.contributions[0];
    assert_eq!(c.name, "ra");
    assert!((c.shapley - a.gap).abs() < 1e-9);
    assert!((c.share.unwrap() - 1.0).abs() < 1e-9);
    assert!(a.gap < 0.0, "slower first step means fewer in R");
    assert!(
        a.explanation.contains("within the model"),
        "{}",
        a.explanation
    );
}

#[test]
fn shapley_values_match_the_closed_form_and_add_up() {
    let (plan, actual, t) = ((0.2, 0.5), (0.1, 0.3), 6.0);
    // Three rates differ, but `idle` is used by nothing: a dummy player.
    let a = attribute(
        &chain(plan.0, plan.1, 1.0),
        &chain(actual.0, actual.1, 7.0),
        None,
        "R",
        t,
        &timeline(),
        Options::default(),
    )
    .unwrap();
    let v = |ra_switched: bool, rb_switched: bool| {
        expected_r(
            if ra_switched { actual.0 } else { plan.0 },
            if rb_switched { actual.1 } else { plan.1 },
            t,
        )
    };
    // Two real players: average of the two orders of switching.
    let shapley_a = 0.5 * ((v(true, false) - v(false, false)) + (v(true, true) - v(false, true)));
    let shapley_b = 0.5 * ((v(false, true) - v(false, false)) + (v(true, true) - v(true, false)));
    let get = |name: &str| a.contributions.iter().find(|c| c.name == name).unwrap();
    assert!((get("ra").shapley - shapley_a).abs() < 1e-6, "{a:?}");
    assert!((get("rb").shapley - shapley_b).abs() < 1e-6, "{a:?}");
    assert!(get("idle").shapley.abs() < 1e-9, "dummy player: {a:?}");
    // Efficiency: the parts add up to the gap, whatever the interaction.
    let total: f64 = a.contributions.iter().map(|c| c.shapley).sum();
    assert!((total - a.gap).abs() < 1e-9, "{total} vs {}", a.gap);
    assert!((a.gap - (v(true, true) - v(false, false))).abs() < 1e-6);
    // Largest contribution first.
    assert!(a.contributions[0].shapley.abs() >= a.contributions[1].shapley.abs());
    // The two rates interact, so "alone" effects do not add up to the gap.
    let alone: f64 = a.contributions.iter().map(|c| c.alone).sum();
    assert!((alone - a.gap).abs() > 1e-3, "{alone} vs {}", a.gap);
}

#[test]
fn models_that_differ_in_more_than_rates_are_refused() {
    let plan = chain(0.2, 0.5, 1.0);
    // An extra stage: same parameters, different structure.
    let actual = "ra = 0.1; rb = 0.5; idle = 1;
        P = (a, ra).Q; Q = (b, rb).S; S = (c, 1).R; R = 0;
        P[10]";
    let e = attribute(
        &plan,
        actual,
        None,
        "R",
        6.0,
        &timeline(),
        Options::default(),
    )
    .unwrap_err();
    assert!(e[0].message.contains("more than"), "{}", e[0].message);
    // Different parameter names.
    let other = "k = 0.1; P = (a, k).R; R = 0; P[10]";
    let e = attribute(
        &plan,
        other,
        None,
        "R",
        6.0,
        &timeline(),
        Options::default(),
    )
    .unwrap_err();
    assert!(e[0].message.contains("same parameters"), "{}", e[0].message);
}

#[test]
fn identical_models_have_nothing_to_attribute() {
    let a = attribute(
        &chain(0.2, 0.5, 1.0),
        &chain(0.2, 0.5, 1.0),
        None,
        "R",
        6.0,
        &timeline(),
        Options::default(),
    )
    .unwrap();
    assert!(a.contributions.is_empty());
    assert!(a.gap.abs() < 1e-9);
    assert!(a.explanation.contains("nothing to attribute"));
}

#[test]
fn what_if_matches_the_closed_form() {
    let t = 6.0;
    let w = what_if(
        &chain(0.2, 0.5, 1.0),
        &[Intervention {
            param: "ra".to_owned(),
            value: 0.4,
        }],
        None,
        "R",
        &[t],
        &timeline(),
        Options::default(),
    )
    .unwrap();
    assert!((w.baseline.expected[0] - expected_r(0.2, 0.5, t)).abs() < 1e-6);
    assert!((w.intervened.expected[0] - expected_r(0.4, 0.5, t)).abs() < 1e-6);
    assert!((w.effect[0] - (expected_r(0.4, 0.5, t) - expected_r(0.2, 0.5, t))).abs() < 1e-6);
    assert!(w.explanation.contains("model"), "{}", w.explanation);
}

#[test]
fn bad_interventions_are_reported() {
    let src = chain(0.2, 0.5, 1.0);
    let t = timeline();
    let go = |iv: Vec<Intervention>| what_if(&src, &iv, None, "R", &[3.0], &t, Options::default());
    let e = go(vec![]).unwrap_err();
    assert!(e[0].message.contains("at least one"));
    let e = go(vec![Intervention {
        param: "nope".into(),
        value: 1.0,
    }])
    .unwrap_err();
    assert!(
        e[0].message.contains("no parameter `nope`"),
        "{}",
        e[0].message
    );
    let e = go(vec![Intervention {
        param: "ra".into(),
        value: f64::NAN,
    }])
    .unwrap_err();
    assert!(e[0].message.contains("finite"), "{}", e[0].message);
}

#[test]
fn too_many_differing_rates_are_refused() {
    let n = MAX_ATTRIBUTION_PARAMS + 1;
    let model = |scale: f64| {
        let params: String = (0..n)
            .map(|i| format!("r{i} = {};", (i + 1) as f64 * scale))
            .collect();
        format!("{params} P = (a, r0).R; R = 0; P[2]")
    };
    let e = attribute(
        &model(1.0),
        &model(2.0),
        None,
        "R",
        1.0,
        &timeline(),
        Options::default(),
    )
    .unwrap_err();
    assert!(e[0].message.contains("at most"), "{}", e[0].message);
}

#[test]
fn json_operations_expose_what_if_and_attribute() {
    use bisim_api::handle_json;
    let parse = |s: String| -> serde_json::Value { serde_json::from_str(&s).unwrap() };
    let tl = r#"{"start":"01/01/2026","unit":"day"}"#;
    let attr = parse(handle_json(&format!(
        r#"{{"op":"attribute","plan":{},"actual":{},"timeline":{tl},"state":"R","at":6}}"#,
        serde_json::to_string(&chain(0.2, 0.5, 1.0)).unwrap(),
        serde_json::to_string(&chain(0.1, 0.5, 1.0)).unwrap(),
    )));
    let ok = &attr["ok"];
    assert!(ok["gap"].as_f64().unwrap() < 0.0, "{attr}");
    assert_eq!(ok["contributions"][0]["name"], "ra");
    assert_eq!(attr["version"], 1);
    let what = parse(handle_json(&format!(
        r#"{{"op":"whatif","source":{},"timeline":{tl},"state":"R","times":[6],
            "interventions":[{{"param":"ra","value":0.4}}]}}"#,
        serde_json::to_string(&chain(0.2, 0.5, 1.0)).unwrap(),
    )));
    assert!(what["ok"]["effect"][0].as_f64().unwrap() > 0.0, "{what}");
    // Unknown fields are rejected here too.
    let bad = handle_json(&format!(
        r#"{{"op":"attribute","plan":"x","actual":"y","timeline":{tl},"state":"R","at":6,"att":1}}"#
    ));
    assert!(bad.contains("unknown field"), "{bad}");
}

/// Three interacting rates, checked against a different computation: the
/// average, over all 3! orders of switching the rates one by one, of the
/// change each switch makes. It uses only plain forecasts of models with the
/// values substituted, not the subset-weight formula inside `attribute`.
#[test]
fn shapley_agrees_with_averaging_over_all_orders() {
    use bisim_api::forecast;
    let model = |ra: f64, rb: f64, rc: f64| {
        format!(
            "ra = {ra}; rb = {rb}; rc = {rc};
             P = (a, ra).Q; Q = (b, rb).S; S = (c, rc).R; R = 0;
             P[10]"
        )
    };
    let (plan, actual, t) = ([0.3, 0.5, 0.8], [0.1, 0.9, 0.4], 7.0);
    let outcome = |on: [bool; 3]| {
        let v = |i: usize| if on[i] { actual[i] } else { plan[i] };
        forecast(
            &model(v(0), v(1), v(2)),
            None,
            "R",
            &[t],
            &timeline(),
            Options::default(),
        )
        .unwrap()
        .expected[0]
    };
    let mut by_orders = [0.0f64; 3];
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for order in orders {
        let mut on = [false; 3];
        for &i in &order {
            let before = outcome(on);
            on[i] = true;
            by_orders[i] += (outcome(on) - before) / orders.len() as f64;
        }
    }
    let a = attribute(
        &model(plan[0], plan[1], plan[2]),
        &model(actual[0], actual[1], actual[2]),
        None,
        "R",
        t,
        &timeline(),
        Options::default(),
    )
    .unwrap();
    for (i, name) in ["ra", "rb", "rc"].iter().enumerate() {
        let c = a.contributions.iter().find(|c| c.name == *name).unwrap();
        assert!(
            (c.shapley - by_orders[i]).abs() < 1e-9,
            "{name}: {} vs {}",
            c.shapley,
            by_orders[i]
        );
    }
    let total: f64 = by_orders.iter().sum();
    assert!((total - a.gap).abs() < 1e-9);
    // The rates really do interact here (otherwise this proves little).
    let alone: f64 = a.contributions.iter().map(|c| c.alone).sum();
    assert!((alone - a.gap).abs() > 1e-3, "{alone} vs {}", a.gap);
}
