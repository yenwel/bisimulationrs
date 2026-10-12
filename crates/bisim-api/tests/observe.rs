use bisim_api::{
    Observation, Options, Status, TimeUnit, Timeline, analyse, check_observations, fit_parameters,
    forecast, transient,
};

const GARDEN: &str = include_str!("../../../examples/garden/tomatoes.pepa");

fn timeline() -> Timeline {
    Timeline {
        start: "15/05/2026".into(),
        unit: TimeUnit::Day,
        level: 0.95,
    }
}

fn obs(date: &str, state: &str, count: u32) -> Observation {
    Observation {
        date: date.into(),
        component: None,
        state: state.into(),
        count,
    }
}

fn season() -> Vec<Observation> {
    vec![
        obs("05/06/2026", "Vegetative", 17),
        obs("05/06/2026", "Flowering", 5),
        obs("19/06/2026", "Vegetative", 12),
        obs("19/06/2026", "Fruiting", 5),
        obs("03/07/2026", "Vegetative", 8),
        obs("03/07/2026", "Fruiting", 8),
        obs("17/07/2026", "Vegetative", 6),
        obs("17/07/2026", "Ripe", 1),
    ]
}

#[test]
fn mid_season_is_borderline_and_july_is_off_plan() {
    let june = check_observations(GARDEN, &season()[..4], &timeline(), Options::default()).unwrap();
    assert!(june.on_track, "{}", june.explanation);
    assert_eq!(june.outside_range, 1);
    assert_eq!(june.rows[2].status, Status::AbovePlan);
    assert!(june.explanation.starts_with("Still consistent"));
    assert_eq!(
        june.methods,
        vec!["Tomato[24]: binomial over 24 independent copies"]
    );

    let july = check_observations(GARDEN, &season(), &timeline(), Options::default()).unwrap();
    assert!(!july.on_track);
    assert_eq!(july.outside_range, 4);
    assert!((july.threshold - 0.05 / 8.0).abs() < 1e-12);
    let ripe = &july.rows[7];
    assert_eq!(
        (ripe.status, ripe.low, ripe.high),
        (Status::BelowPlan, 4, 13)
    );
    assert!(july.explanation.contains("Ripe on 17/07/2026"));
}

#[test]
fn fitting_finds_the_slow_stage() {
    let fit = fit_parameters(
        GARDEN,
        &season(),
        &["veg_days".to_string()],
        &timeline(),
        Options::default(),
    )
    .unwrap();
    let veg = &fit.params[0];
    assert!(
        veg.fitted > 35.0 && veg.fitted < 55.0,
        "fitted {}",
        veg.fitted
    );
    assert!(fit.fitted_log_likelihood > fit.planned_log_likelihood);
    assert!(fit.fitted_source.contains("veg_days = "));
    // The fitted model is a valid model text.
    assert!(bisim_api::check(&fit.fitted_source).is_empty());
}

#[test]
fn forecast_band_and_dates() {
    let f = forecast(
        GARDEN,
        None,
        "Flowering",
        &[0.0, 21.0],
        &timeline(),
        Options::default(),
    )
    .unwrap();
    assert_eq!(f.dates, vec!["15/05/2026", "05/06/2026"]);
    assert_eq!((f.low[0], f.high[0], f.copies), (0, 0, 24));
    assert!(f.low[1] <= f.expected[1].floor() as u32 && f.expected[1] <= f.high[1] as f64);
}

#[test]
fn independent_and_full_chain_agree() {
    // An always-ready passive partner changes nothing, but forces the full chain.
    let plain = "Tomato = (grow, 0.1).Big + (lose, 0.01).Lost; Big = (lose, 0.01).Lost; Lost = 0; Tomato[4]";
    let coupled = "Tomato = (grow, 0.1).Big + (lose, 0.01).Lost; Big = (lose, 0.01).Lost; Lost = 0;
                   Gardener = (grow, infty).Gardener; Tomato[4] <grow> Gardener";
    let o = vec![
        Observation {
            component: Some("Tomato".into()),
            ..obs("25/05/2026", "Big", 2)
        },
        Observation {
            component: Some("Tomato".into()),
            ..obs("14/06/2026", "Lost", 1)
        },
    ];
    let a = check_observations(plain, &o, &timeline(), Options::default()).unwrap();
    let b = check_observations(coupled, &o, &timeline(), Options::default()).unwrap();
    assert!(a.methods[0].contains("binomial") && b.methods[0].contains("full Markov chain"));
    for (x, y) in a.rows.iter().zip(&b.rows) {
        assert!((x.expected - y.expected).abs() < 1e-8);
        assert!((x.p_value - y.p_value).abs() < 1e-8);
        assert_eq!((x.low, x.high), (y.low, y.high));
    }
}

#[test]
fn helpful_errors() {
    let bad = |o: Vec<Observation>| {
        check_observations(GARDEN, &o, &timeline(), Options::default()).unwrap_err()[0]
            .message
            .clone()
    };
    let unknown = bad(vec![obs("05/06/2026", "Blooming", 3)]);
    assert!(unknown.contains("has no state `Blooming`"));
    for stage in ["Vegetative", "Flowering", "Fruiting", "Ripe", "Lost"] {
        assert!(unknown.contains(stage), "{unknown}");
    }
    assert!(
        !unknown.contains("Vegetative_1"),
        "stages are listed without step suffixes"
    );
    assert!(bad(vec![obs("01/05/2026", "Ripe", 3)]).contains("before the start date"));
    assert!(bad(vec![obs("31/06/2026", "Ripe", 3)]).contains("dd/mm/yyyy"));
    assert!(bad(vec![obs("05/06/2026", "Ripe", 30)]).contains("only 24 copies"));
}

#[test]
fn large_independent_populations_skip_the_full_state_space() {
    let big = GARDEN.replace("Tomato[24]", "Tomato[500]");
    let a = analyse(&big, Options { max_states: 1_000 }).unwrap();
    assert!(a.note.is_some() && a.full.states.is_empty());
    let t = transient(&big, &[0.0, 50.0], Options { max_states: 1_000 }).unwrap();
    let total: f64 = t.occupancy.iter().map(|s| s.values[1]).sum();
    assert!((total - 500.0).abs() < 1e-6);
}
