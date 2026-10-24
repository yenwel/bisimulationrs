//! The learned distance against the exact one on small chains.

use bisim_metric::{Chain, LearnOptions, MetricOptions, distance, learn_metric};

fn ring(k: usize, p: f64) -> Chain {
    Chain {
        rows: (0..k)
            .map(|s| vec![(s, 1.0 - p), ((s + 1) % k, p)])
            .collect(),
        observation: (0..k).map(|s| if s < k / 2 { 1.0 } else { 0.0 }).collect(),
        initial: 0,
    }
}

fn one_hot(i: usize, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    v[i] = 1.0;
    v
}

#[test]
fn learned_distance_tracks_the_exact_one() {
    let (g, k) = (0.8, 4);
    let opts = LearnOptions {
        batch: 32,
        ..LearnOptions::default()
    };
    let truth = ring(k, 0.5);
    let fa: Vec<Vec<f64>> = (0..k).map(|i| one_hot(i, 2 * k)).collect();
    let fb: Vec<Vec<f64>> = (0..k).map(|i| one_hot(k + i, 2 * k)).collect();
    let mut at_start = Vec::new();
    for other in [ring(k, 0.5), ring(k, 0.9)] {
        let m = learn_metric(&truth, &other, &fa, &fb, g, opts).unwrap();
        let mut errors = Vec::new();
        for (x, fx) in fa.iter().enumerate() {
            for (y, fy) in fb.iter().enumerate() {
                let (mut a, mut b) = (truth.clone(), other.clone());
                (a.initial, b.initial) = (x, y);
                let exact = distance(&a, &b, g, MetricOptions::default()).unwrap().value;
                errors.push((m.distance(fx, fy) - exact).abs());
            }
        }
        let mae = errors.iter().sum::<f64>() / errors.len() as f64;
        println!(
            "mean error {mae:.3}, worst {:.3}",
            errors.iter().copied().fold(0.0, f64::max)
        );
        assert!(mae < 0.1, "{mae}");
        at_start.push(m.distance(&fa[0], &fb[0]));
    }
    // The same chain is learned as closer than a different one.
    assert!(at_start[0] < at_start[1], "{at_start:?}");
}

#[test]
fn bad_inputs_are_rejected() {
    let a = ring(3, 0.5);
    let f: Vec<Vec<f64>> = (0..3).map(|i| one_hot(i, 3)).collect();
    assert!(learn_metric(&a, &a, &f, &f, 1.0, LearnOptions::default()).is_err());
    assert!(learn_metric(&a, &a, &f[..2], &f, 0.8, LearnOptions::default()).is_err());
    let nan = vec![vec![f64::NAN; 3]; 3];
    assert!(learn_metric(&a, &a, &nan, &f, 0.8, LearnOptions::default()).is_err());
}
