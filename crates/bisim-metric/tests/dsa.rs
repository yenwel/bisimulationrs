//! Dynamical similarity analysis: a Procrustes oracle, invariance under a
//! change of sensor coordinates, and separation of different dynamics.

use bisim_metric::linalg::Mat;
use bisim_metric::{DsaOptions, dsa, procrustes};

/// `Q·diag(d)·Qᵀ` for a fixed rotation `Q` in the first two coordinates.
fn rotated_diagonal(d: &[f64], angle: f64) -> Mat {
    let r = d.len();
    let mut q = Mat::identity(r);
    let (c, s) = (angle.cos(), angle.sin());
    q[(0, 0)] = c;
    q[(0, 1)] = -s;
    q[(1, 0)] = s;
    q[(1, 1)] = c;
    let mut dm = Mat::zeros(r, r);
    for (i, &v) in d.iter().enumerate() {
        dm[(i, i)] = v;
    }
    q.mul(&dm).mul(&q.transpose())
}

#[test]
fn procrustes_matches_hoffman_wielandt_for_symmetric_matrices() {
    // For symmetric matrices the minimum is ‖sorted eigenvalue gaps‖.
    let a = rotated_diagonal(&[1.0, 2.0, 3.0], 0.0);
    let b = rotated_diagonal(&[2.5, 1.5, 2.0], 0.7);
    let (d, c) = procrustes(&a, &b, 8);
    let want = (0.25f64 + 0.0 + 0.25).sqrt();
    assert!((d - want).abs() < 1e-6, "{d} vs {want}");
    assert!(
        c.transpose().mul(&c).sub(&Mat::identity(3)).norm() < 1e-9,
        "C is orthogonal"
    );
}

/// A damped oscillation at frequency `omega` in two channels, plus a slow
/// decay in a third.
fn series(omega: f64, steps: usize) -> Vec<Vec<f64>> {
    (0..steps)
        .map(|t| {
            let t = t as f64 * 0.1;
            vec![
                (-0.05 * t).exp() * (omega * t).cos(),
                (-0.05 * t).exp() * (omega * t).sin(),
                (-0.2 * t).exp(),
            ]
        })
        .collect()
}

#[test]
fn a_change_of_sensor_coordinates_does_not_matter() {
    let x = series(1.0, 400);
    // The same system seen through rotated, reflected sensors.
    let (c, s) = (0.6f64, 0.8f64);
    let y: Vec<Vec<f64>> = x
        .iter()
        .map(|v| vec![c * v[0] - s * v[2], -v[1], s * v[0] + c * v[2]])
        .collect();
    let opts = DsaOptions {
        delays: 2,
        rank: 3,
        ..DsaOptions::default()
    };
    let same = dsa(&x, &y, opts).unwrap();
    assert!(same.distance < 1e-4, "{same:?}");
    let other = dsa(&x, &series(2.0, 400), opts).unwrap();
    assert!(
        other.distance > 100.0 * same.distance.max(1e-6),
        "{} vs {}",
        other.distance,
        same.distance
    );
    assert!(other.angle > same.angle);
}

#[test]
fn bad_inputs_are_rejected() {
    let opts = DsaOptions::default();
    assert!(dsa(&[], &series(1.0, 50), opts).is_err());
    assert!(dsa(&vec![vec![f64::NAN; 3]; 50], &series(1.0, 50), opts).is_err());
    let too_high = DsaOptions {
        delays: 1,
        rank: 4,
        ..opts
    };
    assert!(dsa(&series(1.0, 50), &series(1.0, 50), too_high).is_err());
}
