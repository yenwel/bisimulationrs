//! Fixed-step RK4 integration for fluid (mean-field) ODEs.
//!
//! [`integrate`] advances `dx/dt = f(x)` through requested times with
//! classic fourth-order Runge–Kutta and internal substeps of at most
//! `dt`. Times need not be sorted; the result aligns with the input
//! order. No positivity is enforced: with sensible `dt`, population
//! drifts stay non-negative; a negative value signals `dt` is too large.

/// Integrates from `x0` at time 0 through `times`, evaluating `f` with
/// substeps no larger than `dt`. Returns one state per requested time.
pub fn integrate<F, E>(f: F, x0: &[f64], times: &[f64], dt: f64) -> Result<Vec<Vec<f64>>, E>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, E>,
{
    assert!(dt > 0.0 && dt.is_finite(), "dt must be positive and finite");
    let mut order: Vec<usize> = (0..times.len()).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
    let mut out = vec![Vec::new(); times.len()];
    let (mut now, mut x) = (0.0, x0.to_vec());
    for i in order {
        let target = times[i];
        let span = target - now;
        if span != 0.0 {
            let steps = (span.abs() / dt).ceil().max(1.0) as usize;
            let h = span / steps as f64;
            for _ in 0..steps {
                x = rk4_step(&f, &x, h)?;
            }
            now = target;
        }
        out[i] = x.clone();
    }
    Ok(out)
}

fn rk4_step<F, E>(f: F, x: &[f64], h: f64) -> Result<Vec<f64>, E>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, E>,
{
    let n = x.len();
    let k1 = f(x)?;
    let x2: Vec<f64> = x.iter().zip(&k1).map(|(x, k)| x + h / 2.0 * k).collect();
    let k2 = f(&x2)?;
    let x3: Vec<f64> = x.iter().zip(&k2).map(|(x, k)| x + h / 2.0 * k).collect();
    let k3 = f(&x3)?;
    let x4: Vec<f64> = x.iter().zip(&k3).map(|(x, k)| x + h * k).collect();
    let k4 = f(&x4)?;
    Ok((0..n)
        .map(|i| x[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_decay_matches_closed_form() {
        // dx/dt = -0.5 x from x0 = 4.
        let f = |x: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![-0.5 * x[0]]) };
        let xs = integrate(f, &[4.0], &[0.0, 1.0, 7.0], 0.05).unwrap();
        for (x, t) in xs.iter().zip([0.0f64, 1.0, 7.0]) {
            assert!((x[0] - 4.0 * (-0.5 * t).exp()).abs() < 1e-6, "t={t}");
        }
    }

    #[test]
    fn halving_step_size_cuts_error_sixteenfold() {
        // Fourth-order convergence on dx/dt = x: the signature of RK4.
        let f = |x: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![x[0]]) };
        let err =
            |dt: f64| integrate(f, &[1.0], &[2.0], dt).unwrap()[0][0] - std::f64::consts::E.powi(2);
        let ratio = err(0.2).abs() / err(0.1).abs();
        assert!((8.0..32.0).contains(&ratio), "error ratio {ratio}");
    }

    #[test]
    fn errors_propagate_and_order_survives_sorting() {
        let f = |_: &[f64]| -> Result<Vec<f64>, String> { Err("boom".to_owned()) };
        assert_eq!(integrate(f, &[1.0], &[1.0], 0.1), Err("boom".to_owned()));
        let g = |x: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![x[0]]) };
        let xs = integrate(g, &[1.0], &[2.0, 0.0, 1.0], 0.05).unwrap();
        assert!((xs[0][0] - 7.389_056_098_9).abs() < 1e-6);
        assert_eq!(xs[1], vec![1.0]);
    }
}
