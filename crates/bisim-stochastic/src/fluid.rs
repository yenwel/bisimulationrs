//! RK4 integration for fluid (mean-field) ODEs.
//!
//! [`integrate`] advances `dx/dt = f(x)` through requested times with
//! fourth-order Runge–Kutta and step-size control: every step is compared
//! with two half steps, rejected and halved while they disagree by more than
//! the tolerance, and grown again where the solution is smooth. `dt` is the
//! largest step. A fixed step cannot be trusted on stiff drifts (a rate of
//! 1000 s⁻¹ with `dt = 0.01` returned populations of −20), so the step
//! follows the solution instead. [`integrate_fixed`] is the plain fixed-step
//! scheme, kept for convergence studies. Times need not be sorted; the
//! result aligns with the input order.

/// Relative and absolute error tolerance of one adaptive step.
const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;

/// Integrates from `x0` at time 0 through `times`, with steps no larger than
/// `dt` and an error controlled to about 1e-8 per step. Returns one state
/// per requested time.
pub fn integrate<F, E>(f: F, x0: &[f64], times: &[f64], dt: f64) -> Result<Vec<Vec<f64>>, E>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, E>,
{
    assert!(dt > 0.0 && dt.is_finite(), "dt must be positive and finite");
    let mut order: Vec<usize> = (0..times.len()).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
    let mut out = vec![Vec::new(); times.len()];
    let (mut now, mut x) = (0.0, x0.to_vec());
    let mut h = dt;
    for i in order {
        let target = times[i];
        // Smallest step worth trying: below this the step is accepted as is.
        let h_min = (target - now).abs().max(1.0) * 1e-12;
        while now != target {
            let remaining = target - now;
            let step = h.min(remaining.abs()) * remaining.signum();
            let full = rk4_step(&f, &x, step)?;
            let half = rk4_step(&f, &x, step / 2.0)?;
            let two = rk4_step(&f, &half, step / 2.0)?;
            let err = x
                .iter()
                .zip(full.iter().zip(&two))
                .map(|(x0, (a, b))| {
                    let scale = ATOL + RTOL * x0.abs().max(b.abs());
                    (a - b).abs() / scale
                })
                .fold(0.0, f64::max);
            // Non-finite error: shrink and try again.
            let ok = err <= 1.0;
            if ok || step.abs() <= h_min {
                // Local extrapolation: the two-half-steps result plus its
                // estimated error.
                x = two
                    .iter()
                    .zip(&full)
                    .map(|(b, a)| b + (b - a) / 15.0)
                    .collect();
                now = if step.abs() >= remaining.abs() {
                    target
                } else {
                    now + step
                };
            }
            let factor = if err.is_finite() && err > 0.0 {
                (0.9 * err.powf(-0.2)).clamp(0.2, 2.0)
            } else if err == 0.0 {
                2.0
            } else {
                0.2
            };
            h = (step.abs() * factor).clamp(h_min, dt);
        }
        out[i] = x.clone();
    }
    Ok(out)
}

/// Fixed-step RK4: substeps of at most `dt`, no error control. Only
/// trustworthy when `dt` is small against every rate in the drift.
pub fn integrate_fixed<F, E>(f: F, x0: &[f64], times: &[f64], dt: f64) -> Result<Vec<Vec<f64>>, E>
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
        let err = |dt: f64| {
            integrate_fixed(f, &[1.0], &[2.0], dt).unwrap()[0][0] - std::f64::consts::E.powi(2)
        };
        let ratio = err(0.2).abs() / err(0.1).abs();
        assert!((8.0..32.0).contains(&ratio), "error ratio {ratio}");
    }

    #[test]
    fn stiff_drift_stays_non_negative_at_the_default_step() {
        // P -> Q at rate R per unit, Q -> P at 1, ten copies: the fixed
        // step is unstable for R = 1000 and dt = 0.01 (R·dt = 10 > 2.78).
        let rate = 1000.0;
        let f = move |x: &[f64]| -> Result<Vec<f64>, String> {
            let flow = rate * x[0] - x[1];
            Ok(vec![-flow, flow])
        };
        let fixed = integrate_fixed(f, &[10.0, 0.0], &[0.5], 0.01).unwrap();
        assert!(
            !(0.0..=10.0).contains(&fixed[0][0]),
            "fixed step: {:?}",
            fixed[0]
        );
        let xs = integrate(f, &[10.0, 0.0], &[0.0, 0.5, 5.0], 0.01).unwrap();
        // Equilibrium P = 10 / (1 + R), conserved total 10.
        let eq = 10.0 / (1.0 + rate);
        for x in &xs[1..] {
            assert!(x[0] >= -1e-9 && x[1] >= -1e-9, "{x:?}");
            assert!((x[0] - eq).abs() < 1e-6, "{x:?}");
            assert!((x[0] + x[1] - 10.0).abs() < 1e-9, "{x:?}");
        }
        assert_eq!(xs[0], vec![10.0, 0.0]);
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
