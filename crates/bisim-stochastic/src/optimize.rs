/// Minimises `f` with the Nelder–Mead simplex method (no derivatives needed).
///
/// `x0` is the starting point and `step` the initial simplex size per
/// coordinate. Returns the best point and its value after at most
/// `max_evaluations` calls, or earlier once the simplex values agree to
/// within `tolerance`.
pub fn nelder_mead(
    mut f: impl FnMut(&[f64]) -> f64,
    x0: &[f64],
    step: f64,
    max_evaluations: usize,
    tolerance: f64,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut evals = 0usize;
    let mut eval = |x: &[f64], evals: &mut usize| {
        *evals += 1;
        let v = f(x);
        if v.is_nan() { f64::INFINITY } else { v }
    };
    if n == 0 {
        let v = eval(x0, &mut evals);
        return (Vec::new(), v);
    }

    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    simplex.push((x0.to_vec(), eval(x0, &mut evals)));
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += step;
        let v = eval(&x, &mut evals);
        simplex.push((x, v));
    }

    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    while evals < max_evaluations {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (best, worst) = (simplex[0].1, simplex[n].1);
        if (worst - best).abs() <= tolerance * (1.0 + best.abs()) {
            break;
        }
        let centroid: Vec<f64> = (0..n)
            .map(|j| simplex[..n].iter().map(|(x, _)| x[j]).sum::<f64>() / n as f64)
            .collect();
        let towards = |coef: f64, from: &[f64]| -> Vec<f64> {
            centroid
                .iter()
                .zip(from)
                .map(|(c, w)| c + coef * (c - w))
                .collect()
        };

        let reflected = towards(alpha, &simplex[n].0);
        let fr = eval(&reflected, &mut evals);
        if fr < simplex[0].1 {
            let expanded = towards(gamma, &simplex[n].0);
            let fe = eval(&expanded, &mut evals);
            simplex[n] = if fe < fr {
                (expanded, fe)
            } else {
                (reflected, fr)
            };
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (reflected, fr);
        } else {
            let contracted = towards(-rho, &simplex[n].0);
            let fc = eval(&contracted, &mut evals);
            if fc < simplex[n].1 {
                simplex[n] = (contracted, fc);
            } else {
                // Shrink everything towards the best point.
                let best_x = simplex[0].0.clone();
                for (x, v) in simplex.iter_mut().skip(1) {
                    for (xi, bi) in x.iter_mut().zip(&best_x) {
                        *xi = bi + sigma * (*xi - bi);
                    }
                    *v = eval(x, &mut evals);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (x, v) = simplex.swap_remove(0);
    (x, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_minimum_of_rosenbrock() {
        let (x, v) = nelder_mead(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[-1.2, 1.0],
            0.5,
            5_000,
            1e-14,
        );
        assert!(v < 1e-8, "value {v}");
        assert!((x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3);
    }
}
