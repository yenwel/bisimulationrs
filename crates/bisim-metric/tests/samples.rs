//! The sample-based estimator against the exact distance, and model
//! selection over a parameter grid.

use bisim_metric::{Chain, MetricOptions, SampleDistance, StepSizes, distance, occupancy_sample};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A ring of `k` states, moving forward with probability `p`, staying
/// otherwise; observation 1 on the first half.
fn ring(k: usize, p: f64) -> Chain {
    Chain {
        rows: (0..k)
            .map(|s| vec![(s, 1.0 - p), ((s + 1) % k, p)])
            .collect(),
        observation: (0..k).map(|s| if s < k / 2 { 1.0 } else { 0.0 }).collect(),
        initial: 0,
    }
}

fn estimate(
    a: &Chain,
    b: &Chain,
    gamma: f64,
    steps: usize,
    sizes: StepSizes,
    rng: &mut Rng,
) -> f64 {
    let cost = |x: usize, y: usize| (a.observation[x] - b.observation[y]).abs();
    let mut est = SampleDistance::new(a.rows.len(), b.rows.len(), cost, gamma, (0, 0))
        .unwrap()
        .with_step_sizes(sizes);
    for _ in 0..steps {
        let sx = occupancy_sample(&a.rows, a.initial, gamma, || rng.uniform());
        let sy = occupancy_sample(&b.rows, b.initial, gamma, || rng.uniform());
        est.push(sx, sy);
    }
    est.estimate()
}

#[test]
fn estimate_approaches_the_exact_distance() {
    let gamma = 0.8;
    let (a, b) = (ring(4, 0.5), ring(4, 0.2));
    let exact = distance(&a, &b, gamma, MetricOptions::default())
        .unwrap()
        .value;
    let mut rng = Rng(0x05A3_F1E5);
    // Identical chains (truth 0) show the 1/√K decay most clearly.
    let errors: Vec<f64> = [25_000, 400_000]
        .iter()
        .map(|&k| estimate(&a, &a, gamma, k, StepSizes::Theorem, &mut rng))
        .collect();
    println!("identical chains, estimates {errors:?}");
    assert!(
        errors[1] < 0.6 * errors[0] && errors[1] < 0.15,
        "{errors:?}"
    );
    let far = (estimate(&a, &b, gamma, 400_000, StepSizes::Theorem, &mut rng) - exact).abs();
    println!("exact {exact:.4}, error {far:.4}");
    assert!(far < 0.1, "{far}");
}

/// Model selection: among well-separated candidates, the estimated
/// distance to the truth is smallest for the true parameter.
#[test]
fn distance_from_samples_selects_the_true_parameter() {
    let gamma = 0.8;
    let truth = ring(4, 0.5);
    let mut rng = Rng(0x00DE_15E1);
    let grid = [0.1, 0.5, 0.9];
    let d: Vec<f64> = grid
        .iter()
        .map(|&p| {
            estimate(
                &truth,
                &ring(4, p),
                gamma,
                200_000,
                StepSizes::Theorem,
                &mut rng,
            )
        })
        .collect();
    println!("distances over the grid: {d:?}");
    let best = (0..grid.len())
        .min_by(|&i, &j| d[i].total_cmp(&d[j]))
        .unwrap();
    assert_eq!(grid[best], 0.5, "{d:?}");
}
