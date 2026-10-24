//! A learned bisimulation distance (deep-RL style): a small neural network
//! maps a state's features to an embedding whose ℓ1 distances approximate
//! the bisimulation distance of [`distance`](crate::distance).
//!
//! The objective follows Deep Bisimulation for Control (Zhang, McAllister,
//! Calandra, Gal, Levine, ICLR 2021, eq. 4): for pairs `(i, j)` of a batch
//! and its permutation,
//!
//! ```text
//! (‖φ(sᵢ) − φ(sⱼ)‖₁ − (1 − γ)·|rᵢ − rⱼ| − γ·W(P(·|sᵢ), P(·|sⱼ); ‖φ̄ − φ̄‖₁))²
//! ```
//!
//! with `φ̄` a slowly updated copy of the network (stop-gradient target).
//! Two deliberate differences, because the model is known here: the
//! next-state term is exact optimal transport between `samples` sampled
//! successors of each state (DBC learns a Gaussian latent dynamics model and
//! uses the closed-form 2-Wasserstein distance), and rewards are scaled by
//! `1 − γ` so the result is on the same `[0, 1]` scale as the exact
//! distance.
//!
//! **Estimates only.** Training is stochastic and the network can be wrong
//! anywhere, so a learned distance is never a verdict: it ranks and
//! approximates. Measured against the exact distance:
//!
//! - permaculture beds (guild vs monoculture, the `learned_distance`
//!   example): 0.112 vs 0.122 (γ = 0.88), 0.123 vs 0.124 (γ = 0.91), 0.121 vs
//!   0.126 (γ = 0.94), 0.143 vs 0.134 (γ = 0.98, 1.7 million state pairs:
//!   36 s against 8.4 minutes) with enough training steps. Too few steps gave values
//!   far too high (0.313 vs 0.126), which the default step count avoids;
//! - 4-state rings with one-hot features (γ = 0.8): mean absolute error 0.03
//!   and 0.08 over all state pairs (tests).
//!
//! Small successor distributions enter the target exactly; sampling them
//! (as model-free methods must) biases the target upward, and bootstrapping
//! multiplies that bias by about `1 / (1 − γ)`. Its use is where the exact computation is out of reach (it
//! never builds the space of state pairs) and where states are described by
//! features, so that it generalises across states. The tests and the
//! `learned_distance` example measure how far it is from the exact value.
//!
//! The network is a plain multilayer perceptron with `tanh` hidden layers,
//! trained by Adam, written here without dependencies; gradients are checked
//! against finite differences in the tests.

use crate::flow::transport;
use crate::kantorovich::{Chain, MetricError};

/// A multilayer perceptron: `tanh` hidden layers, linear output.
#[derive(Debug, Clone)]
pub struct Mlp {
    layers: Vec<Layer>,
}

#[derive(Debug, Clone)]
struct Layer {
    inputs: usize,
    outputs: usize,
    /// Row-major `outputs × inputs`.
    w: Vec<f64>,
    b: Vec<f64>,
}

/// Gradients, shaped like the network's parameters.
#[derive(Debug, Clone)]
struct Grads {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

/// Deterministic randomness (SplitMix64).
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

impl Mlp {
    /// A network with layer widths `sizes` (input first), Xavier-uniform
    /// initial weights from `seed`.
    ///
    /// # Panics
    /// With fewer than two sizes or a zero width.
    pub fn new(sizes: &[usize], seed: u64) -> Self {
        assert!(
            sizes.len() >= 2 && sizes.iter().all(|&s| s > 0),
            "invalid layer sizes"
        );
        let mut rng = Rng(seed);
        let layers = sizes
            .windows(2)
            .map(|w| {
                let (inputs, outputs) = (w[0], w[1]);
                let limit = (6.0 / (inputs + outputs) as f64).sqrt();
                Layer {
                    inputs,
                    outputs,
                    w: (0..inputs * outputs)
                        .map(|_| (2.0 * rng.uniform() - 1.0) * limit)
                        .collect(),
                    b: vec![0.0; outputs],
                }
            })
            .collect();
        Self { layers }
    }

    /// The output for input `x`.
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        self.forward(x).pop().expect("at least one layer")
    }

    /// Activations of every layer, input first.
    fn forward(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let mut acts = vec![x.to_vec()];
        for (k, l) in self.layers.iter().enumerate() {
            let a = acts.last().expect("input");
            let last = k + 1 == self.layers.len();
            let out: Vec<f64> = (0..l.outputs)
                .map(|o| {
                    let z = l.b[o]
                        + (0..l.inputs)
                            .map(|i| l.w[o * l.inputs + i] * a[i])
                            .sum::<f64>();
                    if last { z } else { z.tanh() }
                })
                .collect();
            acts.push(out);
        }
        acts
    }

    fn zero_grads(&self) -> Grads {
        Grads {
            w: self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect(),
            b: self.layers.iter().map(|l| vec![0.0; l.b.len()]).collect(),
        }
    }

    /// Adds to `grads` the gradient of `⟨grad_out, output⟩` for the forward
    /// pass `acts`.
    fn backward(&self, acts: &[Vec<f64>], grad_out: &[f64], grads: &mut Grads) {
        let mut delta = grad_out.to_vec();
        for k in (0..self.layers.len()).rev() {
            let l = &self.layers[k];
            let input = &acts[k];
            for (o, &d) in delta.iter().enumerate() {
                grads.b[k][o] += d;
                let row = &mut grads.w[k][o * l.inputs..(o + 1) * l.inputs];
                for (g, &x) in row.iter_mut().zip(input) {
                    *g += d * x;
                }
            }
            if k > 0 {
                // Through the tanh of the layer below: d tanh = 1 − a².
                delta = (0..l.inputs)
                    .map(|i| {
                        let s: f64 = (0..l.outputs)
                            .map(|o| l.w[o * l.inputs + i] * delta[o])
                            .sum();
                        s * (1.0 - input[i] * input[i])
                    })
                    .collect();
            }
        }
    }

    /// `self ← τ·other + (1 − τ)·self` (a slowly following target network).
    fn follow(&mut self, other: &Self, tau: f64) {
        for (a, b) in self.layers.iter_mut().zip(&other.layers) {
            for (x, y) in a.w.iter_mut().zip(&b.w) {
                *x += tau * (y - *x);
            }
            for (x, y) in a.b.iter_mut().zip(&b.b) {
                *x += tau * (y - *x);
            }
        }
    }
}

/// Adam state for one network.
#[derive(Debug, Clone)]
struct Adam {
    m: Grads,
    v: Grads,
    t: i32,
    rate: f64,
}

impl Adam {
    fn new(net: &Mlp, rate: f64) -> Self {
        Self {
            m: net.zero_grads(),
            v: net.zero_grads(),
            t: 0,
            rate,
        }
    }

    fn step(&mut self, net: &mut Mlp, g: &Grads) {
        let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        self.t += 1;
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for (k, l) in net.layers.iter_mut().enumerate() {
            for (params, grad, m, v) in [
                (&mut l.w, &g.w[k], &mut self.m.w[k], &mut self.v.w[k]),
                (&mut l.b, &g.b[k], &mut self.m.b[k], &mut self.v.b[k]),
            ] {
                for i in 0..params.len() {
                    m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
                    v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
                    params[i] -= self.rate * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
                }
            }
        }
    }
}

/// Successor distributions up to this many states are used exactly in the
/// training target; larger ones are sampled.
pub const EXACT_SUPPORT: usize = 32;

/// Training settings for [`learn_metric`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnOptions {
    /// Width of the two hidden layers.
    pub hidden: usize,
    /// Dimension of the embedding.
    pub embedding: usize,
    /// States per batch (paired with a permutation of themselves).
    pub batch: usize,
    /// Sampled successors per state for the transport term, for states with
    /// more than [`EXACT_SUPPORT`] successors (others are used exactly).
    pub samples: usize,
    /// Gradient steps; `0` chooses `max(4000, ⌈700 / (1 − γ)⌉)`. Distances
    /// propagate one step of the chain per round of bootstrapping, so the
    /// training needed grows like `1 / (1 − γ)`; the constant 700 is what
    /// the permaculture beds needed to come within 0.01 of the exact value.
    pub steps: usize,
    pub learning_rate: f64,
    /// Rate at which the target network follows (`τ`).
    pub target_rate: f64,
    pub seed: u64,
}

impl Default for LearnOptions {
    fn default() -> Self {
        Self {
            hidden: 32,
            embedding: 8,
            batch: 64,
            samples: 8,
            steps: 0,
            learning_rate: 1e-3,
            target_rate: 0.01,
            seed: 0x5EED,
        }
    }
}

/// A trained embedding: distances between feature vectors.
#[derive(Debug, Clone)]
pub struct LearnedMetric {
    net: Mlp,
    /// The final training loss (mean over the last 100 steps).
    pub final_loss: f64,
}

impl LearnedMetric {
    /// The learned distance between two states, given by their features.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        let (a, b) = (self.net.eval(x), self.net.eval(y));
        a.iter().zip(&b).map(|(p, q)| (p - q).abs()).sum()
    }

    /// The embedding of a state.
    pub fn embed(&self, x: &[f64]) -> Vec<f64> {
        self.net.eval(x)
    }
}

/// Learns an embedding for the states of two chains (observations in
/// `[0, 1]`, as for [`distance`](crate::distance)), described by
/// `features_a[s]` and `features_b[s]` (all of one length). Training states
/// are visited by random walks from both initial states, restarting with
/// probability `1 − γ` (the discounted occupancy).
pub fn learn_metric(
    a: &Chain,
    b: &Chain,
    features_a: &[Vec<f64>],
    features_b: &[Vec<f64>],
    gamma: f64,
    options: LearnOptions,
) -> Result<LearnedMetric, MetricError> {
    if !(gamma > 0.0 && gamma < 1.0) {
        return Err(MetricError::Invalid(format!(
            "discount {gamma} not in (0, 1)"
        )));
    }
    if features_a.len() != a.rows.len() || features_b.len() != b.rows.len() {
        return Err(MetricError::Invalid(
            "one feature vector per state is needed".into(),
        ));
    }
    let dim = features_a.first().map_or(0, Vec::len);
    if dim == 0
        || features_a
            .iter()
            .chain(features_b)
            .any(|f| f.len() != dim || f.iter().any(|v| !v.is_finite()))
    {
        return Err(MetricError::Invalid(
            "features must be finite and of one non-zero length".into(),
        ));
    }
    if options.batch < 2 || options.samples == 0 || options.hidden == 0 || options.embedding == 0 {
        return Err(MetricError::Invalid(
            "batch ≥ 2, samples, hidden and embedding ≥ 1".into(),
        ));
    }
    // The union of both chains: states of `b` follow those of `a`.
    let na = a.rows.len();
    let rows: Vec<Vec<(usize, f64)>> = a
        .rows
        .iter()
        .cloned()
        .chain(
            b.rows
                .iter()
                .map(|r| r.iter().map(|&(t, p)| (t + na, p)).collect()),
        )
        .collect();
    let features: Vec<&Vec<f64>> = features_a.iter().chain(features_b).collect();
    let reward: Vec<f64> = a
        .observation
        .iter()
        .chain(&b.observation)
        .copied()
        .collect();
    let starts = [a.initial, na + b.initial];

    let mut rng = Rng(options.seed);
    let step_of = |s: usize, u: f64| {
        let mut acc = 0.0;
        for &(t, p) in &rows[s] {
            acc += p;
            if u < acc {
                return t;
            }
        }
        rows[s].last().map_or(s, |e| e.0)
    };
    // States to train on: random walks from both initial states.
    let mut buffer = Vec::new();
    for w in 0..(options.batch * 64) {
        let mut s = starts[w % 2];
        buffer.push(s);
        while rng.uniform() < gamma && buffer.len() < options.batch * 64 * 4 {
            s = step_of(s, rng.uniform());
            buffer.push(s);
        }
    }

    let mut net = Mlp::new(
        &[dim, options.hidden, options.hidden, options.embedding],
        options.seed,
    );
    let mut target = net.clone();
    let mut adam = Adam::new(&net, options.learning_rate);
    let mut losses = Vec::new();
    let k = options.samples;
    let steps = if options.steps == 0 {
        ((700.0 / (1.0 - gamma)).ceil() as usize).max(4_000)
    } else {
        options.steps
    };
    for _ in 0..steps {
        let batch: Vec<usize> = (0..options.batch)
            .map(|_| buffer[rng.below(buffer.len())])
            .collect();
        let mut partner = batch.clone();
        for i in (1..partner.len()).rev() {
            partner.swap(i, rng.below(i + 1));
        }
        let mut grads = net.zero_grads();
        let mut loss = 0.0;
        for (&i, &j) in batch.iter().zip(&partner) {
            let (ai, aj) = (net.forward(features[i]), net.forward(features[j]));
            let (zi, zj) = (ai.last().expect("output"), aj.last().expect("output"));
            let d: f64 = zi.iter().zip(zj).map(|(p, q)| (p - q).abs()).sum();
            // Target: reward gap plus transport between the successor
            // distributions, measured with the target network. Small
            // supports are used exactly (no sampling bias); larger ones are
            // replaced by `k` samples each.
            let successors = |s: usize, rng: &mut Rng| -> (Vec<Vec<f64>>, Vec<f64>) {
                if rows[s].len() <= EXACT_SUPPORT {
                    rows[s]
                        .iter()
                        .map(|&(t, p)| (target.eval(features[t]), p))
                        .unzip()
                } else {
                    (0..k)
                        .map(|_| {
                            (
                                target.eval(features[step_of(s, rng.uniform())]),
                                1.0 / k as f64,
                            )
                        })
                        .unzip()
                }
            };
            let (next_i, wi) = successors(i, &mut rng);
            let (next_j, wj) = successors(j, &mut rng);
            let cost: Vec<Vec<f64>> = next_i
                .iter()
                .map(|x| {
                    next_j
                        .iter()
                        .map(|y| x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum())
                        .collect()
                })
                .collect();
            let (ti, tj): (f64, f64) = (wi.iter().sum(), wj.iter().sum());
            let wj: Vec<f64> = wj.iter().map(|w| w * ti / tj).collect();
            let (ot, _) = transport(&wi, &wj, &cost);
            let t = (1.0 - gamma) * (reward[i] - reward[j]).abs() + gamma * ot;
            let err = d - t;
            loss += err * err;
            // d‖zi − zj‖₁/dzi = sign(zi − zj).
            let g: Vec<f64> = zi
                .iter()
                .zip(zj)
                .map(|(p, q)| 2.0 * err * (p - q).signum())
                .collect();
            let neg: Vec<f64> = g.iter().map(|x| -x).collect();
            net.backward(&ai, &g, &mut grads);
            net.backward(&aj, &neg, &mut grads);
        }
        let scale = 1.0 / options.batch as f64;
        for (w, b) in grads.w.iter_mut().zip(grads.b.iter_mut()) {
            w.iter_mut().for_each(|x| *x *= scale);
            b.iter_mut().for_each(|x| *x *= scale);
        }
        adam.step(&mut net, &grads);
        target.follow(&net, options.target_rate);
        losses.push(loss * scale);
    }
    let tail = &losses[losses.len().saturating_sub(100)..];
    let final_loss = tail.iter().sum::<f64>() / tail.len().max(1) as f64;
    Ok(LearnedMetric { net, final_loss })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradients_match_finite_differences() {
        let net = Mlp::new(&[3, 5, 4, 2], 7);
        let x = [0.3, -0.7, 0.2];
        let g_out = [0.6, -1.1];
        let mut grads = net.zero_grads();
        net.backward(&net.forward(&x), &g_out, &mut grads);
        let f = |n: &Mlp| {
            n.eval(&x)
                .iter()
                .zip(&g_out)
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        let h = 1e-6;
        for k in 0..net.layers.len() {
            for i in 0..net.layers[k].w.len() {
                let (mut up, mut down) = (net.clone(), net.clone());
                up.layers[k].w[i] += h;
                down.layers[k].w[i] -= h;
                let numeric = (f(&up) - f(&down)) / (2.0 * h);
                assert!(
                    (numeric - grads.w[k][i]).abs() < 1e-6,
                    "w[{k}][{i}]: {numeric} vs {}",
                    grads.w[k][i]
                );
            }
            for i in 0..net.layers[k].b.len() {
                let (mut up, mut down) = (net.clone(), net.clone());
                up.layers[k].b[i] += h;
                down.layers[k].b[i] -= h;
                let numeric = (f(&up) - f(&down)) / (2.0 * h);
                assert!((numeric - grads.b[k][i]).abs() < 1e-6, "b[{k}][{i}]");
            }
        }
    }

    #[test]
    fn adam_fits_a_simple_function() {
        // y = sin(x) on [−2, 2].
        let mut net = Mlp::new(&[1, 16, 1], 3);
        let mut adam = Adam::new(&net, 1e-2);
        let xs: Vec<f64> = (0..40).map(|i| -2.0 + 4.0 * i as f64 / 39.0).collect();
        for _ in 0..3_000 {
            let mut g = net.zero_grads();
            for &x in &xs {
                let acts = net.forward(&[x]);
                let err = acts.last().unwrap()[0] - x.sin();
                net.backward(&acts, &[2.0 * err / xs.len() as f64], &mut g);
            }
            adam.step(&mut net, &g);
        }
        let worst = xs
            .iter()
            .map(|&x| (net.eval(&[x])[0] - x.sin()).abs())
            .fold(0.0, f64::max);
        assert!(worst < 0.05, "{worst}");
    }
}
