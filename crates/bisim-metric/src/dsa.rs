//! Dynamical similarity analysis (Ostrow, Eisen, Kozachkov, Fiete, "Beyond
//! Geometry: Comparing the Temporal Structure of Computation in Neural
//! Circuits with Dynamical Similarity Analysis", NeurIPS 2023): compares
//! two *continuous* multichannel time series, such as soil-moisture or
//! biomass sensors, by the linear dynamics fitted to them.
//!
//! This is not a bisimulation. It answers "do these two signals evolve by
//! the same rules, up to a change of coordinates?", and complements the
//! discrete-state distances of this crate when the data are measurements
//! rather than states. Steps, as in the paper:
//!
//! 1. delay (Hankel) embedding with `delays` lags of every channel;
//! 2. the leading `rank` singular directions of the embedding, as reduced
//!    coordinates `z_t` (HAVOK-style);
//! 3. a linear model `z_{t+1} ≈ A·z_t` by ridge least squares;
//! 4. Procrustes over vector fields: `min over orthogonal C of
//!    ‖A_x − C·A_y·Cᵀ‖_F`, solved here by Riemannian gradient descent with
//!    the Cayley retraction from several starts, including reflections,
//!    since the Cayley map alone only reaches rotations (as the paper notes).
//!
//! The Frobenius form is a metric on the fitted operators (up to orthogonal
//! conjugation); the angular form, in `[0, π]`, ignores overall scale. For
//! symmetric operators the minimum is the distance between the sorted
//! eigenvalues (Hoffman–Wielandt), which the tests use as an oracle.

use crate::kantorovich::MetricError;
use crate::linalg::Mat;

/// Settings for [`dsa`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DsaOptions {
    /// Lags per channel in the delay embedding.
    pub delays: usize,
    /// Reduced dimension `r` of the fitted operator.
    pub rank: usize,
    /// Ridge penalty of the regression.
    pub ridge: f64,
    /// Random starts of the Procrustes search (besides the identity and a
    /// reflection).
    pub restarts: usize,
}

impl Default for DsaOptions {
    fn default() -> Self {
        Self {
            delays: 4,
            rank: 4,
            ridge: 1e-8,
            restarts: 8,
        }
    }
}

/// The comparison of two series.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsa {
    /// `min ‖A_x − C·A_y·Cᵀ‖_F`.
    pub distance: f64,
    /// The angle between `A_x` and the best `C·A_y·Cᵀ`, in `[0, π]`.
    pub angle: f64,
    /// The fitted operators.
    pub a_x: Mat,
    pub a_y: Mat,
}

/// The fitted linear operator of a series (`series[t][channel]`).
pub fn fit_operator(series: &[Vec<f64>], options: DsaOptions) -> Result<Mat, MetricError> {
    let channels = series.first().map_or(0, Vec::len);
    let p = options.delays.max(1);
    let dim = channels * p;
    if channels == 0 || series.iter().any(|row| row.len() != channels) {
        return Err(MetricError::Invalid(
            "series must be non-empty with equal channel counts".into(),
        ));
    }
    if series.iter().flatten().any(|v| !v.is_finite()) {
        return Err(MetricError::Invalid(
            "series contains non-finite values".into(),
        ));
    }
    let rows = series.len().saturating_sub(p - 1);
    if options.rank == 0 || options.rank > dim || rows < options.rank + 2 {
        return Err(MetricError::Invalid(format!(
            "rank {} needs at most {dim} dimensions and more than {} embedded samples (have {rows})",
            options.rank,
            options.rank + 1
        )));
    }
    // Hankel matrix: row i holds x_{i+p−1}, x_{i+p−2}, …, x_i.
    let hankel: Vec<Vec<f64>> = (0..rows)
        .map(|i| {
            (0..p)
                .flat_map(|lag| series[i + p - 1 - lag].iter().copied())
                .collect()
        })
        .collect();
    let h = Mat::from_rows(&hankel);
    // Right singular vectors of H from the eigenvectors of HᵀH; the reduced
    // coordinates are z = H·w / σ (the left singular vectors).
    let (values, vectors) = h.transpose().mul(&h).symmetric_eigen();
    let r = options.rank;
    let mut w = Mat::zeros(dim, r);
    for k in 0..r {
        let sigma = values[k].max(0.0).sqrt();
        if sigma <= 1e-12 * values[0].max(1e-300).sqrt() {
            return Err(MetricError::Invalid(format!(
                "rank {r} exceeds the numerical rank of the embedding"
            )));
        }
        for d in 0..dim {
            w[(d, k)] = vectors[(d, k)] / sigma;
        }
    }
    let z = h.mul(&w);
    // Ridge regression z_{t+1} = A z_t: Aᵀ = (ZᵀZ + λI)⁻¹ Zᵀ Z′.
    let take = |from: usize, to: usize| {
        Mat::from_rows(&(from..to).map(|i| z.row(i).to_vec()).collect::<Vec<_>>())
    };
    let (z0, z1) = (take(0, rows - 1), take(1, rows));
    let gram = z0
        .transpose()
        .mul(&z0)
        .add(&Mat::identity(r).scale(options.ridge));
    let a_t = gram
        .solve(&z0.transpose().mul(&z1))
        .ok_or_else(|| MetricError::Invalid("singular regression".into()))?;
    Ok(a_t.transpose())
}

/// Compares two series by their fitted dynamics.
pub fn dsa(x: &[Vec<f64>], y: &[Vec<f64>], options: DsaOptions) -> Result<Dsa, MetricError> {
    let (a_x, a_y) = (fit_operator(x, options)?, fit_operator(y, options)?);
    let (distance, c) = procrustes(&a_x, &a_y, options.restarts);
    let aligned = c.mul(&a_y).mul(&c.transpose());
    let cos = a_x.dot(&aligned) / (a_x.norm() * a_y.norm()).max(1e-300);
    Ok(Dsa {
        distance,
        angle: cos.clamp(-1.0, 1.0).acos(),
        a_x,
        a_y,
    })
}

/// `min over orthogonal C of ‖a − C·b·Cᵀ‖_F` and the minimising `C`.
pub fn procrustes(a: &Mat, b: &Mat, restarts: usize) -> (f64, Mat) {
    let r = a.rows();
    let mut starts = vec![Mat::identity(r)];
    let mut reflection = Mat::identity(r);
    reflection[(0, 0)] = -1.0;
    starts.push(reflection);
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..restarts {
        starts.push(random_orthogonal(r, &mut seed));
    }
    starts
        .into_iter()
        .map(|c| descend(a, b, c))
        .min_by(|x, y| x.0.total_cmp(&y.0))
        .expect("at least one start")
}

/// Riemannian gradient descent on the orthogonal group from `c` (Cayley
/// retraction, Armijo backtracking).
fn descend(a: &Mat, b: &Mat, mut c: Mat) -> (f64, Mat) {
    let r = a.rows();
    let f = |c: &Mat| a.sub(&c.mul(b).mul(&c.transpose())).norm();
    let mut value = f(&c);
    let mut tau = 1.0;
    for _ in 0..50_000 {
        let res = a.sub(&c.mul(b).mul(&c.transpose()));
        // Euclidean gradient of ‖res‖² in C.
        let g = res
            .mul(&c)
            .mul(&b.transpose())
            .add(&res.transpose().mul(&c).mul(b))
            .scale(-2.0);
        let w = g.mul(&c.transpose()).sub(&c.mul(&g.transpose()));
        let slope = w.norm();
        if slope < 1e-12 {
            break;
        }
        let mut accepted = false;
        while tau > 1e-12 {
            let half = w.scale(tau / 2.0);
            let Some(next) = Mat::identity(r)
                .add(&half)
                .solve(&Mat::identity(r).sub(&half).mul(&c))
            else {
                tau /= 2.0;
                continue;
            };
            let v = f(&next);
            if v * v <= value * value - 1e-4 * tau * slope * slope / 2.0 {
                c = next;
                value = v;
                accepted = true;
                tau *= 2.0;
                break;
            }
            tau /= 2.0;
        }
        if !accepted {
            break;
        }
    }
    (value, c)
}

/// A pseudo-random orthogonal matrix (Gram–Schmidt on uniform entries).
fn random_orthogonal(r: usize, seed: &mut u64) -> Mat {
    let mut next = || {
        *seed ^= *seed >> 12;
        *seed ^= *seed << 25;
        *seed ^= *seed >> 27;
        (seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut q = Mat::zeros(r, r);
    for j in 0..r {
        let mut v: Vec<f64> = (0..r).map(|_| next()).collect();
        for k in 0..j {
            let d: f64 = (0..r).map(|i| v[i] * q[(i, k)]).sum();
            for (i, vi) in v.iter_mut().enumerate() {
                *vi -= d * q[(i, k)];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (i, vi) in v.iter().enumerate() {
            q[(i, j)] = vi / norm;
        }
    }
    q
}
