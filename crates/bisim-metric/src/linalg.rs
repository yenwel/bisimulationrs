//! Small dense linear algebra: enough for delay embeddings, least squares
//! and Procrustes problems of a few dozen dimensions. Row-major `f64`.

use std::ops::{Index, IndexMut};

/// A dense row-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// From rows of equal length.
    ///
    /// # Panics
    /// If the rows differ in length.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let cols = rows.first().map_or(0, Vec::len);
        assert!(rows.iter().all(|r| r.len() == cols), "ragged rows");
        Self {
            rows: rows.len(),
            cols,
            data: rows.concat(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t[(j, i)] = self[(i, j)];
            }
        }
        t
    }

    /// `self · other`.
    ///
    /// # Panics
    /// On a dimension mismatch.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "dimension mismatch");
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                if a == 0.0 {
                    continue;
                }
                let (row, out_row) = (other.row(k), i * other.cols);
                for (j, &b) in row.iter().enumerate() {
                    out.data[out_row + j] += a * b;
                }
            }
        }
        out
    }

    pub fn add(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a - b)
    }

    pub fn scale(&self, k: f64) -> Self {
        Self {
            data: self.data.iter().map(|x| x * k).collect(),
            ..self.clone()
        }
    }

    fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "dimension mismatch"
        );
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
            ..self.clone()
        }
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Trace of `selfᵀ · other` (the Frobenius inner product).
    pub fn dot(&self, other: &Self) -> f64 {
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    /// Solves `self · X = rhs` by LU with partial pivoting; `None` if
    /// singular.
    pub fn solve(&self, rhs: &Self) -> Option<Self> {
        assert_eq!(self.rows, self.cols, "square system");
        assert_eq!(self.rows, rhs.rows, "dimension mismatch");
        let n = self.rows;
        let mut a = self.clone();
        let mut b = rhs.clone();
        let scale = self
            .data
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()))
            .max(1.0);
        for k in 0..n {
            let p = (k..n).max_by(|&i, &j| a[(i, k)].abs().total_cmp(&a[(j, k)].abs()))?;
            if a[(p, k)].abs() <= 1e-14 * scale {
                return None;
            }
            if p != k {
                a.swap_rows(p, k);
                b.swap_rows(p, k);
            }
            for i in k + 1..n {
                let f = a[(i, k)] / a[(k, k)];
                if f == 0.0 {
                    continue;
                }
                for j in k..n {
                    a[(i, j)] -= f * a[(k, j)];
                }
                for j in 0..b.cols {
                    b[(i, j)] -= f * b[(k, j)];
                }
            }
        }
        for j in 0..b.cols {
            for i in (0..n).rev() {
                let mut s = b[(i, j)];
                for k in i + 1..n {
                    s -= a[(i, k)] * b[(k, j)];
                }
                b[(i, j)] = s / a[(i, i)];
            }
        }
        Some(b)
    }

    fn swap_rows(&mut self, i: usize, j: usize) {
        for c in 0..self.cols {
            self.data.swap(i * self.cols + c, j * self.cols + c);
        }
    }

    /// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations:
    /// eigenvalues in descending order and the matching eigenvectors as
    /// columns.
    pub fn symmetric_eigen(&self) -> (Vec<f64>, Self) {
        assert_eq!(self.rows, self.cols, "square matrix");
        let n = self.rows;
        let mut a = self.clone();
        let mut v = Self::identity(n);
        for _sweep in 0..100 {
            let off: f64 = (0..n)
                .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
                .map(|(i, j)| a[(i, j)] * a[(i, j)])
                .sum();
            if off <= 1e-30 * a.norm().powi(2).max(1e-300) {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    let apq = a[(p, q)];
                    if apq.abs() < 1e-300 {
                        continue;
                    }
                    let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let (akp, akq) = (a[(k, p)], a[(k, q)]);
                        a[(k, p)] = c * akp - s * akq;
                        a[(k, q)] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let (apk, aqk) = (a[(p, k)], a[(q, k)]);
                        a[(p, k)] = c * apk - s * aqk;
                        a[(q, k)] = s * apk + c * aqk;
                    }
                    for k in 0..n {
                        let (vkp, vkq) = (v[(k, p)], v[(k, q)]);
                        v[(k, p)] = c * vkp - s * vkq;
                        v[(k, q)] = s * vkp + c * vkq;
                    }
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| a[(j, j)].total_cmp(&a[(i, i)]));
        let values = order.iter().map(|&i| a[(i, i)]).collect();
        let mut vectors = Self::zeros(n, n);
        for (new, &old) in order.iter().enumerate() {
            for k in 0..n {
                vectors[(k, new)] = v[(k, old)];
            }
        }
        (values, vectors)
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_and_eigen_reconstruct() {
        let a = Mat::from_rows(&[
            vec![4.0, 1.0, 2.0],
            vec![1.0, 3.0, 0.5],
            vec![2.0, 0.5, 5.0],
        ]);
        let b = Mat::from_rows(&[vec![1.0], vec![2.0], vec![3.0]]);
        let x = a.solve(&b).unwrap();
        assert!(a.mul(&x).sub(&b).norm() < 1e-12);
        let (vals, vecs) = a.symmetric_eigen();
        assert!(vals.windows(2).all(|w| w[0] >= w[1]));
        let mut d = Mat::zeros(3, 3);
        for i in 0..3 {
            d[(i, i)] = vals[i];
        }
        let back = vecs.mul(&d).mul(&vecs.transpose());
        assert!(back.sub(&a).norm() < 1e-10);
        assert!(vecs.transpose().mul(&vecs).sub(&Mat::identity(3)).norm() < 1e-10);
    }

    #[test]
    fn singular_systems_are_reported() {
        let a = Mat::from_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(a.solve(&Mat::identity(2)).is_none());
    }
}
