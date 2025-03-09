use crate::constants::SQRT_EPS;
use linalg_traits::Vector;

/// Jacobian of a multivariate, vector-valued function using the forward difference approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function, $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`SQRT_EPS`].
///
/// # Returns
///
/// Jacobian of $\mathbf{f}$ with respect to $\mathbf{x}$, evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\mathbf{J}(\mathbf{x}\_{0})={\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}}\in\mathbb{R}^{m\times n}$$
///
/// # Note
///
/// This function performs $n+1$ evaluations of $f(x)$.
///
/// # Warning
///
/// This function will always return a dynamically-sized matrix, even if the function `f` uses
/// statically-sized vectors. This is to avoid needing to pass a const generic to this function to
/// define the number of rows ($m$) of the Jacobian. Instead, the number of rows is determined at
/// runtime.
///
/// # Example
///
/// Approximate the Jacobian of
///
/// $$
/// \mathbf{f}(\mathbf{x})=
/// \begin{bmatrix}
///     x_{0} \\\\
///     5x_{2} \\\\
///     4x_{1}^{2}-2x_{2} \\\\
///     x_{2}\sin{x_{0}}
/// \end{bmatrix}
/// $$
///
/// at $\mathbf{x}=(5,6,7)^{T}$, and compare the result to the true result of
///
/// $$
/// \mathbf{J}\left((5,6,7)^{T}\right)=
/// \begin{bmatrix}
///     1 & 0 & 0 \\\\
///     0 & 0 & 5 \\\\
///     0 & 48 & -2 \\\\
///     7\cos{(5)} & 0 & \sin{(5)}
/// \end{bmatrix}
/// $$
///
/// #### Using standard vectors
///
/// ```
/// use numtest::*;
///
/// use linalg_traits::{Mat, Matrix};
/// use numdiff::forward_difference::jacobian;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| {
///     vec![
///         x[0],
///         5.0 * x[2],
///         4.0 * x[1].powi(2) - 2.0 * x[2],
///         x[2] * x[0].sin(),
///     ]
/// };
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 6.0, 7.0];
///
/// // Approximate the Jacobian of f(x) at the evaluation point.
/// let jac: Mat<f64> = jacobian(&f, &x0, None);
///
/// // True Jacobian of f(x) at the evaluation point.
/// let jac_true: Mat<f64> = Mat::from_row_slice(
///     4,
///     3,
///     &[
///         1.0,
///         0.0,
///         0.0,
///         0.0,
///         0.0,
///         5.0,
///         0.0,
///         48.0,
///         -2.0,
///         7.0 * 5.0_f64.cos(),
///         0.0,
///         5.0_f64.sin(),
///     ],
/// );
///
/// // Check the accuracy of the Jacobian approximation.
/// assert_arrays_equal_to_decimal!(jac, jac_true, 6);
/// ```
///
/// #### Using other vector types
///
/// We can also use other types of vectors, such as `nalgebra::SVector`, `nalgebra::DVector`,
/// `ndarray::Array1`, `faer::Mat`, or any other type of vector that implements the
/// `linalg_traits::Vector` trait.
///
/// ```
/// use faer::Mat as FMat;
/// use linalg_traits::{Mat, Matrix, Vector};
/// use nalgebra::{dvector, DMatrix, DVector, SVector};
/// use ndarray::{array, Array1, Array2};
/// use numtest::*;
///
/// use numdiff::forward_difference::jacobian;
///
/// let jac_true_row_major: Mat<f64> = Mat::from_row_slice(
///     4,
///     3,
///     &[
///         1.0,
///         0.0,
///         0.0,
///         0.0,
///         0.0,
///         5.0,
///         0.0,
///         48.0,
///         -2.0,
///         7.0 * 5.0_f64.cos(),
///         0.0,
///         5.0_f64.sin(),
///     ],
/// );
/// let jac_true_col_major: DMatrix<f64> = DMatrix::from_row_slice(
///     4,
///     3,
///     &[
///         1.0,
///         0.0,
///         0.0,
///         0.0,
///         0.0,
///         5.0,
///         0.0,
///         48.0,
///         -2.0,
///         7.0 * 5.0_f64.cos(),
///         0.0,
///         5.0_f64.sin(),
///     ],
/// );
///
/// // nalgebra::DVector
/// let f_dvector = |x: &DVector<f64>| {
///     dvector![
///         x[0],
///         5.0 * x[2],
///         4.0 * x[1].powi(2) - 2.0 * x[2],
///         x[2] * x[0].sin(),
///     ]
/// };
/// let x0_dvector: DVector<f64> = dvector![5.0, 6.0, 7.0];
/// let jac_dvector: DMatrix<f64> = jacobian(&f_dvector, &x0_dvector, None);
/// assert_arrays_equal_to_decimal!(jac_dvector, jac_true_col_major, 6);
///
/// // nalgebra::SVector
/// let f_svector = |x: &SVector<f64, 3>| {
///     SVector::<f64, 4>::from_row_slice(&[
///         x[0],
///         5.0 * x[2],
///         4.0 * x[1].powi(2) - 2.0 * x[2],
///         x[2] * x[0].sin(),
///     ])
/// };
/// let x0_svector: SVector<f64, 3> = SVector::from_row_slice(&[5.0, 6.0, 7.0]);
/// let jac_svector: DMatrix<f64> = jacobian(&f_svector, &x0_svector, None);
/// assert_arrays_equal_to_decimal!(jac_svector, jac_true_col_major, 6);
///
/// // ndarray::Array1
/// let f_array1 = |x: &Array1<f64>| {
///     array![
///         x[0],
///         5.0 * x[2],
///         4.0 * x[1].powi(2) - 2.0 * x[2],
///         x[2] * x[0].sin(),
///     ]
/// };
/// let x0_array1: Array1<f64> = array![5.0, 6.0, 7.0];
/// let jac_array1: Array2<f64> = jacobian(&f_array1, &x0_array1, None);
/// assert_arrays_equal_to_decimal!(jac_array1, jac_true_row_major, 6);
///
/// // faer::Mat
/// let f_mat = |x: &FMat<f64>| {
///     FMat::<f64>::from_slice(&[
///         x[(0, 0)],
///         5.0 * x[(2, 0)],
///         4.0 * x[(1, 0)].powi(2) - 2.0 * x[(2, 0)],
///         x[(2, 0)] * x[(0, 0)].sin(),
///     ])
/// };
/// let x0_mat: FMat<f64> = FMat::from_slice(&[5.0, 6.0, 7.0]);
/// let jac_mat: FMat<f64> = jacobian(&f_mat, &x0_mat, None);
/// assert_arrays_equal_to_decimal!(jac_mat.as_col_slice(), jac_true_col_major, 6);
/// ```
///
/// #### Modifying the relative step size
///
/// We can also modify the relative step size. Choosing a coarser relative step size, we get a worse
/// approximation.
///
/// ```
/// use linalg_traits::{Mat, Matrix};
/// use numtest::*;
///
/// use numdiff::forward_difference::jacobian;
///
/// let f = |x: &Vec<f64>| {
///     vec![
///         x[0],
///         5.0 * x[2],
///         4.0 * x[1].powi(2) - 2.0 * x[2],
///         x[2] * x[0].sin(),
///     ]
/// };
/// let x0 = vec![5.0, 6.0, 7.0];
///
/// let jac: Mat<f64> = jacobian(&f, &x0, Some(0.001));
/// let jac_true: Mat<f64> = Mat::from_row_slice(
///     4,
///     3,
///     &[
///         1.0,
///         0.0,
///         0.0,
///         0.0,
///         0.0,
///         5.0,
///         0.0,
///         48.0,
///         -2.0,
///         7.0 * 5.0_f64.cos(),
///         0.0,
///         5.0_f64.sin(),
///     ],
/// );
///
/// assert_arrays_equal_to_decimal!(jac, jac_true, 1);
/// ```
pub fn jacobian<V, U>(f: &impl Fn(&V) -> U, x0: &V, h: Option<f64>) -> V::DMatrixMxN
where
    V: Vector<f64>,
    U: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = √(ε) if not specified.
    let h = h.unwrap_or(*SQRT_EPS);

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Determine the size of the Jacobian.
    let m = f0.len();
    let n = x0.len();

    // Initialize a matrix of zeros to store the Jacobian.
    let mut jac = x0.new_dmatrix_m_by_n(m);

    // Variable to store the absolute step size in the kth direction.
    let mut dxk: f64;

    // Variable to store the original value of the evaluation point in the kth direction.
    let mut x0k: f64;

    // Variable to store the partial derivative of f with respect to xₖ.
    let mut dfk;

    // Evaluate the Jacobian.
    for k in 0..n {
        // Original value of the evaluation point in the kth direction.
        x0k = x0.vget(k);

        // Absolute step size in the kth direction.
        dxk = h * (1.0 + x0k.abs());

        // Step forward in the kth direction.
        x0.vset(k, x0k + dxk);

        // Partial derivative of f with respect to xₖ.
        dfk = f(&x0).sub(&f0).div(dxk);

        // Reset the evaluation point.
        x0.vset(k, x0k);

        // Store the partial derivative of f with respect to xₖ in the kth column of the Jacobian.
        for i in 0..m {
            jac[(i, k)] = dfk.vget(i);
        }
    }

    jac
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::{Mat, Matrix};
    use nalgebra::{dvector, DMatrix, DVector, SMatrix, SVector};
    use ndarray::{array, Array1, Array2};
    use numtest::*;

    #[test]
    fn test_jacobian_1() {
        let f = |x: &Vec<f64>| vec![x[0].powi(2)];
        let x0 = vec![2.0];
        let jac = |x: &Vec<f64>| Mat::from_row_slice(1, 1, &[2.0 * x[0]]);
        assert_arrays_equal_to_decimal!(jacobian(&f, &x0, None), jac(&x0), 7);
    }

    #[test]
    fn test_jacobian_2() {
        let f = |x: &Array1<f64>| array![x[0].powi(2), x[0].powi(3)];
        let x0 = array![2.0];
        let jac = |x: &Array1<f64>| {
            Array2::<f64>::from_row_slice(2, 1, &[2.0 * x[0], 3.0 * x[0].powi(2)])
        };
        assert_arrays_equal_to_decimal!(jacobian(&f, &x0, None), jac(&x0), 6);
    }

    #[test]
    fn test_jacobian_3() {
        let f = |x: &DVector<f64>| dvector![x[0].powi(2) + x[1].powi(3)];
        let x0 = dvector![1.0, 2.0];
        let jac = |x: &DVector<f64>| {
            <DMatrix<f64> as Matrix<f64>>::from_row_slice(1, 2, &[2.0 * x[0], 3.0 * x[1].powi(2)])
        };
        assert_arrays_equal_to_decimal!(jacobian(&f, &x0, None), jac(&x0), 6);
    }

    #[test]
    fn test_jacobian_4() {
        let f =
            |x: &SVector<f64, 2>| SVector::<f64, 2>::from_row_slice(&[x[0].powi(2), x[1].powi(3)]);
        let x0 = SVector::<f64, 2>::from_row_slice(&[1.0, 2.0]);
        let jac = |x: &SVector<f64, 2>| {
            DMatrix::<f64>::from_row_slice(2, 2, &[2.0 * x[0], 0.0, 0.0, 3.0 * x[1].powi(2)])
        };
        assert_arrays_equal_to_decimal!(jacobian(&f, &x0, None), jac(&x0), 6);
    }

    #[test]
    fn test_jacobian_5() {
        let f = |x: &SVector<f64, 3>| {
            SVector::<f64, 4>::from_row_slice(&[
                x[0],
                5.0 * x[2],
                4.0 * x[1].powi(2) - 2.0 * x[2],
                x[2] * x[0].sin(),
            ])
        };
        let x0: SVector<f64, 3> = SVector::from_row_slice(&[5.0, 6.0, 7.0]);
        let jac = |x: &SVector<f64, 3>| {
            SMatrix::<f64, 4, 3>::from_row_slice(&[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0,
                0.0,
                8.0 * x[1],
                -2.0,
                x[2] * x[0].cos(),
                0.0,
                x[0].sin(),
            ])
        };
        assert_arrays_equal_to_decimal!(jacobian(&f, &x0, None), jac(&x0), 6);
    }
}
