use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Hessian of a multivariate, scalar-valued function using the forward difference approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Hessian of $f$ with respect to $\mathbf{x}$, evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\mathbf{H}(\mathbf{x}_{0})\in\mathbb{R}^{n\times n}$$
///
/// # Note
///
/// This function performs $\frac{n(n+1)}{2}+1$ evaluations of $f(x)$.
///
/// # Example
///
/// Approximate the Hessian of
///
/// $$f(\mathbf{x})=x_{0}^{5}x_{1}+x_{0}\sin^{3}{x_{1}}$$
///
/// at $\mathbf{x}=(5,8)^{T}$, and compare the result to the true result of
///
/// $$
/// \begin{aligned}
///     \mathbf{H}\left((5,8)^{T}\right)&=
///     \begin{bmatrix}
///         20x_{0}^{3}x_{1} & 5x_{0}^{4}+3\sin^{2}{x_{1}}\cos{x_{1}} \\\\
///         5x_{0}^{4}+3\sin^{2}{x_{1}}\cos{x_{1}} & 6x_{0}\sin{x_{1}}\cos^{2}{x_{1}}-3x_{0}\sin^{3}{x_{1}}
///     \end{bmatrix}
///     \bigg\rvert_{\mathbf{x}=(5,8)^{T}} \\\\
///     &=
///     \begin{bmatrix}
///         20(5)^{3}(8) & 5(5)^{4}+3\sin^{2}{(8)}\cos{(8)} \\\\
///         5(5)^{4}+3\sin^{2}{(8)}\cos{(8)} & 6(5)\sin{(8)}\cos^{2}{(8)}-3(5)\sin^{3}{(8)}
///     \end{bmatrix} \\\\
///     &=
///     \begin{bmatrix}
///         20000 & 3125+3\sin^{2}{(8)}\cos{(8)} \\\\
///         3125+3\sin^{2}{(8)}\cos{(8)} & 30\sin{(8)}\cos^{2}{(8)}-15\sin^{3}{(8)}
///     \end{bmatrix}
/// \end{aligned}
/// $$
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Mat, Matrix};
/// use numtest::*;
///
/// use numdiff::forward_difference::shessian;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3);
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 8.0];
///
/// // Approximate the Hessian of f(x) at the evaluation point.
/// let hess: Mat<f64> = shessian(&f, &x0, None);
///
/// // True Hessian of f(x) at the evaluation point.
/// let hess_true: Mat<f64> = Mat::from_row_slice(
///     2,
///     2,
///     &[
///         20000.0,
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         30.0 * 8.0_f64.sin() * 8.0_f64.cos().powi(2) - 15.0 * 8.0_f64.sin().powi(3)
///     ]
/// );
///
/// // Check the accuracy of the Hessian approximation.
/// assert_arrays_equal_to_decimal!(hess, hess_true, 0);
/// ```
///
/// #### Using other vector types
///
/// We can also use other types of vectors, such as `nalgebra::SVector`, `nalgebra::DVector`,
/// `ndarray::Array1`, or any other type of vector that implements the `linalg_traits::Vector`
/// trait.
///
/// ```
/// use linalg_traits::{Mat, Matrix};
/// use nalgebra::{dvector, DMatrix, DVector, SMatrix, SVector};
/// use ndarray::{array, Array1, Array2};
/// use numtest::*;
///
/// use numdiff::forward_difference::shessian;
///
/// let hess_true: Mat<f64> = Mat::from_row_slice(
///     2,
///     2,
///     &[
///         20000.0,
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         30.0 * 8.0_f64.sin() * 8.0_f64.cos().powi(2) - 15.0 * 8.0_f64.sin().powi(3)
///     ]
/// );
///
/// // nalgebra::DVector
/// let f_dvector = |x: &DVector<f64>| x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3);
/// let x0_dvector: DVector<f64> = dvector![5.0, 8.0];
/// let hess_dvector: DMatrix<f64> = shessian(&f_dvector, &x0_dvector, None);
/// assert_arrays_equal_to_decimal!(hess_dvector, hess_true, 0);
///
/// // nalgebra::SVector
/// let f_svector = |x: &SVector<f64,2>| x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3);
/// let x0_svector: SVector<f64, 2> = SVector::from_row_slice(&[5.0, 8.0]);
/// let hess_svector: SMatrix<f64, 2, 2> = shessian(&f_svector, &x0_svector, None);
/// assert_arrays_equal_to_decimal!(hess_svector, hess_true, 0);
///
/// // ndarray::Array1
/// let f_array1 = |x: &Array1<f64>| x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3);
/// let x0_array1: Array1<f64> = array![5.0, 8.0];
/// let hess_array1: Array2<f64> = shessian(&f_array1, &x0_array1, None);
/// assert_arrays_equal_to_decimal!(hess_array1, hess_true, 0);
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
/// use numdiff::forward_difference::shessian;
///
/// let f = |x: &Vec<f64>| x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3);
/// let x0 = vec![5.0, 8.0];
///
/// let hess: Mat<f64> = shessian(&f, &x0, Some(0.0001));
/// let hess_true: Mat<f64> = Mat::from_row_slice(
///     2,
///     2,
///     &[
///         20000.0,
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         30.0 * 8.0_f64.sin() * 8.0_f64.cos().powi(2) - 15.0 * 8.0_f64.sin().powi(3)
///     ]
/// );
///
/// assert_arrays_equal_to_decimal!(hess, hess_true, -1);
/// ```
pub fn shessian<V>(f: &impl Fn(&V) -> f64, x0: &V, h: Option<f64>) -> V::MatrixNxN
where
    V: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Determine the dimension of x.
    let n = x0.len();

    // Preallocate the matrix to store the Hessian.
    let mut hess = x0.new_matrix_n_by_n();

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Variable to store the absolute step size in the kth direction.
    let mut dxk: f64;

    // Variables to store the original value of the evaluation point in the jth and kth directions.
    let mut x0j: f64;
    let mut x0k: f64;

    // Vectors to store absolute step sizes (a) and function evaluations (b).
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];

    // Populate "a" and "b".
    for k in 0..n {
        // Original value of x₀ in the kth direction.
        x0k = x0[k];

        // Absolute step size in the kth direction.
        dxk = h * (1.0 + x0[k].abs());

        // Step in the kth direction.
        x0[k] += dxk;

        // Function evaluation.
        b[k] = f(&x0);

        // Reset x₀.
        x0[k] = x0k;

        // Store Δxₖ in a.
        a[k] = dxk;
    }

    // Evaluate the Hessian, iterating over the upper triangular elements.
    for k in 0..n {
        for j in k..n {
            // Original value of x₀ in the jth and kth directions.
            x0j = x0[j];
            x0k = x0[k];

            // Step in the jth and kth directions.
            x0[j] += a[j];
            x0[k] += a[k];

            // Evaluate the (j,k)th element of the Hessian.
            hess[(j, k)] = (f(&x0) - b[j] - b[k] + f0) / (a[j] * a[k]);

            // Evaluate the (k,j)th element of the Hessian via symmetry.
            hess[(k, j)] = hess[(j, k)];

            // Reset x₀.
            x0[j] = x0j;
            x0[k] = x0k;
        }
    }

    // Return the result.
    hess
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::{Mat, Matrix};
    use nalgebra::{SMatrix, SVector};
    use ndarray::{Array1, Array2};
    use numtest::*;

    #[test]
    fn test_shessian_1() {
        let f = |x: &Vec<f64>| x[0].powi(3);
        let x0 = vec![2.0];
        let hess = |x: &Vec<f64>| Mat::from_row_slice(1, 1, &[6.0 * x[0]]);
        assert_arrays_equal_to_decimal!(shessian(&f, &x0, None), hess(&x0), 4);
    }

    #[test]
    fn test_shessian_2() {
        let f = |x: &SVector<f64, 2>| x[0].powi(2) + x[1].powi(3);
        let x0 = SVector::from_row_slice(&[1.0, 2.0]);
        let hess = |x: &SVector<f64, 2>| {
            SMatrix::<f64, 2, 2>::from_row_slice(&[2.0, 0.0, 0.0, 6.0 * x[1]])
        };
        assert_arrays_equal_to_decimal!(shessian(&f, &x0, None), hess(&x0), 4);
    }

    #[test]
    fn test_shessian_3() {
        let f = |x: &Array1<f64>| x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3);
        let x0 = Array1::from_slice(&[1.0, 2.0]);
        let hess = |x: &Array1<f64>| {
            Array2::<f64>::from_row_slice(
                2,
                2,
                &[
                    20.0 * x[0].powi(3) * x[1],
                    5.0 * x[0].powi(4) + 3.0 * x[1].sin().powi(2) * x[1].cos(),
                    5.0 * x[0].powi(4) + 3.0 * x[1].sin().powi(2) * x[1].cos(),
                    6.0 * x[0] * x[1].sin() * x[1].cos().powi(2) - 3.0 * x[0] * x[1].sin().powi(3),
                ],
            )
        };
        assert_arrays_equal_to_decimal!(shessian(&f, &x0, None), hess(&x0), 3);
    }
}
