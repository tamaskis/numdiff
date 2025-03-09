use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Hessian of a multivariate, vector-valued function using the central difference approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function, $\mathbf{}:\mathbb{R}^{n}\to\mathbb{R}^{m}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Hessian of $\mathbf{f}$ with respect to $\mathbf{x}$, evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\mathbf{H}(\mathbf{f}(\mathbf{x}_{0}))\in\mathbb{R}^{m\times n\times n}$$
///
/// The returned result is stored as a length-$m$ [`Vec`] of $n\times n$ matrices.
///
/// # Note
///
/// This function performs $2n(n+1)$ evaluations of $f(x)$.
///
/// # Example
///
/// Approximate the Hessian of
///
/// $$
/// \mathbf{f}(\mathbf{x})=
/// \begin{bmatrix}
///     f_{0}(\mathbf{x}) \\\\
///     f_{1}(\mathbf{x})
/// \end{bmatrix}=
/// \begin{bmatrix}
///     x_{0}^{5}x_{1}+x_{0}\sin^{3}{x_{1}} \\\\
///     x_{0}^{3}+x_{1}^{4}-3x_{0}^{2}x_{1}^{2}
/// \end{bmatrix}
/// $$
///
/// at $\mathbf{x}=\mathbf{x}_{0}=(5,8)^{T}$, and compare the result to the true result of
///
/// $$
/// \mathbf{H}\left(\mathbf{x}_{0}\right)=\left[\mathbf{H}\left(f\_{0}(\mathbf{x}\_{0})\right),
/// \\;\mathbf{H}\left(f\_{1}(\mathbf{x}\_{0})\right)\right]
/// $$
///
/// where
///
/// $$
/// \begin{aligned}
///     \mathbf{H}\left(f\_{0}(\mathbf{x}\_{0})\right)&=
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
///     \end{bmatrix} \\\\
///     \mathbf{H}\left(f\_{1}(\mathbf{x}\_{0})\right)&=
///     \begin{bmatrix}
///         6x_{0}-6x_{1}^{2} & -12x_{0}x_{1} \\\\
///         -12x_{0}x_{1} & 12x_{1}^{2}-6x_{0}^{2}
///     \end{bmatrix}
///     \bigg\rvert_{\mathbf{x}=(5,8)^{T}} \\\\
///     &=
///     \begin{bmatrix}
///         6(5)-6(8)^{2} & -12(5)(8) \\\\
///         -12(5)(8) & 12(8)^{2}-6(5)^{2}
///     \end{bmatrix} \\\\
///     &=
///     \begin{bmatrix}
///         -354 & -480 \\\\
///         -480 & 618
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
/// use numdiff::central_difference::vhessian;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| vec![
///     x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3),
///     x[0].powi(3) + x[1].powi(4) - 3.0 * x[0].powi(2) * x[1].powi(2)
/// ];
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 8.0];
///
/// // Approximate the Hessian of f(x) at the evaluation point.
/// let hess: Vec<Mat<f64>> = vhessian(&f, &x0, None);
///
/// // True Hessian of f(x) at the evaluation point.
/// let hess_f0_true: Mat<f64> = Mat::from_row_slice(
///     2,
///     2,
///     &[
///         20000.0,
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         30.0 * 8.0_f64.sin() * 8.0_f64.cos().powi(2) - 15.0 * 8.0_f64.sin().powi(3)
///     ]
/// );
/// let hess_f1_true: Mat<f64> = Mat::from_row_slice(2, 2, &[-354.0, -480.0, -480.0, 618.0]);
/// let hess_true: Vec<Mat<f64>> = vec![hess_f0_true, hess_f1_true];
///
/// // Check the accuracy of the Hessian approximation.
/// assert_arrays_equal_to_decimal!(hess[0], hess_true[0], 3);
/// assert_arrays_equal_to_decimal!(hess[1], hess_true[1], 3);
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
/// use nalgebra::{dvector, DMatrix, DVector, SMatrix, SVector};
/// use ndarray::{array, Array1, Array2};
/// use numtest::*;
///
/// use numdiff::central_difference::vhessian;
///
/// let hess_f0_true: Mat<f64> = Mat::from_row_slice(
///     2,
///     2,
///     &[
///         20000.0,
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         30.0 * 8.0_f64.sin() * 8.0_f64.cos().powi(2) - 15.0 * 8.0_f64.sin().powi(3)
///     ]
/// );
/// let hess_f1_true: Mat<f64> = Mat::from_row_slice(2, 2, &[-354.0, -480.0, -480.0, 618.0]);
/// let hess_true: Vec<Mat<f64>> = vec![hess_f0_true, hess_f1_true];
///
/// // nalgebra::DVector
/// let f_dvector = |x: &DVector<f64>| dvector![
///     x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3),
///     x[0].powi(3) + x[1].powi(4) - 3.0 * x[0].powi(2) * x[1].powi(2)
/// ];
/// let x0_dvector: DVector<f64> = dvector![5.0, 8.0];
/// let hess_dvector: Vec<DMatrix<f64>> = vhessian(&f_dvector, &x0_dvector, None);
/// assert_arrays_equal_to_decimal!(hess_dvector[0], hess_true[0], 3);
/// assert_arrays_equal_to_decimal!(hess_dvector[1], hess_true[1], 3);
///
/// // nalgebra::SVector
/// let f_svector = |x: &SVector<f64, 2>| SVector::<f64, 2>::from_row_slice(&[
///     x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3),
///     x[0].powi(3) + x[1].powi(4) - 3.0 * x[0].powi(2) * x[1].powi(2)
/// ]);
/// let x0_svector: SVector<f64, 2> = SVector::from_row_slice(&[5.0, 8.0]);
/// let hess_svector: Vec<SMatrix<f64, 2, 2>> = vhessian(&f_svector, &x0_svector, None);
/// assert_arrays_equal_to_decimal!(hess_svector[0], hess_true[0], 3);
/// assert_arrays_equal_to_decimal!(hess_svector[1], hess_true[1], 3);
///
/// // ndarray::Array1
/// let f_array1 = |x: &Array1<f64>| array![
///     x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3),
///     x[0].powi(3) + x[1].powi(4) - 3.0 * x[0].powi(2) * x[1].powi(2)
/// ];
/// let x0_array1: Array1<f64> = array![5.0, 8.0];
/// let hess_array1: Vec<Array2<f64>> = vhessian(&f_array1, &x0_array1, None);
/// assert_arrays_equal_to_decimal!(hess_array1[0], hess_true[0], 3);
/// assert_arrays_equal_to_decimal!(hess_array1[1], hess_true[1], 3);
///
/// // faer::Mat
/// let f_mat = |x: &FMat<f64>| FMat::from_slice(
///     &[
///         x[(0, 0)].powi(5) * x[(1, 0)] + x[(0, 0)] * x[(1, 0)].sin().powi(3),
///         x[(0, 0)].powi(3) + x[(1, 0)].powi(4) - 3.0 * x[(0, 0)].powi(2) * x[(1, 0)].powi(2)
///     ]
/// );
/// let x0_mat: FMat<f64> = FMat::from_slice(&[5.0, 8.0]);
/// let hess_mat: Vec<FMat<f64>> = vhessian(&f_mat, &x0_mat, None);
/// assert_arrays_equal_to_decimal!(hess_mat[0].as_row_slice(), hess_true[0], 3);
/// assert_arrays_equal_to_decimal!(hess_mat[1].as_row_slice(), hess_true[1], 3);
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
/// use numdiff::central_difference::vhessian;
///
/// let f = |x: &Vec<f64>| vec![
///     x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3),
///     x[0].powi(3) + x[1].powi(4) - 3.0 * x[0].powi(2) * x[1].powi(2)
/// ];
/// let x0 = vec![5.0, 8.0];
///
/// let hess: Vec<Mat<f64>> = vhessian(&f, &x0, Some(0.001));
/// let hess_f0_true: Mat<f64> = Mat::from_row_slice(
///     2,
///     2,
///     &[
///         20000.0,
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         3125.0 + 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos(),
///         30.0 * 8.0_f64.sin() * 8.0_f64.cos().powi(2) - 15.0 * 8.0_f64.sin().powi(3)
///     ]
/// );
/// let hess_f1_true: Mat<f64> = Mat::from_row_slice(2, 2, &[-354.0, -480.0, -480.0, 618.0]);
/// let hess_true: Vec<Mat<f64>> = vec![hess_f0_true, hess_f1_true];
///
/// assert_arrays_equal_to_decimal!(hess[0], hess_true[0], 1);
/// assert_arrays_equal_to_decimal!(hess[1], hess_true[1], 3);
/// ```
pub fn vhessian<V, U>(f: &impl Fn(&V) -> U, x0: &V, h: Option<f64>) -> Vec<V::MatrixNxN>
where
    V: Vector<f64>,
    U: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Determine the dimension of x.
    let n = x0.len();

    // Variable to store the number of rows in the Hessian.
    let mut m = 0;

    // Variable to store the Hessian.
    let mut hess: Vec<V::MatrixNxN> = vec![x0.new_matrix_n_by_n(); m];

    // Variables to store the original values of the evaluation points in the jth and kth
    // directions.
    let mut x0j: f64;
    let mut x0k: f64;

    // Variable to store the (j,k)th and (k,j)th elements of each Hessian.
    let mut hess_jk: U;

    // Vector to store absolute step sizes.
    let mut a = vec![0.0; n];

    // Populate vector of absolute step sizes.
    for (k, ak) in a.iter_mut().enumerate().take(n) {
        *ak = h * (1.0 + x0.vget(k).abs());
    }

    // Variables to store evaluations of f(x) at various perturbed points.
    let mut b: U;
    let mut c: U;
    let mut d: U;
    let mut e: U;

    // Track whether the vectors of Hessians has been initialized.
    let mut not_initialized = true;

    // Evaluate the Hessian, iterating over the upper triangular elements.
    for k in 0..n {
        for j in k..n {
            // Original value of the evaluation point in the jth and kth directions.
            x0j = x0.vget(j);
            x0k = x0.vget(k);

            // Step forward in the jth and kth directions.
            x0.vset(j, x0.vget(j) + a[j]);
            x0.vset(k, x0.vget(k) + a[k]);
            b = f(&x0);
            x0.vset(j, x0j);
            x0.vset(k, x0k);

            // Step forward in the jth direction and backward in the kth direction.
            x0.vset(j, x0.vget(j) + a[j]);
            x0.vset(k, x0.vget(k) - a[k]);
            c = f(&x0);
            x0.vset(j, x0j);
            x0.vset(k, x0k);

            // Step backward in the jth direction and forward in the kth direction.
            x0.vset(j, x0.vget(j) - a[j]);
            x0.vset(k, x0.vget(k) + a[k]);
            d = f(&x0);
            x0.vset(j, x0j);
            x0.vset(k, x0k);

            // Step backward in the jth and kth directions.
            x0.vset(j, x0.vget(j) - a[j]);
            x0.vset(k, x0.vget(k) - a[k]);
            e = f(&x0);
            x0.vset(j, x0j);
            x0.vset(k, x0k);

            // Evaluate the (j,k)th and (k,j)th elements of each Hessian.
            hess_jk = b.sub(&c).sub(&d).add(&e).div(4.0 * a[j] * a[k]);

            // In the very first iteration, determine the number of rows in each Hessian.
            if not_initialized {
                m = hess_jk.len();
                hess = vec![x0.new_matrix_n_by_n(); m];
                not_initialized = false;
            }

            // Store the (j,k)th and (k,j)th elements of each Hessian.
            for (i, hess_i) in hess.iter_mut().enumerate().take(m) {
                hess_i[(j, k)] = hess_jk.vget(i);
                hess_i[(k, j)] = hess_jk.vget(i);
            }
        }
    }

    // Return the result.
    hess
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::{Mat, Matrix};
    use nalgebra::{DMatrix, DVector, SMatrix, SVector};
    use ndarray::{Array1, Array2};
    use numtest::*;

    #[test]
    fn test_vhessian_1() {
        let f = |x: &Vec<f64>| vec![x[0].powi(3)];
        let x0 = vec![2.0];
        let hess = |x: &Vec<f64>| vec![Mat::from_row_slice(1, 1, &[6.0 * x[0]])];
        assert_arrays_equal_to_decimal!(vhessian(&f, &x0, None)[0], hess(&x0)[0], 7);
    }

    #[test]
    fn test_vhessian_2() {
        let f =
            |x: &SVector<f64, 2>| SVector::<f64, 1>::from_row_slice(&[x[0].powi(2) + x[1].powi(3)]);
        let x0 = SVector::from_row_slice(&[1.0, 2.0]);
        let hess = |x: &SVector<f64, 2>| {
            vec![SMatrix::<f64, 2, 2>::from_row_slice(&[
                2.0,
                0.0,
                0.0,
                6.0 * x[1],
            ])]
        };
        assert_arrays_equal_to_decimal!(vhessian(&f, &x0, None)[0], hess(&x0)[0], 5);
    }

    #[test]
    fn test_vhessian_3() {
        let f = |x: &Array1<f64>| {
            Array1::<f64>::from_slice(&[x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3)])
        };
        let x0 = Array1::from_slice(&[1.0, 2.0]);
        let hess = |x: &Array1<f64>| {
            vec![Array2::<f64>::from_row_slice(
                2,
                2,
                &[
                    20.0 * x[0].powi(3) * x[1],
                    5.0 * x[0].powi(4) + 3.0 * x[1].sin().powi(2) * x[1].cos(),
                    5.0 * x[0].powi(4) + 3.0 * x[1].sin().powi(2) * x[1].cos(),
                    6.0 * x[0] * x[1].sin() * x[1].cos().powi(2) - 3.0 * x[0] * x[1].sin().powi(3),
                ],
            )]
        };
        assert_arrays_equal_to_decimal!(vhessian(&f, &x0, None)[0], hess(&x0)[0], 5);
    }

    #[test]
    fn test_vhessian_4() {
        let f = |x: &DVector<f64>| {
            DVector::<f64>::from_slice(&[
                x[0].powi(5) * x[1] + x[0] * x[1].sin().powi(3),
                x[0].powi(3) + x[1].powi(4) - 3.0 * x[0].powi(2) * x[1].powi(2),
            ])
        };
        let x0 = DVector::<f64>::from_slice(&[1.0, 2.0]);
        let hess = |x: &DVector<f64>| {
            vec![
                DMatrix::<f64>::from_row_slice(
                    2,
                    2,
                    &[
                        20.0 * x[0].powi(3) * x[1],
                        5.0 * x[0].powi(4) + 3.0 * x[1].sin().powi(2) * x[1].cos(),
                        5.0 * x[0].powi(4) + 3.0 * x[1].sin().powi(2) * x[1].cos(),
                        6.0 * x[0] * x[1].sin() * x[1].cos().powi(2)
                            - 3.0 * x[0] * x[1].sin().powi(3),
                    ],
                ),
                DMatrix::<f64>::from_row_slice(
                    2,
                    2,
                    &[
                        6.0 * x[0].powi(2) - 6.0 * x[1].powi(2),
                        -12.0 * x[0] * x[1],
                        -12.0 * x[0] * x[1],
                        12.0 * x[1].powi(2) - 6.0 * x[0].powi(2),
                    ],
                ),
            ]
        };
        assert_arrays_equal_to_decimal!(vhessian(&f, &x0, None)[0], hess(&x0)[0], 5);
        assert_arrays_equal_to_decimal!(vhessian(&f, &x0, None)[1], hess(&x0)[1], 5);
    }
}
