use crate::constants::SQRT_EPS;
use linalg_traits::Vector;

/// Gradient of a multivariate, scalar-valued function using the forward difference approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`SQRT_EPS`].
///
/// # Returns
///
/// Gradient of $f$ with respect to $\mathbf{x}$, evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\nabla f(\mathbf{x}_{0})\in\mathbb{R}^{n}$$
///
/// # Note
///
/// This function performs $n+1$ evaluations of $f(\mathbf{x})$.
///
/// # Example
///
/// Approximate the gradient of
///
/// $$f(\mathbf{x})=x_{0}^{5}+\sin^{3}{x_{1}}$$
///
/// at $\mathbf{x}=(5,8)^{T}$, and compare the result to the true result of
///
/// $$
/// \nabla f\left((5,8)^{T}\right)=
/// \begin{bmatrix}
///     3125 \\\\
///     3\sin^{2}{(8)}\cos{(8)}
/// \end{bmatrix}
/// $$
///
/// #### Using standard vectors
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::gradient;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| x[0].powi(5) + x[1].sin().powi(3);
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 8.0];
///
/// // Approximate the gradient of f(x) at the evaluation point.
/// let grad: Vec<f64> = gradient(&f, &x0, None);
///
/// // True gradient of f(x) at the evaluation point.
/// let grad_true: Vec<f64> = vec![3125.0, 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos()];
///
/// // Check the accuracy of the gradient approximation.
/// assert_arrays_equal_to_decimal!(grad, grad_true, 4);
/// ```
///
/// #### Using other vector types
///
/// We can also use other types of vectors, such as `nalgebra::SVector`, `nalgebra::DVector`,
/// `ndarray::Array1`, or any other type of vector that implements the `linalg_traits::Vector`
/// trait.
///
/// ```
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
/// use numtest::*;
///
/// use numdiff::forward_difference::gradient;
///
/// let grad_true: Vec<f64> = vec![3125.0, 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos()];
///
/// // nalgebra::DVector
/// let f_dvector = |x: &DVector<f64>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0_dvector: DVector<f64> = dvector![5.0, 8.0];
/// let grad_dvector: DVector<f64> = gradient(&f_dvector, &x0_dvector, None);
/// assert_arrays_equal_to_decimal!(grad_dvector, grad_true, 4);
///
/// // nalgebra::SVector
/// let f_svector = |x: &SVector<f64,2>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0_svector: SVector<f64, 2> = SVector::from_row_slice(&[5.0, 8.0]);
/// let grad_svector: SVector<f64, 2> = gradient(&f_svector, &x0_svector, None);
/// assert_arrays_equal_to_decimal!(grad_svector, grad_true, 4);
///
/// // ndarray::Array1
/// let f_array1 = |x: &Array1<f64>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0_array1: Array1<f64> = array![5.0, 8.0];
/// let grad_array1: Array1<f64> = gradient(&f_array1, &x0_array1, None);
/// assert_arrays_equal_to_decimal!(grad_array1, grad_true, 4);
/// ```
///
/// #### Modifying the relative step size
///
/// We can also modify the relative step size. Choosing a coarser relative step size, we get a worse
/// approximation.
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::gradient;
///
/// let f = |x: &Vec<f64>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0 = vec![5.0, 8.0];
///
/// let grad: Vec<f64> = gradient(&f, &x0, Some(0.001));
/// let grad_true: Vec<f64> = vec![3125.0, 3.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos()];
///
/// assert_arrays_equal_to_decimal!(grad, grad_true, -1);
/// ```
pub fn gradient<V>(f: &impl Fn(&V) -> f64, x0: &V, h: Option<f64>) -> V
where
    V: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = √(ε) if not specified.
    let h = h.unwrap_or(*SQRT_EPS);

    // Determine the dimension of x.
    let n = x0.len();

    // Preallocate the vector to store the gradient.
    let mut g = V::new_with_length(n);

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Variable to store the absolute step size in the kth direction.
    let mut dxk: f64;

    // Variable to store the original value of the evaluation point in the kth direction.
    let mut x0k: f64;

    // Evaluate the gradient.
    for k in 0..n {
        // Original value of the evaluation point in the kth direction.
        x0k = x0[k];

        // Absolute step size in the kth direction.
        dxk = h * (1.0 + x0k.abs());

        // Step forward in the kth direction.
        x0[k] += dxk;

        // Partial derivative of f with respect to xₖ.
        g[k] = (f(&x0) - f0) / dxk;

        // Reset the evaluation point.
        x0[k] = x0k;
    }

    // Return the result.
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    use ndarray::{array, Array1};
    use numtest::*;

    #[test]
    fn test_gradient_1() {
        let f = |x: &Vec<f64>| x[0].powi(2);
        let x0 = vec![2.0];
        let g = |x: &Vec<f64>| vec![2.0 * x[0]];
        assert_arrays_equal_to_decimal!(gradient(&f, &x0, None), g(&x0), 7);
    }

    #[test]
    fn test_gradient_2() {
        let f = |x: &SVector<f64, 2>| x[0].powi(2) + x[1].powi(3);
        let x0: SVector<f64, 2> = SVector::from_slice(&[1.0, 2.0]);
        let g =
            |x: &SVector<f64, 2>| SVector::<f64, 2>::from_slice(&[2.0 * x[0], 3.0 * x[1].powi(2)]);
        assert_arrays_equal_to_decimal!(gradient(&f, &x0, None), g(&x0), 6);
    }

    #[test]
    fn test_gradient_3() {
        let f = |x: &Array1<f64>| x[0].powi(5) + x[1].sin().powi(3);
        let x0 = array![5.0, 8.0];
        let g = |x: &Array1<f64>| array![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()];
        assert_arrays_equal_to_decimal!(gradient(&f, &x0, None), g(&x0), 4);
    }
}
