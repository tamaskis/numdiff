use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Partial derivative of a multivariate, vector-valued function using the central difference
/// approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function, $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `k` - Element of $\mathbf{x}$ to differentiate with respect to. Note that this uses 0-based
///         indexing (e.g. $\mathbf{x}=\left(x_{0},...,x_{k},...,x_{n-1}\right)^{T}$).
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Partial derivative of $\mathbf{f}$ with respect to $x_{k}$, evaluated at
/// $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\frac{d\mathbf{f}}{dx_{k}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}\in\mathbb{R}^{m}$$
///
/// # Note
///
/// This function performs 2 evaluations of $\mathbf{f}(\mathbf{x})$.
///
/// # Example
///
/// Approximate the partial derivative of
///
/// $$\mathbf{f}(\mathbf{x})=\begin{bmatrix}\sin{x_{0}}\sin{x_{1}}\\\\\cos{x_{0}}\cos{x_{1}}\end{bmatrix}$$
///
/// with respect to $x_{0}$ at $\mathbf{x}=(1,2)^{T}$, and compare the result to the
/// true result of
///
/// $$
/// \frac{\partial\mathbf{f}}{\partial x_{0}}\bigg\rvert_{\mathbf{x}=(1,2)^{T}}=
/// \begin{bmatrix}
/// \cos{(1)}\sin{(2)} \\\\
/// -\sin{(1)}\cos{(2)}
/// \end{bmatrix}
/// $$
///
/// #### Using standard vectors
///
/// ```
/// use numtest::*;
///
/// use numdiff::central_difference::vpartial_derivative;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| vec![x[0].sin() * x[1].sin(), x[0].cos() * x[1].cos()];
///
/// // Define the evaluation point.
/// let x0 = vec![1.0, 2.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 0;
///
/// // Approximate the partial derivative of f(x) with respect to x₀ at the evaluation point.
/// let pf: Vec<f64> = vpartial_derivative(&f, &x0, k, None);
///
/// // True partial derivative of f(x) with respect to x₀ at the evaluation point.
/// let pf_true: Vec<f64> = vec![1.0_f64.cos() * 2.0_f64.sin(), -1.0_f64.sin() * 2.0_f64.cos()];
///
/// // Check the accuracy of the partial derivative approximation.
/// assert_arrays_equal_to_decimal!(pf, pf_true, 11);
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
/// use numdiff::central_difference::vpartial_derivative;
///
/// let k = 0;
///
/// let pf_true: Vec<f64> = vec![1.0_f64.cos() * 2.0_f64.sin(), -1.0_f64.sin() * 2.0_f64.cos()];
///
/// // nalgebra::DVector
/// let f_dvector = |x: &DVector<f64>| dvector![x[0].sin() * x[1].sin(), x[0].cos() * x[1].cos()];
/// let x0_dvector: DVector<f64> = dvector![1.0, 2.0];
/// let pf_dvector: DVector<f64> = vpartial_derivative(&f_dvector, &x0_dvector, k, None);
/// assert_arrays_equal_to_decimal!(pf_dvector, pf_true, 11);
///
/// // nalgebra::SVector
/// let f_svector = |x: &SVector<f64, 2>| {
///     SVector::from_row_slice(&[x[0].sin() * x[1].sin(), x[0].cos() * x[1].cos()])
/// };
/// let x0_svector: SVector<f64, 2> = SVector::from_row_slice(&[1.0, 2.0]);
/// let pf_svector: SVector<f64, 2> = vpartial_derivative(&f_svector, &x0_svector, k, None);
/// assert_arrays_equal_to_decimal!(pf_svector, pf_true, 11);
///
/// // ndarray::Array1
/// let f_array1 = |x: &Array1<f64>| array![x[0].sin() * x[1].sin(), x[0].cos() * x[1].cos()];
/// let x0_array1: Array1<f64> = array![1.0, 2.0];
/// let pf_array1: Array1<f64> = vpartial_derivative(&f_array1, &x0_array1, k, None);
/// assert_arrays_equal_to_decimal!(pf_array1, pf_true, 11);
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
/// use numdiff::central_difference::vpartial_derivative;
///
/// let f = |x: &Vec<f64>| vec![x[0].sin() * x[1].sin(), x[0].cos() * x[1].cos()];
/// let x0 = vec![1.0, 2.0];
/// let k = 0;
///
/// let pf: Vec<f64> = vpartial_derivative(&f, &x0, k, Some(0.001));
/// let pf_true: Vec<f64> = vec![1.0_f64.cos() * 2.0_f64.sin(), -1.0_f64.sin() * 2.0_f64.cos()];
///
/// assert_arrays_equal_to_decimal!(pf, pf_true, 6);
/// ```
pub fn vpartial_derivative<V, U>(f: &impl Fn(&V) -> U, x0: &V, k: usize, h: Option<f64>) -> U
where
    V: Vector<f64>,
    U: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Store the original value of the evaluation point in the kth direction.
    let x0k = x0[k];

    // Absolute step size in the kth direction.
    let dxk = h * (1.0 + x0[k].abs());

    // Step forward in the kth direction.
    x0[k] += dxk;
    let f1 = f(&x0);

    // Step backward in the kth direction.
    x0[k] = x0k - dxk;
    let f2 = f(&x0);

    // Evaluate the partial derivative of f with respect to xₖ.
    f1.sub(&f2).div(2.0 * dxk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dvector, DVector, SVector};
    use ndarray::{array, Array1};
    use numtest::*;

    #[test]
    fn test_vpartial_derivative_1() {
        let f = |x: &Vec<f64>| vec![x[0].powi(2)];
        let x0 = vec![2.0];
        let k = 0;
        let dfk = |x: &Vec<f64>| vec![2.0 * x[0]];
        assert_arrays_equal_to_decimal!(vpartial_derivative(&f, &x0, k, None), dfk(&x0), 11);
    }

    #[test]
    fn test_vpartial_derivative_2() {
        let f = |x: &DVector<f64>| dvector![x[0].powi(4), x[1].powi(3)];
        let x0 = dvector![1.0, 2.0];
        let k = 1;
        let dfk = |x: &DVector<f64>| dvector![0.0, 3.0 * x[1].powi(2)];
        assert_arrays_equal_to_decimal!(vpartial_derivative(&f, &x0, k, None), dfk(&x0), 9);
    }

    #[test]
    fn test_vpartial_derivative_3() {
        let f = |x: &Vec<f64>| vec![x[0].powi(3) * x[1].powi(3)];
        let x0 = vec![3.0, 2.0];
        let k = 1;
        let dfk = |x: &Vec<f64>| vec![3.0 * x[0].powi(3) * x[1].powi(2)];
        assert_arrays_equal_to_decimal!(vpartial_derivative(&f, &x0, k, None), dfk(&x0), 8);
    }

    #[test]
    fn test_vpartial_derivative_4() {
        let f = |x: &Array1<f64>| array![x[0].powi(4), x[1].powi(3)];
        let x0 = array![1.0, 2.0];
        let k = 1;
        let dfk = |x: &Array1<f64>| array![0.0, 3.0 * x[1].powi(2)];
        assert_arrays_equal_to_decimal!(vpartial_derivative(&f, &x0, k, None), dfk(&x0), 9);
    }

    #[test]
    fn test_vpartial_derivative_5() {
        let f = |x: &SVector<f64, 3>| {
            SVector::<f64, 4>::from_row_slice(&[
                x[0],
                5.0 * x[2],
                4.0 * x[1].powi(2) - 2.0 * x[2],
                x[2] * x[0].sin(),
            ])
        };
        let x0: SVector<f64, 3> = SVector::from_row_slice(&[5.0, 6.0, 7.0]);
        let k = 2;
        let dfk =
            |x: &SVector<f64, 3>| SVector::<f64, 4>::from_row_slice(&[0.0, 5.0, -2.0, x[0].sin()]);
        assert_arrays_equal_to_decimal!(vpartial_derivative(&f, &x0, k, None), dfk(&x0), 10);
    }
}
