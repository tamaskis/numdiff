use crate::constants::SQRT_EPS;
use linalg_traits::Vector;

/// Partial derivative of a multivariate, scalar-valued function using the forward difference
/// approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `k` - Element of $\mathbf{x}$ to differentiate with respect to. Note that this uses 0-based
///         indexing (e.g. $\mathbf{x}=\left(x_{0},...,x_{k},...,x_{n-1}\right)^{T}$).
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`SQRT_EPS`].
///
/// # Returns
///
/// Partial derivative of $f$ with respect to $x_{k}$, evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\frac{df}{dx_{k}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}\in\mathbb{R}$$
///
/// # Note
///
/// This function performs 2 evaluations of $f(x)$.
///
/// # Example
///
/// Approximate the partial derivative of
///
/// $$f(x)=x^{3}\sin{y}$$
///
/// with respect to $y$ at $(x,y)=(5,1)$, and compare the result to the true result of
///
/// $$\frac{\partial f}{\partial y}\bigg\rvert_{(x,y)=(5,1)}=5^{3}\cos{(1)}$$
///
/// First, note that we can rewrite this function as
///
/// $$f(\mathbf{x})=x_{0}^{3}\sin{x_{1}}$$
///
/// where $\mathbf{x}=(x_{0},x_{1})^{T}$ (note that we use 0-based indexing to aid with the
/// computational implementation). We are then trying to find
///
/// $$\frac{\partial f}{\partial x_{1}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}$$
///
/// where $\mathbf{x}_{0}=(5,1)^{T}$.
///
/// #### Using standard vectors
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::spartial_derivative;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| x[0].powi(3) * x[1].sin();
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 1.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 1;
///
/// // Approximate the partial derivative of f(x) with respect to x₁ at the evaluation point.
/// let pf: f64 = spartial_derivative(&f, &x0, k, None);
///
/// // Check the accuracy of the partial derivative approximation.
/// assert_equal_to_decimal!(pf, 5.0_f64.powi(3) * 1.0_f64.cos(), 5);
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
/// use numdiff::forward_difference::spartial_derivative;
///
/// let k = 1;
///
/// let pf_true: f64 = 5.0_f64.powi(3) * 1.0_f64.cos();
///
/// // nalgebra::DVector
/// let f_dvector = |x: &DVector<f64>| x[0].powi(3) * x[1].sin();
/// let x0_dvector: DVector<f64> = dvector![5.0, 1.0];
/// let pf_dvector: f64 = spartial_derivative(&f_dvector, &x0_dvector, k, None);
/// assert_equal_to_decimal!(pf_dvector, pf_true, 5);
///
/// // nalgebra::SVector
/// let f_svector = |x: &SVector<f64, 2>| x[0].powi(3) * x[1].sin();
/// let x0_svector: SVector<f64, 2> = SVector::from_row_slice(&[5.0, 1.0]);
/// let pf_svector: f64 = spartial_derivative(&f_svector, &x0_svector, k, None);
/// assert_equal_to_decimal!(pf_svector, pf_true, 5);
///
/// // ndarray::Array1
/// let f_array1 = |x: &Array1<f64>| x[0].powi(3) * x[1].sin();
/// let x0_array1: Array1<f64> = array![5.0, 1.0];
/// let pf_array1: f64 = spartial_derivative(&f_array1, &x0_array1, k, None);
/// assert_equal_to_decimal!(pf_array1, pf_true, 5);
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
/// use numdiff::forward_difference::spartial_derivative;
///
/// let f = |x: &Vec<f64>| x[0].powi(3) * x[1].sin();
/// let x0 = vec![5.0, 1.0];
/// let k = 1;
///
/// let pf: f64 = spartial_derivative(&f, &x0, k, Some(0.001));
/// let pf_true: f64 = 5.0_f64.powi(3) * 1.0_f64.cos();
///
/// assert_equal_to_decimal!(pf, pf_true, 1);
/// ```
pub fn spartial_derivative<V>(f: &impl Fn(&V) -> f64, x0: &V, k: usize, h: Option<f64>) -> f64
where
    V: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = √(ε) if not specified.
    let h = h.unwrap_or(*SQRT_EPS);

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Absolute step size in the kth direction.
    let dxk = h * (1.0 + x0[k].abs());

    // Step in the kth direction.
    x0[k] += dxk;

    // Evaluate the partial derivative of f with respect to xₖ.
    (f(&x0) - f0) / dxk
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    use numtest::*;

    #[test]
    fn test_spartial_1() {
        let f = |x: &Vec<f64>| x[0].powi(2);
        let x0 = vec![2.0];
        let k = 0;
        let dfk = |x: &Vec<f64>| 2.0 * x[0];
        assert_equal_to_decimal!(spartial_derivative(&f, &x0, k, None), dfk(&x0), 7);
    }

    #[test]
    fn test_spartial_2() {
        let f = |x: &SVector<f64, 2>| x[0].powi(3) * x[1].powi(3);
        let x0: SVector<f64, 2> = SVector::from_slice(&[3.0, 2.0]);
        let k = 1;
        let dfk = |x: &SVector<f64, 2>| 3.0 * x[0].powi(3) * x[1].powi(2);
        assert_equal_to_decimal!(spartial_derivative(&f, &x0, k, None), dfk(&x0), 5);
    }
}
