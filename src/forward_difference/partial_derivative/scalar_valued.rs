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
///   indexing (e.g. $\mathbf{x}=\left(x_{0},...,x_{k},...,x_{n-1}\right)^{T}$).
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
/// This function performs 2 evaluations of $f(\mathbf{x})$.
///
/// # Examples
///
/// ## Basic Example
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
/// `ndarray::Array1`, `faer::Mat`, or any other type of vector that implements the
/// `linalg_traits::Vector` trait.
///
/// ```
/// use faer::Mat;
/// use linalg_traits::Vector;  // to provide from_slice method for faer::Mat
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
///
/// // faer::Mat
/// let f_mat = |x: &Mat<f64>| x[(0, 0)].powi(3) * x[(1, 0)].sin();
/// let x0_mat: Mat<f64> = Mat::from_slice(&[5.0, 1.0]);
/// let pf_mat: f64 = spartial_derivative(&f_mat, &x0_mat, k, None);
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
///
/// ## Example Passing Runtime Parameters
///
/// Approximate the partial derivative of a parameterized function
///
/// $$f(\mathbf{x})=ax_{0}^{2}+bx_{1}^{2}+cx_{0}x_{1}+d\sin(ex_{0})$$
///
/// where $a$, $b$, $c$, $d$, and $e$ are runtime parameters. The partial derivatives are:
///
/// * $\dfrac{\partial f}{\partial x_{0}}=2ax_{0}+cx_{1}+de\cos(ex_{0})$
/// * $\dfrac{\partial f}{\partial x_{1}}=2bx_{1}+cx_{0}$
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::spartial_derivative;
///
/// // Runtime parameters.
/// let a = 1.5;
/// let b = 2.0;
/// let c = 0.8;
/// let d = 3.0;
/// let e = 0.5;
///
/// // Define the parameterized function.
/// fn f_param(x: &Vec<f64>, a: f64, b: f64, c: f64, d: f64, e: f64) -> f64 {
///     a * x[0].powi(2) + b * x[1].powi(2) + c * x[0] * x[1] + d * (e * x[0]).sin()
/// }
///
/// // Wrap the parameterized function with a closure that captures the parameters.
/// let f = |x: &Vec<f64>| f_param(x, a, b, c, d, e);
///
/// // Evaluation point.
/// let x0 = vec![1.0, -0.5];
///
/// // True partial derivative functions.
/// let df_dx0_true = |x: &[f64]| 2.0 * a * x[0] + c * x[1] + d * e * (e * x[0]).cos();
/// let df_dx1_true = |x: &[f64]| 2.0 * b * x[1] + c * x[0];
///
/// // Approximate ∂f/∂x₀ at x₀ and compare with true function.
/// let df_dx0: f64 = spartial_derivative(&f, &x0, 0, None);
/// let expected_df_dx0 = df_dx0_true(&x0);
/// assert_equal_to_decimal!(df_dx0, expected_df_dx0, 5);
///
/// // Approximate ∂f/∂x₁ at x₀ and compare with true function.
/// let df_dx1: f64 = spartial_derivative(&f, &x0, 1, None);
/// let expected_df_dx1 = df_dx1_true(&x0);
/// assert_equal_to_decimal!(df_dx1, expected_df_dx1, 6);
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

    // Original value of the evaluation point in the kth direction.
    let x0k = x0.vget(k);

    // Absolute step size in the kth direction.
    let dxk = h * (1.0 + x0k.abs());

    // Step forward in the kth direction.
    x0.vset(k, x0k + dxk);

    // Evaluate the partial derivative of f with respect to xₖ.
    (f(&x0) - f0) / dxk
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    use numtest::*;

    #[test]
    fn test_spartial_derivative_1() {
        let f = |x: &Vec<f64>| x[0].powi(2);
        let x0 = vec![2.0];
        let k = 0;
        let dfk = |x: &Vec<f64>| 2.0 * x[0];
        assert_equal_to_decimal!(spartial_derivative(&f, &x0, k, None), dfk(&x0), 7);
    }

    #[test]
    fn test_spartial_derivative_2() {
        let f = |x: &SVector<f64, 2>| x[0].powi(3) * x[1].powi(3);
        let x0: SVector<f64, 2> = SVector::from_slice(&[3.0, 2.0]);
        let k = 1;
        let dfk = |x: &SVector<f64, 2>| 3.0 * x[0].powi(3) * x[1].powi(2);
        assert_equal_to_decimal!(spartial_derivative(&f, &x0, k, None), dfk(&x0), 5);
    }
}
