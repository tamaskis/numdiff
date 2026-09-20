use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Second-order partial derivative of a multivariate, scalar-valued function using the forward
/// difference approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `k` - Element of $\mathbf{x}$ to differentiate with respect to twice. Note that this uses
///   0-based indexing (e.g. $\mathbf{x}=\left(x_{0},...,x_{k},...,x_{n-1}\right)^{T}$).
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Second-order partial derivative of $f$ with respect to $x_{k}$, evaluated at
/// $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\frac{\partial^{2}f}{\partial x_{k}^{2}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}\in\mathbb{R}$$
///
/// # Note
///
/// This function performs 3 evaluations of $f(\mathbf{x})$.
///
/// # Examples
///
/// ## Basic Example
///
/// Approximate the second-order partial derivative of
///
/// $$f(x,y)=x^{4}+2x^{2}y+y^{3}$$
///
/// with respect to $x$ at $(x,y)=(2,1)$, and compare the result to the true result of
///
/// $$\frac{\partial^{2}f}{\partial x^{2}}\bigg\rvert_{(x,y)=(2,1)}=12x^{2}+4y\bigg\rvert_{(x,y)=(2,1)}=52$$
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::spartial_derivative2;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| x[0].powi(4) + 2.0 * x[0].powi(2) * x[1] + x[1].powi(3);
///
/// // Define the evaluation point.
/// let x0 = vec![2.0, 1.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with
/// // respect to.
/// let k = 0;
///
/// // Approximate the second-order partial derivative of f(x) with respect to x₀.
/// let d2f: f64 = spartial_derivative2(&f, &x0, k, None);
///
/// // Check the accuracy of the second-order partial derivative approximation.
/// assert_equal_to_decimal!(d2f, 52.0, 2);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Approximate the second-order partial derivative of a parameterized function
///
/// $$f(\mathbf{x})=ax_{0}^{2}+bx_{1}^{2}+cx_{0}x_{1}$$
///
/// where $a$, $b$, and $c$ are runtime parameters. Compare the result against the true second-order
/// partial derivative of $\dfrac{\partial^{2}f}{\partial x_{0}^{2}}=2a$.
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::spartial_derivative2;
///
/// // Runtime parameters.
/// let a = 1.5;
/// let b = 2.0;
/// let c = 0.8;
///
/// // Define the parameterized function.
/// fn f_param(x: &Vec<f64>, a: f64, b: f64, c: f64) -> f64 {
///     a * x[0].powi(2) + b * x[1].powi(2) + c * x[0] * x[1]
/// }
///
/// // Wrap the parameterized function with a closure that captures the parameters.
/// let f = |x: &Vec<f64>| f_param(x, a, b, c);
///
/// // Evaluation point.
/// let x0 = vec![1.0, -0.5];
///
/// // Approximate ∂²f/∂x₀² at x₀ and compare with the true result.
/// let d2f_dx0dx0: f64 = spartial_derivative2(&f, &x0, 0, None);
/// assert_equal_to_decimal!(d2f_dx0dx0, 2.0 * a, 2);
/// ```
pub fn spartial_derivative2<V>(f: &impl Fn(&V) -> f64, x0: &V, k: usize, h: Option<f64>) -> f64
where
    V: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Original value of the evaluation point in the kth direction.
    let x0k = x0[k];

    // Absolute step size in the kth direction.
    let dxk = h * (1.0 + x0k.abs());

    // Step forward once in the kth direction.
    x0[k] = x0k + dxk;
    let f1 = f(&x0);

    // Step forward again in the kth direction.
    x0[k] = x0k + 2.0 * dxk;
    let f2 = f(&x0);

    // Evaluate the second-order partial derivative of f with respect to xₖ.
    (f2 - 2.0 * f1 + f0) / dxk.powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "nalgebra")]
    use nalgebra::SVector;
    use numtest::*;

    #[test]
    fn test_spartial_derivative2_1() {
        let f = |x: &Vec<f64>| x[0].powi(4) + 2.0 * x[0].powi(2) * x[1] + x[1].powi(3);
        let x0 = vec![2.0, 1.0];
        assert_equal_to_decimal!(spartial_derivative2(&f, &x0, 0, None), 52.0, 2);
        assert_equal_to_decimal!(spartial_derivative2(&f, &x0, 1, None), 6.0 * x0[1], 3);
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_spartial_derivative2_2() {
        let f = |x: &SVector<f64, 2>| x[0].powi(4) + 2.0 * x[0].powi(2) * x[1] + x[1].powi(3);
        let x0 = SVector::from_row_slice(&[2.0, 1.0]);
        assert_equal_to_decimal!(spartial_derivative2(&f, &x0, 0, None), 52.0, 2);
        assert_equal_to_decimal!(spartial_derivative2(&f, &x0, 1, None), 6.0 * x0[1], 3);
    }
}
