use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Second derivative of a univariate, vector-valued function using the forward difference
/// approximation.
///
/// # Arguments
///
/// * `f` - Univariate, vector-valued function, $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$.
/// * `x0` - Evaluation point, $x_{0}\in\mathbb{R}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Second derivative of $\mathbf{f}$ with respect to $x$, evaluated at $x=x_{0}$.
///
/// $$\frac{d^{2}\mathbf{f}}{dx^{2}}\bigg\rvert_{x=x_{0}}\in\mathbb{R}^{m}$$
///
/// # Note
///
/// This function performs 3 evaluations of $f(x)$.
///
/// # Examples
///
/// ## Basic Example
///
/// Approximate the second derivative of
///
/// $$f(t)=\begin{bmatrix}\sin{t}\\\\\cos{t}\end{bmatrix}$$
///
/// at $t=1$, and compare the result to the true result of
///
/// $$\frac{d^{2}\mathbf{f}}{dt^{2}}\bigg\rvert_{t=1}=\begin{bmatrix}-\sin{(1)}\\\\-\cos{(1)}\end{bmatrix}$$
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::vderivative2;
///
/// // Define the function, f(t).
/// let f = |t: f64| vec![t.sin(), t.cos()];
///
/// // Approximate the second derivative of f(t) at the evaluation point.
/// let d2f: Vec<f64> = vderivative2(&f, 1.0, None);
///
/// // True second derivative of f(t) at the evaluation point.
/// let d2f_true: Vec<f64> = vec![-1.0_f64.sin(), -1.0_f64.cos()];
///
/// // Check the accuracy of the second derivative approximation.
/// assert_arrays_equal_to_decimal!(d2f, d2f_true, 3);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Approximate the second derivative of a parameterized vector function
///
/// $$f(t)=\begin{bmatrix}at^{2}+b\\\\ce^{t}+d\end{bmatrix}$$
///
/// where $a$, $b$, $c$, and $d$ are runtime parameters. Compare the result against the true second
/// derivative of
///
/// $$f''(t)=\begin{bmatrix}2a\\\\ce^{t}\end{bmatrix}$$
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::vderivative2;
///
/// // Runtime parameters.
/// let a = 1.5;
/// let b = -2.0;
/// let c = 0.8;
/// let d = 3.0;
///
/// // Define the parameterized function.
/// fn f_param(t: f64, a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
///     vec![a * t.powi(2) + b, c * t.exp() + d]
/// }
///
/// // Wrap the parameterized function with a closure that captures the parameters.
/// let f = |t: f64| f_param(t, a, b, c, d);
///
/// // True second derivative function.
/// let d2f_true = |t: f64| vec![2.0 * a, c * t.exp()];
///
/// // Approximate the second derivative at t = 1.0 and compare with true second derivative.
/// let d2f_at_1: Vec<f64> = vderivative2(&f, 1.0, None);
/// let d2f_at_1_true: Vec<f64> = d2f_true(1.0);
/// assert_arrays_equal_to_decimal!(d2f_at_1, d2f_at_1_true, 3);
/// ```
pub fn vderivative2<V>(f: &impl Fn(f64) -> V, x0: f64, h: Option<f64>) -> V
where
    V: Vector<f64>,
{
    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Absolute step size.
    let dx = h * (1.0 + x0.abs());

    // Evaluate the second derivative.
    (f(x0 + 2.0 * dx).sub(&f(x0 + dx).mul(2.0)).add(&f(x0))).div(dx.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;

    #[test]
    fn test_vderivative2() {
        let f = |x: f64| vec![x.sin(), x.cos()];
        let x0 = 2.0;
        let d2f = |x: f64| vec![-x.sin(), -x.cos()];
        assert_arrays_equal_to_decimal!(vderivative2(&f, x0, None), d2f(x0), 3);
    }

    #[test]
    fn test_vderivative2_polynomial() {
        let f = |x: f64| vec![x.powi(3), x.powi(4)];
        let x0 = 1.5;
        let d2f = |x: f64| vec![6.0 * x, 12.0 * x.powi(2)];
        assert_arrays_equal_to_decimal!(vderivative2(&f, x0, None), d2f(x0), 2);
    }
}
