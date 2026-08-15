use crate::constants::CBRT_EPS;

/// Second derivative of a univariate, scalar-valued function using the forward difference
/// approximation.
///
/// # Arguments
///
/// * `f` - Univariate, scalar-valued function, $f:\mathbb{R}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $x_{0}\in\mathbb{R}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Second derivative of $f$ with respect to $x$, evaluated at $x=x_{0}$.
///
/// $$\frac{d^{2}f}{dx^{2}}\bigg\rvert_{x=x_{0}}\in\mathbb{R}$$
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
/// $$f(x)=x^{3}$$
///
/// at $x=2$, and compare the result to the true result of $f''(2)=12$.
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::sderivative2;
///
/// // Define the function, f(x).
/// let f = |x: f64| x.powi(3);
///
/// // Approximate the second derivative of f(x) at the evaluation point.
/// let d2f: f64 = sderivative2(&f, 2.0, None);
///
/// // Check the accuracy of the second derivative approximation.
/// assert_equal_to_decimal!(d2f, 12.0, 3);
/// ```
///
/// We can also modify the relative step size. Choosing a coarser relative step size, we get a worse
/// approximation.
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::sderivative2;
///
/// let f = |x: f64| x.powi(3);
/// let d2f: f64 = sderivative2(&f, 2.0, Some(0.001));
/// assert_equal_to_decimal!(d2f, 12.0, 1);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Approximate the second derivative of a parameterized function
///
/// $$f(x)=ax^{2}+bx+c$$
///
/// where $a$, $b$, and $c$ are runtime parameters. Compare the result against the true second
/// derivative of $f''(x)=2a$.
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::sderivative2;
///
/// // Define the parameterized function.
/// fn f_param(x: f64, a: f64, b: f64, c: f64) -> f64 {
///     a * x.powi(2) + b * x + c
/// }
///
/// // Runtime parameters.
/// let a = 2.5;
/// let b = -1.3;
/// let c = 4.7;
///
/// // Wrap the parameterized function with a closure that captures the parameters.
/// let f = |x: f64| f_param(x, a, b, c);
///
/// // True second derivative function.
/// let d2f_true = |_x: f64| 2.0 * a;
///
/// // Approximate the second derivative at x = 1.0 and compare with true second derivative.
/// let d2f_at_1: f64 = sderivative2(&f, 1.0, None);
/// let d2f_at_1_true: f64 = d2f_true(1.0);
/// assert_equal_to_decimal!(d2f_at_1, d2f_at_1_true, 3);
/// ```
pub fn sderivative2(f: &impl Fn(f64) -> f64, x0: f64, h: Option<f64>) -> f64 {
    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Absolute step size.
    let dx = h * (1.0 + x0.abs());

    // Evaluate the second derivative.
    (f(x0 + 2.0 * dx) - 2.0 * f(x0 + dx) + f(x0)) / dx.powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;
    use linalg_traits::RealField;
    use numtest::*;
    use std::f64::consts::PI;

    #[test]
    fn test_product_rule() {
        // f(x), f'(x), and f"(x).
        let f = |x: f64| x.powi(3);
        let df = |x: f64| 3.0 * x.powi(2);
        let d2f = |x: f64| 6.0 * x;

        // g(x), g'(x), and g"(x).
        let g = |x: f64| x.sin();
        let dg = |x: f64| x.cos();
        let d2g = |x: f64| -x.sin();

        // h(x) = f(x)g(x) and h"(x) = f"(x)g(x) + 2f'(x)g'(x) + f(x)g"(x).
        let h = |x: f64| f(x) * g(x);
        let d2h = |x: f64| d2f(x) * g(x) + 2.0 * df(x) * dg(x) + f(x) * d2g(x);

        // Approximation of h"(x).
        let d2h_approx = |x: f64| sderivative2(&h, x, None);

        // Test approximation of h"(x) against true h"(x).
        assert_equal_to_decimal!(d2h_approx(-1.5), d2h(-1.5), 2);
        assert_equal_to_decimal!(d2h_approx(1.5), d2h(1.5), 2);
    }

    #[test]
    fn test_chain_rule_one_composition() {
        // f(x), f'(x), and f"(x).
        let f = |x: f64| x.powi(3);
        let df = |x: f64| 3.0 * x.powi(2);
        let d2f = |x: f64| 6.0 * x;

        // g(x), g'(x), and g"(x).
        let g = |x: f64| x.sin();
        let dg = |x: f64| x.cos();
        let d2g = |x: f64| -x.sin();

        // h(x) = g(f(x)) and h"(x) = g"(f(x))[f'(x)]² + g'(f(x))f"(x).
        let h = |x: f64| g(f(x));
        let d2h = |x: f64| d2g(f(x)) * df(x).powi(2) + dg(f(x)) * d2f(x);

        // Approximation of h"(x).
        let d2h_approx = |x: f64| sderivative2(&h, x, None);

        // Test approximation of h"(x) against true h"(x).
        assert_equal_to_decimal!(d2h_approx(-1.5), d2h(-1.5), 1);
        assert_equal_to_decimal!(d2h_approx(1.5), d2h(1.5), 1);
    }

    #[test]
    fn test_sderivative2_polynomial() {
        let f = |x: f64| x.powi(4);
        let d2f = |x: f64| 12.0 * x.powi(2);
        assert_equal_to_decimal!(sderivative2(&f, -2.0, None), d2f(-2.0), 2);
        assert_equal_to_decimal!(sderivative2(&f, 2.0, None), d2f(2.0), 2);
    }

    #[test]
    fn test_sderivative2_sine() {
        assert_equal_to_decimal!(
            sderivative2(&f64::sin, PI / 4.0, None),
            test_utils::sin_deriv2(PI / 4.0),
            3
        );
    }

    #[test]
    fn test_sderivative2_exponential() {
        assert_equal_to_decimal!(
            sderivative2(&f64::exp, 1.0, None),
            test_utils::exp_deriv2(1.0),
            3
        );
    }

    #[test]
    fn test_sderivative2_inverse_secant() {
        let f = |x: f64| x.asec();
        assert_equal_to_decimal!(sderivative2(&f, 1.5, None), test_utils::asec_deriv2(1.5), 2);
        assert_equal_to_decimal!(
            sderivative2(&f, -1.5, None),
            test_utils::asec_deriv2(-1.5),
            2
        );
    }

    #[test]
    fn test_sderivative2_inverse_cosecant() {
        let f = |x: f64| x.acsc();
        assert_equal_to_decimal!(sderivative2(&f, 1.5, None), test_utils::acsc_deriv2(1.5), 2);
        assert_equal_to_decimal!(
            sderivative2(&f, -1.5, None),
            test_utils::acsc_deriv2(-1.5),
            2
        );
    }
}
