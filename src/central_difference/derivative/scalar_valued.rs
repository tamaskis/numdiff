use crate::constants::CBRT_EPS;

/// Derivative of a univariate, scalar-valued function using the central difference approximation.
///
/// # Arguments
///
/// * `f` - Univariate, scalar-valued function, $f:\mathbb{R}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $x_{0}\in\mathbb{R}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Derivative of $f$ with respect to $x$, evaluated at $x=x_{0}$.
///
/// $$\frac{df}{dx}\bigg\rvert_{x=x_{0}}\in\mathbb{R}$$
///
/// # Note
///
/// This function performs 2 evaluations of $f(x)$.
///
/// # Example
///
/// Approximate the derivative of
///
/// $$f(x)=x^{3}$$
///
/// at $x=2$, and compare the result to the true result of $f'(2)=12$.
///
/// ```
/// use numtest::*;
///
/// use numdiff::central_difference::sderivative;
///
/// // Define the function, f(x).
/// let f = |x: f64| x.powi(3);
///
/// // Approximate the derivative of f(x) at the evaluation point.
/// let df: f64 = sderivative(&f, 2.0, None);
///
/// // Check the accuracy of the derivative approximation.
/// assert_equal_to_decimal!(df, 12.0, 9);
/// ```
///
/// We can also modify the relative step size. Choosing a coarser relative step size, we get a worse
/// approximation.
///
/// ```
/// use numtest::*;
///
/// use numdiff::central_difference::sderivative;
///
/// let f = |x: f64| x.powi(3);
/// let df: f64 = sderivative(&f, 2.0, Some(0.001));
/// assert_equal_to_decimal!(df, 12.0, 5);
/// ```
pub fn sderivative(f: &impl Fn(f64) -> f64, x0: f64, h: Option<f64>) -> f64 {
    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Absolute step size.
    let dx = h * (1.0 + x0.abs());

    // Evaluate the derivative.
    (f(x0 + dx) - f(x0 - dx)) / (2.0 * dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;
    use numtest::*;
    use std::f64::consts::PI;

    #[cfg(feature = "trig")]
    use trig::Trig;

    #[test]
    fn test_product_rule() {
        // f(x) and f'(x).
        let f = |x: f64| x.powi(3);
        let df = |x: f64| 3.0 * x.powi(2);

        // g(x) and g'(x).
        let g = |x: f64| x.sin();
        let dg = |x: f64| x.cos();

        // h(x) = f(x)g(x) and h'(x) = f'(x)g(x) + f(x)g'(x).
        let h = |x: f64| f(x) * g(x);
        let dh = |x: f64| df(x) * g(x) + f(x) * dg(x);

        // Approximation of h'(x).
        let dh_approx = |x: f64| sderivative(&h, x, None);

        // Test approximation of h'(x) against true h'(x).
        assert_equal_to_decimal!(dh_approx(-1.5), dh(-1.5), 9);
        assert_equal_to_decimal!(dh_approx(1.5), dh(1.5), 9);
    }

    #[test]
    fn test_quotient_rule() {
        // f(x) and f'(x).
        let f = |x: f64| x.powi(3);
        let df = |x: f64| 3.0 * x.powi(2);

        // g(x) and g'(x).
        let g = |x: f64| x.sin();
        let dg = |x: f64| x.cos();

        // h(x) = f(x) / g(x) and h'(x) = (g(x)f'(x) - f(x)g'(x)) / (g(x))².
        let h = |x: f64| f(x) / g(x);
        let dh = |x: f64| (g(x) * df(x) - f(x) * dg(x)) / g(x).powi(2);

        // Approximation of h'(x).
        let dh_approx = |x: f64| sderivative(&h, x, None);

        // Test approximation of h'(x) against true h'(x).
        assert_equal_to_decimal!(dh_approx(-1.5), dh(-1.5), 9);
        assert_equal_to_decimal!(dh_approx(1.5), dh(1.5), 9);
    }

    #[test]
    fn test_chain_rule_one_composition() {
        // f(x) and f'(x).
        let f = |x: f64| x.powi(3);
        let df = |x: f64| 3.0 * x.powi(2);

        // g(x) and g'(x).
        let g = |x: f64| x.sin();
        let dg = |x: f64| x.cos();

        // h(x) = g(f(x)) and h'(x) = [g'(f(x))][f'(x)].
        let h = |x: f64| g(f(x));
        let dh = |x: f64| dg(f(x)) * df(x);

        // Approximation of h'(x).
        let dh_approx = |x: f64| sderivative(&h, x, None);

        // Test approximation of h'(x) against true h'(x).
        assert_equal_to_decimal!(dh_approx(-1.5), dh(-1.5), 8);
        assert_equal_to_decimal!(dh_approx(1.5), dh(1.5), 8);
    }

    #[test]
    fn test_chain_rule_two_compositions() {
        // f(x) and f'(x).
        let f = |x: f64| x.powi(3);
        let df = |x: f64| 3.0 * x.powi(2);

        // g(x) and g'(x).
        let g = |x: f64| x.sin();
        let dg = |x: f64| x.cos();

        // h(x) and h'(x).
        let h = |x: f64| 5.0 / x.powi(2);
        let dh = |x: f64| -10.0 / x.powi(3);

        // j(x) = h(g(f(x))) and j'(x) = [h'(g(f(x)))][g'(f(x))][f'(x)].
        let j = |x: f64| h(g(f(x)));
        let dj = |x: f64| dh(g(f(x))) * dg(f(x)) * df(x);

        // Approximation of j'(x).
        let dj_approx = |x: f64| sderivative(&j, x, None);

        // Test approximation of j'(x) against true j'(x).
        assert_equal_to_decimal!(dj_approx(-1.5), dj(-1.5), 2);
        assert_equal_to_decimal!(dj_approx(1.5), dj(1.5), 2);
    }

    #[test]
    fn test_sderivative_polynomial() {
        assert_equal_to_decimal!(
            sderivative(&|_x: f64| 1.0, 2.0, None),
            test_utils::polyi_deriv(0, 2.0),
            16
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x, 2.0, None),
            test_utils::polyi_deriv(1, 2.0),
            12
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(2), 2.0, None),
            test_utils::polyi_deriv(2, 2.0),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(3), 2.0, None),
            test_utils::polyi_deriv(3, 2.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(4), 2.0, None),
            test_utils::polyi_deriv(4, 2.0),
            8
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(7), 2.0, None),
            test_utils::polyi_deriv(7, 2.0),
            6
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(-1), 2.0, None),
            test_utils::polyi_deriv(-1, 2.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(-2), 2.0, None),
            test_utils::polyi_deriv(-2, 2.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(-3), 2.0, None),
            test_utils::polyi_deriv(-3, 2.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powi(-7), 2.0, None),
            test_utils::polyi_deriv(-7, 2.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powf(1.0 / 3.0), 2.0, None),
            test_utils::polyf_deriv(1.0 / 3.0, 2.0),
            16
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powf(7.0 / 3.0), 2.0, None),
            test_utils::polyf_deriv(7.0 / 3.0, 2.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powf(-1.0 / 3.0), 2.0, None),
            test_utils::polyf_deriv(-1.0 / 3.0, 2.0),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.powf(-7.0 / 3.0), 2.0, None),
            test_utils::polyf_deriv(-7.0 / 3.0, 2.0),
            10
        );
    }

    #[test]
    fn test_sderivative_square_root() {
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.sqrt(), 0.5, None),
            test_utils::sqrt_deriv(0.5),
            10
        );
        assert_equal_to_decimal!(sderivative(&|x: f64| x.sqrt(), 1.0, None), 0.5, 11);
        assert_equal_to_decimal!(
            sderivative(&|x: f64| x.sqrt(), 1.5, None),
            test_utils::sqrt_deriv(1.5),
            11
        );
    }

    #[test]
    fn test_sderivative_exponential() {
        let f = |x: f64| x.exp();
        assert_equal_to_decimal!(sderivative(&f, -1.0, None), test_utils::exp_deriv(-1.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::exp_deriv(0.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::exp_deriv(1.0), 10);
    }

    #[test]
    fn test_sderivative_power() {
        let b: f64 = 5.0;
        let f = |x: f64| b.powf(x);
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::power_deriv(b, -1.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, 0.0, None),
            test_utils::power_deriv(b, 0.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, 1.0, None),
            test_utils::power_deriv(b, 1.0),
            9
        );
    }

    #[test]
    fn test_sderivative_natural_logarithm() {
        let f = |x: f64| x.ln();
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::ln_deriv(0.5), 9);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::ln_deriv(1.0), 10);
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::ln_deriv(1.5), 10);
    }

    #[test]
    fn test_sderivative_base_10_logarithm() {
        let f = |x: f64| x.log10();
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::log10_deriv(0.5), 10);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::log10_deriv(1.0), 10);
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::log10_deriv(1.5), 11);
    }

    #[test]
    fn test_sderivative_sine() {
        let f = |x: f64| x.sin();
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::sin_deriv(0.0), 11);
        assert_equal_to_decimal!(
            sderivative(&f, PI / 4.0, None),
            test_utils::sin_deriv(PI / 4.0),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, PI / 2.0, None),
            test_utils::sin_deriv(PI / 2.0),
            16
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 4.0, None),
            test_utils::sin_deriv(3.0 * PI / 4.0),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, PI, None), test_utils::sin_deriv(PI), 10);
        assert_equal_to_decimal!(
            sderivative(&f, 5.0 * PI / 4.0, None),
            test_utils::sin_deriv(5.0 * PI / 4.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 2.0, None),
            test_utils::sin_deriv(3.0 * PI / 2.0),
            15
        );
        assert_equal_to_decimal!(
            sderivative(&f, 7.0 * PI / 4.0, None),
            test_utils::sin_deriv(7.0 * PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 2.0 * PI, None),
            test_utils::sin_deriv(2.0 * PI),
            9
        );
    }

    #[test]
    fn test_sderivative_cosine() {
        let f = |x: f64| x.cos();
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::cos_deriv(0.0), 16);
        assert_equal_to_decimal!(
            sderivative(&f, PI / 4.0, None),
            test_utils::cos_deriv(PI / 4.0),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, PI / 2.0, None),
            test_utils::cos_deriv(PI / 2.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 4.0, None),
            test_utils::cos_deriv(3.0 * PI / 4.0),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, PI, None), test_utils::cos_deriv(PI), 16);
        assert_equal_to_decimal!(
            sderivative(&f, 5.0 * PI / 4.0, None),
            test_utils::cos_deriv(5.0 * PI / 4.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 2.0, None),
            test_utils::cos_deriv(3.0 * PI / 2.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 7.0 * PI / 4.0, None),
            test_utils::cos_deriv(7.0 * PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 2.0 * PI, None),
            test_utils::cos_deriv(2.0 * PI),
            15
        );
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_tangent() {
        let f = |x: f64| x.tan();
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::tan_deriv(0.0), 11);
        assert_equal_to_decimal!(
            sderivative(&f, PI / 4.0, None),
            test_utils::tan_deriv(PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 4.0, None),
            test_utils::tan_deriv(3.0 * PI / 4.0),
            9
        );
        assert_equal_to_decimal!(sderivative(&f, PI, None), test_utils::tan_deriv(PI), 9);
        assert_equal_to_decimal!(
            sderivative(&f, 5.0 * PI / 4.0, None),
            test_utils::tan_deriv(5.0 * PI / 4.0),
            8
        );
        assert_equal_to_decimal!(
            sderivative(&f, 7.0 * PI / 4.0, None),
            test_utils::tan_deriv(7.0 * PI / 4.0),
            8
        );
        assert_equal_to_decimal!(
            sderivative(&f, 2.0 * PI, None),
            test_utils::tan_deriv(2.0 * PI),
            9
        );
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_cosecant() {
        let f = |x: f64| x.csc();
        assert_equal_to_decimal!(
            sderivative(&f, PI / 4.0, None),
            test_utils::csc_deriv(PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 4.0, None),
            test_utils::csc_deriv(3.0 * PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 5.0 * PI / 4.0, None),
            test_utils::csc_deriv(5.0 * PI / 4.0),
            8
        );
        assert_equal_to_decimal!(
            sderivative(&f, 7.0 * PI / 4.0, None),
            test_utils::csc_deriv(7.0 * PI / 4.0),
            8
        );
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_secant() {
        let f = |x: f64| x.sec();
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::sec_deriv(0.0), 16);
        assert_equal_to_decimal!(
            sderivative(&f, PI / 4.0, None),
            test_utils::sec_deriv(PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 4.0, None),
            test_utils::sec_deriv(3.0 * PI / 4.0),
            9
        );
        assert_equal_to_decimal!(sderivative(&f, PI, None), test_utils::sec_deriv(PI), 16);
        assert_equal_to_decimal!(
            sderivative(&f, 5.0 * PI / 4.0, None),
            test_utils::sec_deriv(5.0 * PI / 4.0),
            8
        );
        assert_equal_to_decimal!(
            sderivative(&f, 7.0 * PI / 4.0, None),
            test_utils::sec_deriv(7.0 * PI / 4.0),
            8
        );
        assert_equal_to_decimal!(
            sderivative(&f, 2.0 * PI, None),
            test_utils::sec_deriv(2.0 * PI),
            15
        );
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_cotangent() {
        let f = |x: f64| x.cot();
        assert_equal_to_decimal!(
            sderivative(&f, PI / 4.0, None),
            test_utils::cot_deriv(PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, PI / 2.0, None),
            test_utils::cot_deriv(PI / 2.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 4.0, None),
            test_utils::cot_deriv(3.0 * PI / 4.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 5.0 * PI / 4.0, None),
            test_utils::cot_deriv(5.0 * PI / 4.0),
            8
        );
        assert_equal_to_decimal!(
            sderivative(&f, 3.0 * PI / 2.0, None),
            test_utils::cot_deriv(3.0 * PI / 2.0),
            9
        );
        assert_equal_to_decimal!(
            sderivative(&f, 7.0 * PI / 4.0, None),
            test_utils::cot_deriv(7.0 * PI / 4.0),
            8
        );
    }

    #[test]
    fn test_sderivative_inverse_sine() {
        let f = |x: f64| x.asin();
        assert_equal_to_decimal!(
            sderivative(&f, -0.5, None),
            test_utils::asin_deriv(-0.5),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::asin_deriv(0.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::asin_deriv(0.5), 10);
    }

    #[test]
    fn test_sderivative_inverse_cosine() {
        let f = |x: f64| x.acos();
        assert_equal_to_decimal!(
            sderivative(&f, -0.5, None),
            test_utils::acos_deriv(-0.5),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::acos_deriv(0.0), 10);
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::acos_deriv(0.5), 10);
    }

    #[test]
    fn test_sderivative_inverse_tangent() {
        let f = |x: f64| x.atan();
        assert_equal_to_decimal!(
            sderivative(&f, -1.5, None),
            test_utils::atan_deriv(-1.5),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::atan_deriv(-1.0),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, -0.5, None),
            test_utils::atan_deriv(-0.5),
            11
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::atan_deriv(0.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::atan_deriv(0.5), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::atan_deriv(1.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::atan_deriv(1.5), 11);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_cosecant() {
        let f = |x: f64| x.acsc();
        assert_equal_to_decimal!(
            sderivative(&f, -1.5, None),
            test_utils::acsc_deriv(-1.5),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::acsc_deriv(1.5), 10);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_secant() {
        let f = |x: f64| x.asec();
        assert_equal_to_decimal!(
            sderivative(&f, -1.5, None),
            test_utils::asec_deriv(-1.5),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::asec_deriv(1.5), 10);
    }

    // Note: we do not test the derivative of arccot at 0 because the central difference
    // approximation is numerical unstable about x = 0 for this specific function.
    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_cotangent() {
        let f = |x: f64| x.acot();
        assert_equal_to_decimal!(
            sderivative(&f, -1.5, None),
            test_utils::acot_deriv(-1.5),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::acot_deriv(-1.0),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, -0.5, None),
            test_utils::acot_deriv(-0.5),
            11
        );
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::acot_deriv(0.5), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::acot_deriv(1.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::acot_deriv(1.5), 11);
    }

    #[test]
    fn test_sderivative_hyperbolic_sine() {
        let f = |x: f64| x.sinh();
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::sinh_deriv(-1.0),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::sinh_deriv(0.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::sinh_deriv(1.0), 10);
    }

    #[test]
    fn test_sderivative_hyperbolic_cosine() {
        let f = |x: f64| x.cosh();
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::cosh_deriv(-1.0),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::cosh_deriv(0.0), 16);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::cosh_deriv(1.0), 10);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_tangent() {
        let f = |x: f64| x.tanh();
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::tanh_deriv(-1.0),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::tanh_deriv(0.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::tanh_deriv(1.0), 10);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_cosecant() {
        let f = |x: f64| x.csch();
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::csch_deriv(-1.0),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::csch_deriv(1.0), 10);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_secant() {
        let f = |x: f64| x.sech();
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::sech_deriv(-1.0),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::sech_deriv(0.0), 16);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::sech_deriv(1.0), 10);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_cotangent() {
        let f = |x: f64| x.coth();
        assert_equal_to_decimal!(sderivative(&f, -1.0, None), test_utils::coth_deriv(-1.0), 9);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::coth_deriv(1.0), 9);
    }

    #[test]
    fn test_sderivative_inverse_hyperbolic_sine() {
        let f = |x: f64| x.asinh();
        assert_equal_to_decimal!(
            sderivative(&f, -1.5, None),
            test_utils::asinh_deriv(-1.5),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::asinh_deriv(-1.0),
            11
        );
        assert_equal_to_decimal!(
            sderivative(&f, -0.5, None),
            test_utils::asinh_deriv(-0.5),
            11
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::asinh_deriv(0.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::asinh_deriv(0.5), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::asinh_deriv(1.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::asinh_deriv(1.5), 11);
    }

    #[test]
    fn test_sderivative_inverse_hyperbolic_cosine() {
        let f = |x: f64| x.acosh();
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::acosh_deriv(1.5), 10);
    }

    #[test]
    fn test_sderivative_inverse_hyperbolic_tangent() {
        let f = |x: f64| x.atanh();
        assert_equal_to_decimal!(
            sderivative(&f, -0.5, None),
            test_utils::atanh_deriv(-0.5),
            10
        );
        assert_equal_to_decimal!(sderivative(&f, 0.0, None), test_utils::atanh_deriv(0.0), 11);
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::atanh_deriv(0.5), 10);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_hyperbolic_cosecant() {
        let f = |x: f64| x.acsch();
        assert_equal_to_decimal!(
            sderivative(&f, -1.5, None),
            test_utils::acsch_deriv(-1.5),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, -1.0, None),
            test_utils::acsch_deriv(-1.0),
            10
        );
        assert_equal_to_decimal!(
            sderivative(&f, -0.5, None),
            test_utils::acsch_deriv(-0.5),
            9
        );
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::acsch_deriv(0.5), 9);
        assert_equal_to_decimal!(sderivative(&f, 1.0, None), test_utils::acsch_deriv(1.0), 10);
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::acsch_deriv(1.5), 10);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_hyperbolic_secant() {
        let f = |x: f64| x.asech();
        assert_equal_to_decimal!(sderivative(&f, 0.5, None), test_utils::asech_deriv(0.5), 9);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_hyperbolic_cotangent() {
        let f = |x: f64| x.acoth();
        assert_equal_to_decimal!(
            sderivative(&f, -1.5, None),
            test_utils::acoth_deriv(-1.5),
            9
        );
        assert_equal_to_decimal!(sderivative(&f, 1.5, None), test_utils::acoth_deriv(1.5), 9);
    }
}
