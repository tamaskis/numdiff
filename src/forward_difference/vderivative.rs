use crate::constants::SQRT_EPS;
use linalg_traits::Vector;

/// Derivative of a univariate, scalar-valued ($f:\mathbb{R}\to\mathbb{R}$) function using the
/// forward difference approximation.
///
/// # Arguments
///
/// * `x0` - Evaluation point, $x_{0}\in\mathbb{R}$.
///
/// # Returns
///
/// Derivative of $f:\mathbb{R}\to\mathbb{R}$ with respect to $x$, evaluated at $x=x_{0}$.
pub fn vderivative<T: Vector>(f: impl Fn(f64) -> T, x0: f64, h: Option<f64>) -> T {
    // Default the relative step size to h = √(ε) if not specified.
    let h = h.unwrap_or(*SQRT_EPS);

    // Absolute step size.
    let dx = h * (1.0 + x0.abs());

    // Evaluate the derivative.
    (f(x0 + dx).sub(&f(x0))).div(dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;

    #[test]
    fn test_vderivative() {
        let f = |x: f64| vec![x.sin(), x.cos()];
        let df = |x: f64| vec![x.cos(), -x.sin()];
        let x0 = 2.0;
        assert_arrays_equal_to_decimal!(vderivative(f, x0, None), df(x0), 7);
    }
}
