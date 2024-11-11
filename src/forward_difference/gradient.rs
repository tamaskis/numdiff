use crate::constants::SQRT_EPS;
use linalg_traits::Vector;

/// Gradient of a multivariate, scalar-valued ($f:\mathbb{R}^{n}\to\mathbb{R}$) function using the
/// forward difference approximation.
///
/// # Arguments
///
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
///
/// # Returns
///
/// Gradient of $f:\mathbb{R}^{n}\to\mathbb{R}$ with respect to $\mathbf{x}$, evaluated at
/// $\mathbf{x}=\mathbf{x}_{0}$.
pub fn gradient<T: Vector>(f: impl Fn(&T) -> f64, x0: &T, h: Option<f64>) -> T {
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = √(ε) if not specified.
    let h = h.unwrap_or(*SQRT_EPS);

    // Determine the dimension of x.
    let n = x0.len();

    // Preallocate the vector to store the gradient.
    let mut g = T::new_with_length(n);

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Variable to store the absolute step size in the kth direction.
    let mut dxk: f64;

    // Variable store the original value of the evaluation point in the current direction.
    let mut x0k: f64;

    // Evaluate the gradient.
    for k in 0..n {
        // Original value of x₀ in the kth direction.
        x0k = x0[k];

        // Absolute step size.
        dxk = h * (1.0 + x0k.abs());

        // Step in the kth direction.
        x0[k] += dxk;

        // Partial derivative of f with respect to xₖ.
        g[k] = (f(&x0) - f0) / dxk;

        // Reset x₀.
        x0[k] = x0k;
    }

    // Return the result.
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;

    #[test]
    fn test_gradient_1() {
        let f = |x: &Vec<f64>| x[0].powi(2);
        let g = |x: &Vec<f64>| vec![2.0 * x[0]];
        let x0 = vec![2.0];
        assert_arrays_equal_to_decimal!(gradient(f, &x0, None), g(&x0), 7);
    }

    #[test]
    fn test_gradient_2() {
        let f = |x: &Vec<f64>| x[0].powi(2) + x[1].powi(3);
        let g = |x: &Vec<f64>| vec![2.0 * x[0], 3.0 * x[1].powi(2)];
        let x0 = vec![1.0, 2.0];
        assert_arrays_equal_to_decimal!(gradient(f, &x0, None), g(&x0), 6);
    }
}
