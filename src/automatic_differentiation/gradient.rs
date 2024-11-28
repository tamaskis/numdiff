/// Get a function that returns the gradient of the provided multivariate, scalar-valued function.
///
/// The gradient is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `func_name` - Name of the function that will return the gradient of $f(\mathbf{x})$ at any
///                 point $\mathbf{x}\in\mathbb{R}^{n}$.
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform $n$ evaluations of $f(x)$ to evaluate its
/// gradient.
///
/// # Example
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use ndarray::{array, Array1};
/// use numtest::*;
///
/// use numdiff::{get_gradient, Dual, DualVector};
///
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
///     x[0].powi(5) + x[1].sin().powi(3)
/// }
/// let x0 = array![5.0, 8.0];
/// let g = |x: &Array1<f64>| array![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()];
/// get_gradient!(grad_approx, f);
/// assert_arrays_equal_to_decimal!(grad_approx(&x0), g(&x0), 16);
/// ```
#[macro_export]
macro_rules! get_gradient {
    ($func_name:ident, $generic_func:ident) => {
        /// Gradient of a multivariate, scalar-valued function `f: ℝⁿ → ℝ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_gradient!` macro.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, `x₀ ∈ ℝⁿ`.
        ///
        /// # Returns
        ///
        /// Gradient of `f` with respect to `x`, evaluated at `x = x₀`.
        ///
        /// `∇f(x₀) ∈ ℝⁿ`
        fn $func_name<S, V>(x0: &V) -> V::Vectorf64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Preallocate the vector to store the gradient.
            let mut g: V::Vectorf64 = x0.new_vector_f64();

            // Dual version of the evaluation point.
            let mut x0_dual = x0.clone().to_dual_vector();

            // Variable to store the original value of the evaluation point in the kth direction.
            let mut x0k: Dual;

            // Evaluate the gradient.
            for k in 0..x0_dual.len() {
                // Original value of the evaluation point in the kth direction.
                x0k = x0_dual[k];

                // Step forward in the dual kth direction.
                x0_dual[k] = Dual::new(x0k.get_real(), 1.0);

                // Partial derivative of f with respect to xₖ.
                g[k] = $generic_func(&x0_dual).get_dual();

                // Reset the evaluation point.
                x0_dual[k] = x0k;
            }

            // Return the result.
            g
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::automatic_differentiation::dual::Dual;
    use crate::automatic_differentiation::dual_vector::DualVector;
    use linalg_traits::{Scalar, Vector};
    use nalgebra::DVector;
    use ndarray::{array, Array1};
    use numtest::*;

    #[test]
    fn test_gradient_1() {
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x[0].powi(2)
        }
        let x0 = vec![2.0];
        let g = |x: &Vec<f64>| vec![2.0 * x[0]];
        get_gradient!(g_autodiff, f);
        let grad_autodiff: Vec<f64> = g_autodiff(&x0);
        let grad_exact: Vec<f64> = g(&x0);
        assert_arrays_equal_to_decimal!(grad_autodiff, grad_exact, 16);
    }

    #[test]
    fn test_gradient_2() {
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x[0].powi(2) + x[1].powi(3)
        }
        let x0: DVector<f64> = DVector::from_slice(&[1.0, 2.0]);
        let g = |x: &DVector<f64>| DVector::<f64>::from_slice(&[2.0 * x[0], 3.0 * x[1].powi(2)]);
        get_gradient!(g_autodiff, f);
        let grad_autodiff: DVector<f64> = g_autodiff(&x0);
        let grad_exact: DVector<f64> = g(&x0);
        assert_arrays_equal_to_decimal!(grad_autodiff, grad_exact, 16);
    }

    #[test]
    fn test_gradient_3() {
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x[0].powi(5) + x[1].sin().powi(3)
        }
        let x0 = array![5.0, 8.0];
        let g = |x: &Array1<f64>| array![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()];
        get_gradient!(g_autodiff, f);
        let grad_autodiff: Array1<f64> = g_autodiff(&x0);
        let grad_exact: Array1<f64> = g(&x0);
        assert_arrays_equal_to_decimal!(grad_autodiff, grad_exact, 16);
    }
}
