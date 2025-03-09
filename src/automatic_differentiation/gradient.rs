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
/// The function produced by this macro will perform $n$ evaluations of $f(\mathbf{x})$ to evaluate
/// its gradient.
///
/// # Example
///
/// Compute the gradient of
///
/// $$f(\mathbf{x})=x_{0}^{5}+\sin^{3}{x_{1}}$$
///
/// at $\mathbf{x}=(5,8)^{T}$, and compare the result to the true result of
///
/// $$
/// \nabla f\left((5,8)^{T}\right)=
/// \begin{bmatrix}
///     3125 \\\\
///     3\sin^{2}{(8)}\cos{(8)}
/// \end{bmatrix}
/// $$
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_gradient, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
///     x.vget(0).powi(5) + x.vget(1).sin().powi(3)
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 8.0];
///
/// // Autogenerate the function "g" that can be used to compute the gradient of f(x) at any point
/// // x.
/// get_gradient!(f, g);
///
/// // Function defining the true gradient of f(x).
/// let g_true = |x: &Vec<f64>| vec![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()];
///
/// // Verify that the gradient function obtained using get_gradient! computes the gradient
/// // correctly.
/// assert_arrays_equal_to_decimal!(g(&x0), g_true(&x0), 16);
/// ```
///
/// #### Using other vector types
///
/// The function produced by `get_gradient!` can accept _any_ type for `x0`, as long as it
/// implements the `linalg_traits::Vector` trait.
///
/// ```
/// use faer::Mat;
/// use linalg_traits::{Scalar, Vector};
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
///
/// use numdiff::{get_gradient, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
///     x.vget(0).powi(5) + x.vget(1).sin().powi(3)
/// }
///
/// // Autogenerate the function "g" that can be used to compute the gradient of f(x) at any point
/// // x.
/// get_gradient!(f, g);
///
/// // nalgebra::DVector
/// let x0: DVector<f64> = dvector![5.0, 8.0];
/// let g_eval: DVector<f64> = g(&x0);
///
/// // nalgebra::SVector
/// let x0: SVector<f64, 2> = SVector::from_slice(&[5.0, 8.0]);
/// let g_eval: SVector<f64, 2> = g(&x0);
///
/// // ndarray::Array1
/// let x0: Array1<f64> = array![5.0, 8.0];
/// let g_eval: Array1<f64> = g(&x0);
///
/// // faer::Mat
/// let x0: Mat<f64> = Mat::from_slice(&[5.0, 8.0]);
/// let g_eval: Mat<f64> = g(&x0);
/// ```
#[macro_export]
macro_rules! get_gradient {
    ($f:ident, $func_name:ident) => {
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

            // Promote the evaluation point to a vector of dual numbers.
            let mut x0_dual = x0.clone().to_dual_vector();

            // Variable to store the original value of the evaluation point in the kth dual
            // direction.
            let mut x0k: Dual;

            // Evaluate the gradient.
            for k in 0..x0_dual.len() {
                // Original value of the evaluation point in the kth dual direction.
                x0k = x0_dual.vget(k);

                // Take a unit step forward in the kth dual direction.
                x0_dual.vset(k, Dual::new(x0k.get_real(), 1.0));

                // Partial derivative of f with respect to xₖ.
                g.vset(k, $f(&x0_dual).get_dual());

                // Reset the evaluation point.
                x0_dual.vset(k, x0k);
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
        // Function to find the gradient of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x.vget(0).powi(2)
        }

        // Evaluation point.
        let x0 = vec![2.0];

        // True gradient function.
        let g = |x: &Vec<f64>| vec![2.0 * x[0]];

        // Gradient function obtained via forward-mode automatic differentiation.
        get_gradient!(f, g_autodiff);

        // Evaluate the gradient using both functions.
        let g_eval_autodiff: Vec<f64> = g_autodiff(&x0);
        let g_eval: Vec<f64> = g(&x0);

        // Test autodiff gradient against true gradient.
        assert_arrays_equal_to_decimal!(g_eval_autodiff, g_eval, 16);
    }

    #[test]
    fn test_gradient_2() {
        // Function to find the gradient of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x.vget(0).powi(2) + x.vget(1).powi(3)
        }

        // Evaluation point.
        let x0: DVector<f64> = DVector::from_slice(&[1.0, 2.0]);

        // True gradient function.
        let g = |x: &DVector<f64>| DVector::<f64>::from_slice(&[2.0 * x[0], 3.0 * x[1].powi(2)]);

        // Gradient function obtained via forward-mode automatic differentiation.
        get_gradient!(f, g_autodiff);

        // Evaluate the gradient using both functions.
        let g_eval_autodiff: DVector<f64> = g_autodiff(&x0);
        let g_eval: DVector<f64> = g(&x0);

        // Test autodiff gradient against true gradient.
        assert_arrays_equal_to_decimal!(g_eval_autodiff, g_eval, 16);
    }

    #[test]
    fn test_gradient_3() {
        // Function to find the gradient of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x.vget(0).powi(5) + x.vget(1).sin().powi(3)
        }

        // Evaluation point.
        let x0 = array![5.0, 8.0];

        // True gradient function.
        let g = |x: &Array1<f64>| array![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()];

        // Gradient function obtained via forward-mode automatic differentiation.
        get_gradient!(f, g_autodiff);

        // Evaluate the gradient using both functions.
        let g_eval_autodiff: Array1<f64> = g_autodiff(&x0);
        let g_eval: Array1<f64> = g(&x0);

        // Test autodiff gradient against true gradient.
        assert_arrays_equal_to_decimal!(g_eval_autodiff, g_eval, 16);
    }
}
