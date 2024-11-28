/// TODO.
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
        /// Gradient of a multivariate, scalar-valued function using the forward difference approximation.
        ///
        /// # Arguments
        ///
        /// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
        /// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
        ///
        /// # Returns
        ///
        /// Gradient of $f$ with respect to $\mathbf{x}$, evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
        ///
        /// $$\nabla f(\mathbf{x}_{0})\in\mathbb{R}^{n}$$
        ///
        /// // TODO more docs
        fn $func_name<S, V>(x0: &V) -> V::Vectorf64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // TODO: need a method to create a vector of dual numbers
            // TODO: need a method to take the dual portion of a vector of dual numbers

            let mut result: V::Vectorf64 = x0.new_vector_f64();
            let mut x0_dual = x0.clone().to_dual_vector();
            let mut x0k: Dual;

            for k in 0..x0_dual.len() {
                x0k = x0_dual[k];
                x0_dual[k] += Dual::new(0.0, 1.0);
                result[k] = $generic_func(&x0_dual).get_dual();
                x0_dual[k] = x0k;
            }
            result
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

    // TODO clean up these tests
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
        get_gradient!(grad_approx, f);
        assert_arrays_equal_to_decimal!(grad_approx(&x0), g(&x0), 16);
    }

    #[test]
    fn test_gradient_3() {
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x[0].powi(5) + x[1].sin().powi(3)
        }
        let x0 = array![5.0, 8.0];
        let g = |x: &Array1<f64>| array![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()];
        get_gradient!(grad_approx, f);
        assert_arrays_equal_to_decimal!(grad_approx(&x0), g(&x0), 16);
    }
}
