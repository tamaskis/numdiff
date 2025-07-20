/// Get a function that returns the directional derivative of the provided multivariate,
/// scalar-valued function.
///
/// The directional derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `func_name` - Name of the function that will return the directional derivative of
///   $f(\mathbf{x})$ at any point $\mathbf{x}\in\mathbb{R}^{n}$ and in any direction
///   $\mathbf{v}\in\mathbb{R}^{n}$.
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $f(\mathbf{x})$ to evaluate its
/// directional derivative.
///
/// # Example
///
/// Compute the directional derivative of
///
/// $$f(\mathbf{x})=x_{0}^{5}+\sin^{3}{x_{1}}$$
///
/// at $\mathbf{x}=(5,8)^{T}$ in the direction of $\mathbf{v}=(10,20)^{T}$, and compare the result
/// to the expected result of
///
/// $$\nabla f_{(10,20)^{T}}\left((5,8)^{T}\right)=31250+60\sin^{2}{(8)}\cos{(8)}$$
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_directional_derivative, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
///     x.vget(0).powi(5) + x.vget(1).sin().powi(3)
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 8.0];
///
/// // Define the direction of differentiation.
/// let v = vec![10.0, 20.0];
///
/// // Autogenerate the function "df_v" that can be used to compute the directional derivative of
/// // f(x) at any point x and along any specified direction v.
/// get_directional_derivative!(f, df_v);
///
/// // Function defining the true directional derivative of f(x).
/// let df_v_true = |x: &Vec<f64>, v: &Vec<f64>| vec![
///     5.0 * x[0].powi(4),
///     3.0 * x[1].sin().powi(2) * x[1].cos()
/// ].dot(v);
///
/// // Verify that the directional derivative function obtained using get_directional_derivative!
/// // computes the directional derivative correctly.
/// assert_eq!(df_v(&x0, &v), df_v_true(&x0, &v));
/// ```
///
/// #### Using other vector types
///
/// The function produced by `get_directional_derivative!` can accept _any_ type for `x0` and `v`,
/// as long as they implement the `linalg_traits::Vector` trait.
///
/// ```
/// use faer::Mat;
/// use linalg_traits::{Scalar, Vector};
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
/// use numtest::*;
///
/// use numdiff::{get_directional_derivative, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
///     x.vget(0).powi(5) + x.vget(1).sin().powi(3)
/// }
///
/// // Autogenerate the function "df_v" that can be used to compute the directional derivative of
/// // f(x) at any point x and along any specified direction v.
/// get_directional_derivative!(f, df_v);
///
/// // nalgebra::DVector
/// let x0: DVector<f64> = dvector![5.0, 8.0];
/// let v: DVector<f64> = dvector![10.0, 20.0];
/// let df_v_eval: f64 = df_v(&x0, &v);
///
/// // nalgebra::SVector
/// let x0: SVector<f64, 2> = SVector::from_slice(&[5.0, 8.0]);
/// let v: SVector<f64, 2> = SVector::from_slice(&[10.0, 20.0]);
/// let df_v_eval: f64 = df_v(&x0, &v);
///
/// // ndarray::Array1
/// let x0: Array1<f64> = array![5.0, 8.0];
/// let v: Array1<f64> = array![10.0, 20.0];
/// let df_v_eval: f64 = df_v(&x0, &v);
///
/// // faer::Mat
/// let x0: Mat<f64> = Mat::from_slice(&[5.0, 8.0]);
/// let v: Mat<f64> = Mat::from_slice(&[10.0, 20.0]);
/// let df_v_eval: f64 = df_v(&x0, &v);
/// ```
#[macro_export]
macro_rules! get_directional_derivative {
    ($f:ident, $func_name:ident) => {
        /// Directional derivative of a multivariate, scalar-valued function `f: ℝⁿ → ℝ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_directional_derivative!` macro.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, `x₀ ∈ ℝⁿ`.
        /// * `v` - Vector defining the direction of differentiation, `v ∈ ℝⁿ`.
        ///
        /// # Returns
        ///
        /// Directional derivative of `f` with respect to `x` in the direction of `v`, evaluated at
        /// `x = x₀`.
        ///
        /// `∇ᵥf(x₀) = ∇f(x₀)ᵀv ∈ ℝ`
        fn $func_name<S, V>(x0: &V, v: &V) -> f64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Promote the evaluation point to a vector of dual numbers.
            let x0_dual = x0.clone().to_dual_vector();

            // Promote the direction of differentiation to a vector of dual numbers.
            let v_dual = v.clone().to_dual_vector();

            // Evaluate the directional derivative.
            $f(&x0_dual.add(&v_dual.mul(Dual::new(0.0, 1.0)))).get_dual()
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::automatic_differentiation::dual::Dual;
    use crate::automatic_differentiation::dual_vector::DualVector;
    use linalg_traits::{Scalar, Vector};
    use nalgebra::SVector;
    use ndarray::{Array1, array};

    #[test]
    fn test_directional_derivative_1() {
        // Function to find the directional derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x.vget(0).powi(2)
        }

        // Evaluation point.
        let x0 = vec![2.0];

        // Direction of differentiation.
        let v = vec![0.6];

        // True directional derivative function.
        let df_v = |x: &Vec<f64>, v: &Vec<f64>| 2.0 * x[0] * v[0];

        // Directional derivative function obtained via forward-mode automatic differentiation.
        get_directional_derivative!(f, df_v_autodiff);

        // Evaluate the directional derivative using both functions.
        let df_v_eval: f64 = df_v(&x0, &v);
        let df_v_eval_autodiff: f64 = df_v_autodiff(&x0, &v);

        // Test autodiff directional derivative against true directional derivative.
        assert_eq!(df_v_eval_autodiff, df_v_eval);
    }

    #[test]
    fn test_directional_derivative_2() {
        // Function to find the directional derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x.vget(0).powi(2) + x.vget(1).powi(3)
        }

        // Evaluation point.
        let x0: SVector<f64, 2> = SVector::from_slice(&[1.0, 2.0]);

        // Direction of differentiation.
        let v: SVector<f64, 2> = SVector::from_slice(&[3.0, 4.0]);

        // True directional derivative function.
        let df_v = |x: &SVector<f64, 2>, v: &SVector<f64, 2>| {
            SVector::<f64, 2>::from_slice(&[2.0 * x[0], 3.0 * x[1].powi(2)]).dot(v)
        };

        // Directional derivative function obtained via forward-mode automatic differentiation.
        get_directional_derivative!(f, df_v_autodiff);

        // Evaluate the directional derivative using both functions.
        let df_v_eval: f64 = df_v(&x0, &v);
        let df_v_eval_autodiff: f64 = df_v_autodiff(&x0, &v);

        // Test autodiff directional derivative against true directional derivative.
        assert_eq!(df_v_eval_autodiff, df_v_eval);
    }

    #[test]
    fn test_directional_derivative_3() {
        // Function to find the directional derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> S {
            x.vget(0).powi(5) + x.vget(1).sin().powi(3)
        }

        // Evaluation point.
        let x0: Array1<f64> = array![5.0, 8.0];

        // Direction of differentiation.
        let v: Array1<f64> = array![10.0, 20.0];

        // True directional derivative function.
        let df_v = |x: &Array1<f64>, v: &Array1<f64>| {
            array![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()].dot(v)
        };

        // Directional derivative function obtained via forward-mode automatic differentiation.
        get_directional_derivative!(f, df_v_autodiff);

        // Evaluate the directional derivative using both functions.
        let df_v_eval: f64 = df_v(&x0, &v);
        let df_v_eval_autodiff: f64 = df_v_autodiff(&x0, &v);

        // Test autodiff directional derivative against true directional derivative.
        assert_eq!(df_v_eval_autodiff, df_v_eval);
    }
}
