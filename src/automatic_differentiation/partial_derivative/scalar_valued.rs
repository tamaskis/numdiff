/// Get a function that returns the partial derivative of the provided multivariate, scalar-valued
/// function.
///
/// The partial derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `func_name` - Name of the function that will return the partial derivative of $f(\mathbf{x})$
///   with respect to $x_{k}$ at any point $\mathbf{x}\in\mathbb{R}^{n}$.
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $f(\mathbf{x})$ to evaluate its
/// partial derivative with respect to $x_{k}$.
///
/// # Examples
///
/// ## Basic Example
///
/// Compute the partial derivative of
///
/// $$f(x)=x^{3}\sin{y}$$
///
/// with respect to $y$ at $(x,y)=(5,1)$, and compare the result to the true result of
///
/// $$\frac{\partial f}{\partial y}\bigg\rvert_{(x,y)=(5,1)}=5^{3}\cos{(1)}$$
///
/// First, note that we can rewrite this function as
///
/// $$f(\mathbf{x})=x_{0}^{3}\sin{x_{1}}$$
///
/// where $\mathbf{x}=(x_{0},x_{1})^{T}$ (note that we use 0-based indexing to aid with the
/// computational implementation). We are then trying to find
///
/// $$\frac{\partial f}{\partial x_{1}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}$$
///
/// where $\mathbf{x}_{0}=(5,1)^{T}$.
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
///
/// use numdiff::{get_spartial_derivative, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
///     x.vget(0).powi(3) * x.vget(1).sin()
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 1.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 1;
///
/// // Autogenerate the function "dfk" that can be used to compute the partial derivative of f(x)
/// // with respect to xₖ at any point x.
/// get_spartial_derivative!(f, dfk);
///
/// // Verify that the partial derivative function obtained using get_spartial_derivative! computes
/// // the partial derivative correctly.
/// assert_eq!(dfk(&x0, k, &[]), 5.0_f64.powi(3) * 1.0_f64.cos());
/// ```
///
/// #### Using other vector types
///
/// The function produced by `get_spartial_derivative!` can accept _any_ type for `x0`, as long as
/// it implements the `linalg_traits::Vector` trait.
///
/// ```
/// use faer::Mat;
/// use linalg_traits::{Scalar, Vector};
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
///
/// use numdiff::{get_spartial_derivative, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
///     x.vget(0).powi(3) * x.vget(1).sin()
/// }
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 1;
///
/// // Autogenerate the function "dfk" that can be used to compute the partial derivative of f(x)
/// // with respect to xₖ at any point x.
/// get_spartial_derivative!(f, dfk);
///
/// // nalgebra::DVector
/// let x0: DVector<f64> = dvector![5.0, 1.0];
/// let dfk_eval: f64 = dfk(&x0, k, &[]);
///
/// // nalgebra::SVector
/// let x0: SVector<f64, 2> = SVector::from_slice(&[5.0, 1.0]);
/// let dfk_eval: f64 = dfk(&x0, k, &[]);
///
/// // ndarray::Array1
/// let x0: Array1<f64> = array![5.0, 1.0];
/// let dfk_eval: f64 = dfk(&x0, k, &[]);
///
/// // faer::Mat
/// let x0: Mat<f64> = Mat::from_slice(&[5.0, 1.0]);
/// let dfk_eval: f64 = dfk(&x0, k, &[]);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Compute the partial derivative of a parameterized function
///
/// $$f(\mathbf{x})=ax_{0}^{2}+bx_{1}^{2}+cx_{0}x_{1}+d\sin(ex_{0})$$
///
/// where $a$, $b$, $c$, $d$, and $e$ are runtime parameters. The partial derivatives are:
///
/// * $\dfrac{\partial f}{\partial x_{0}}=2ax_{0}+cx_{1}+de\cos(ex_{0})$
/// * $\dfrac{\partial f}{\partial x_{1}}=2bx_{1}+cx_{0}$
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_spartial_derivative, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> S {
///     let a = S::new(p[0]);
///     let b = S::new(p[1]);
///     let c = S::new(p[2]);
///     let d = S::new(p[3]);
///     let e = S::new(p[4]);
///     a * x.vget(0).powi(2) + b * x.vget(1).powi(2) + c * x.vget(0) * x.vget(1) + d * (e * x.vget(0)).sin()
/// }
///
/// // Define individual parameters.
/// let a = 1.5;
/// let b = 2.0;
/// let c = 0.8;
/// let d = 3.0;
/// let e = 0.5;
///
/// // Parameter vector.
/// let p = [a, b, c, d, e];
///
/// // Evaluation point.
/// let x0 = vec![1.0, -0.5];
///
/// // Autogenerate the partial derivative function.
/// get_spartial_derivative!(f, dfk);
///
/// // True partial derivative functions.
/// let df_dx0_true = |x: &[f64]| 2.0 * a * x[0] + c * x[1] + d * e * (e * x[0]).cos();
/// let df_dx1_true = |x: &[f64]| 2.0 * b * x[1] + c * x[0];
///
/// // Compute ∂f/∂x₀ at x₀ and compare with true function.
/// let df_dx0: f64 = dfk(&x0, 0, &p);
/// let expected_df_dx0 = df_dx0_true(&x0);
/// assert_equal_to_decimal!(df_dx0, expected_df_dx0, 14);
///
/// // Compute ∂f/∂x₁ at x0 and compare with true function.
/// let df_dx1: f64 = dfk(&x0, 1, &p);
/// let expected_df_dx1 = df_dx1_true(&x0);
/// assert_equal_to_decimal!(df_dx1, expected_df_dx1, 15);
/// ```
#[macro_export]
macro_rules! get_spartial_derivative {
    ($f:ident, $func_name:ident) => {
        /// Partial derivative of a multivariate, scalar-valued function `f: ℝⁿ → ℝ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_spartial_derivative!` macro.
        ///
        /// # Arguments
        ///
        /// `x0` - Evaluation point, `x₀ ∈ ℝⁿ`.
        /// `k` - Element of `x` to differentiate with respect to. Note that this uses 0-based
        ///       indexing (e.g. `x = (x₀,...,xₖ,...,xₙ₋₁)ᵀ).
        /// `p` - Parameter vector. This is a vector of additional runtime parameters that the
        ///       function may depend on but is not differentiated with respect to.
        ///
        /// # Returns
        ///
        /// Partial derivative of `f` with respect to `xₖ`, evaluated at `x = x₀`.
        ///
        /// `(∂f/∂xₖ)|ₓ₌ₓ₀ ∈ ℝ`
        fn $func_name<S, V>(x0: &V, k: usize, p: &[f64]) -> f64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Promote the evaluation point to a vector of dual numbers.
            let mut x0_dual = x0.clone().to_dual_vector();

            // Take a unit step forward in the kth dual direction.
            x0_dual.vset(k, Dual::new(x0_dual.vget(k).get_real(), 1.0));

            // Evaluate the function at the dual number.
            let f_x0 = $f(&x0_dual, p);

            // Partial derivative of f with respect to xₖ.
            f_x0.get_dual()
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{Dual, DualVector};
    use linalg_traits::{Scalar, Vector};
    use nalgebra::SVector;

    #[test]
    fn test_spartial_derivative_1() {
        // Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(2)
        }

        // Evaluation point.
        let x0 = vec![2.0];

        // Element to differentiate with respect to.
        let k = 0;

        // True partial derivative function.
        let dfk = |x: &Vec<f64>| 2.0 * x[0];

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_spartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: f64 = dfk_autodiff(&x0, k, &[]);
        let dfk_eval: f64 = dfk(&x0);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }

    #[test]
    fn test_spartial_derivative_2() {
        // Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(3) * x.vget(1).powi(3)
        }

        // Evaluation point.
        let x0: SVector<f64, 2> = SVector::from_slice(&[3.0, 2.0]);

        // Element to differentiate with respect to.
        let k = 1;

        // True partial derivative function.
        let dfk = |x: &SVector<f64, 2>| 3.0 * x[0].powi(3) * x[1].powi(2);

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_spartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: f64 = dfk_autodiff(&x0, k, &[]);
        let dfk_eval: f64 = dfk(&x0);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }

    #[test]
    fn test_spartial_derivative_3() {
        /// Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> S {
            let a = S::new(p[0]);
            let b = S::new(p[1]);
            let c = S::new(p[2]);
            a * (b * x.vget(0)).exp() + c * x.vget(1).powi(3)
        }

        // Parameter vector.
        let p = [2.5, 0.3, -1.2];

        // Evaluation point.
        let x0: SVector<f64, 2> = SVector::from_slice(&[1.5, 2.0]);

        // Function to take the partial derivative of.
        let k = 0;

        // True partial derivative function.
        let dfk = |x: &SVector<f64, 2>, p: &[f64]| p[0] * p[1] * (p[1] * x[0]).exp();

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_spartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: f64 = dfk_autodiff(&x0, k, &p);
        let dfk_eval: f64 = dfk(&x0, &p);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }
}
