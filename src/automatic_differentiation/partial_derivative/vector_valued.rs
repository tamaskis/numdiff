/// Get a function that returns the partial derivative of the provided multivariate, vector-valued
/// function.
///
/// The partial derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function, $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$.
/// * `func_name` - Name of the function that will return the partial derivative of
///   $\mathbf{f}(\mathbf{x})$ with respect to $x_{k}$ at any point $\mathbf{x}\in\mathbb{R}^{n}$.
///
/// # Defining `f`
///
/// The multivariate, vector-valued function `f` must have the following function signature:
///
/// ```ignore
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
///     // place function contents here
/// }
/// ```
///
/// For the automatic differentiation to work, `f` must be fully generic over the types of scalars
/// and vectors used. Additionally, the function must return an instance of `V::DVector` (a
/// dynamically-sized vector type that is compatible with `V`, see the
/// [`linalg-traits` docs](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Vector.html#associatedtype.DVectorT)
/// for more information) since we did not want to burden the user with having to specify the size
/// of the output vector (i.e. $m$, where $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$) at compile
/// time, especially since users may be using this crate exclusively with dynamically-sized types.
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $\mathbf{f}(\mathbf{x})$ to
/// evaluate its partial derivative with respect to $x_{k}$.
///
/// # Example
///
/// Compute the partial derivative of
///
/// $$\mathbf{f}(\mathbf{x})=\begin{bmatrix}\sin{x_{0}}\sin{x_{1}}\\\\\cos{x_{0}}\cos{x_{1}}\end{bmatrix}$$
///
/// with respect to $x_{0}$ at $\mathbf{x}=(1,2)^{T}$, and compare the result to the
/// true result of
///
/// $$
/// \frac{\partial\mathbf{f}}{\partial x_{0}}\bigg\rvert_{\mathbf{x}=(1,2)^{T}}=
/// \begin{bmatrix}
/// \cos{(1)}\sin{(2)} \\\\
/// -\sin{(1)}\cos{(2)}
/// \end{bmatrix}
/// $$
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
///
/// use numdiff::{get_vpartial_derivative, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
///     V::DVectorT::from_slice(&[
///         x.vget(0).sin() * x.vget(1).sin(),
///         x.vget(0).cos() * x.vget(1).cos(),
///     ])
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![1.0, 2.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 0;
///
/// // Autogenerate the function "dfk" that can be used to compute the partial derivative of f(x)
/// // with respect to xₖ at any point x.
/// get_vpartial_derivative!(f, dfk);
///
/// // Verify that the partial derivative function obtained using get_vpartial_derivative! computes
/// // the partial derivative correctly.
/// assert_eq!(
///     dfk(&x0, k),
///     vec![1.0_f64.cos() * 2.0_f64.sin(), -1.0_f64.sin() * 2.0_f64.cos()]
/// );
/// ```
///
/// #### Using other vector types
///
/// The function produced by `get_vpartial_derivative!` can accept _any_ type for `x0`, as long as
/// it implements the `linalg_traits::Vector` trait.
///
/// ```
/// use faer::Mat;
/// use linalg_traits::{Scalar, Vector};
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
///
/// use numdiff::{get_vpartial_derivative, Dual, DualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
///     V::DVectorT::from_slice(&[
///         x.vget(0).sin() * x.vget(1).sin(),
///         x.vget(0).cos() * x.vget(1).cos(),
///     ])
/// }
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 0;
///
/// // Autogenerate the function "dfk" that can be used to compute the partial derivative of f(x)
/// // with respect to xₖ at any point x.
/// get_vpartial_derivative!(f, dfk);
///
/// // nalgebra::DVector
/// let x0: DVector<f64> = dvector![5.0, 1.0];
/// let dfk_eval: DVector<f64> = dfk(&x0, k);
///
/// // nalgebra::SVector
/// let x0: SVector<f64, 2> = SVector::from_slice(&[5.0, 1.0]);
/// let dfk_eval: DVector<f64> = dfk(&x0, k);
///
/// // ndarray::Array1
/// let x0: Array1<f64> = array![5.0, 1.0];
/// let dfk_eval: Array1<f64> = dfk(&x0, k);
///
/// // faer::Mat
/// let x0: Mat<f64> = Mat::from_slice(&[5.0, 1.0]);
/// let dfk_eval: Mat<f64> = dfk(&x0, k);
/// ```
#[macro_export]
macro_rules! get_vpartial_derivative {
    ($f:ident, $func_name:ident) => {
        /// Partial derivative of a multivariate, vector-valued function `f: ℝⁿ → ℝᵐ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_vpartial_derivative!` macro.
        ///
        /// # Arguments
        ///
        /// `x0` - Evaluation point, `x₀ ∈ ℝⁿ`.
        /// `k` - Element of `x` to differentiate with respect to. Note that this uses 0-based
        ///       indexing (e.g. `x = (x₀,...,xₖ,...,xₙ₋₁)ᵀ).
        ///
        /// # Returns
        ///
        /// Partial derivative of `f` with respect to `xₖ`, evaluated at `x = x₀`.
        ///
        /// `(∂f/∂xₖ)|ₓ₌ₓ₀ ∈ ℝᵐ`
        fn $func_name<S, V>(x0: &V, k: usize) -> V::DVectorf64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Promote the evaluation point to a vector of dual numbers.
            let mut x0_dual = x0.clone().to_dual_vector();

            // Take a unit step forward in the kth dual direction.
            x0_dual.vset(k, Dual::new(x0_dual.vget(k).get_real(), 1.0));

            // Evaluate the function at the dual number.
            let f_x0 = $f(&x0_dual);

            // Partial derivative of f with respect to xₖ.
            let mut dfk_x0 = V::DVectorf64::new_with_length(f_x0.len());
            for i in 0..dfk_x0.len() {
                dfk_x0.vset(i, f_x0.vget(i).get_dual());
            }
            dfk_x0
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{Dual, DualVector};
    use linalg_traits::{Scalar, Vector};
    use nalgebra::{DVector, SVector, dvector};
    use ndarray::{Array1, array};

    #[test]
    fn test_vpartial_derivative_1() {
        // Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
            V::DVectorT::from_slice(&[x.vget(0).powi(2)])
        }

        // Evaluation point.
        let x0 = vec![2.0];

        // Element to differentiate with respect to.
        let k = 0;

        // True partial derivative function.
        let dfk = |x: &Vec<f64>| vec![2.0 * x[0]];

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_vpartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: Vec<f64> = dfk_autodiff(&x0, k);
        let dfk_eval: Vec<f64> = dfk(&x0);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }

    #[test]
    fn test_vpartial_derivative_2() {
        // Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
            V::DVectorT::from_slice(&[x.vget(0).powi(4), x.vget(1).powi(3)])
        }

        // Evaluation point.
        let x0: DVector<f64> = dvector![3.0, 2.0];

        // Element to differentiate with respect to.
        let k = 1;

        // True partial derivative function.
        let dfk = |x: &DVector<f64>| dvector![0.0, 3.0 * x[1].powi(2)];

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_vpartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: DVector<f64> = dfk_autodiff(&x0, k);
        let dfk_eval: DVector<f64> = dfk(&x0);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }

    #[test]
    fn test_vpartial_derivative_3() {
        // Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
            V::DVectorT::from_slice(&[x.vget(0).powi(3) * x.vget(1).powi(3)])
        }

        // Evaluation point.
        let x0 = vec![3.0, 2.0];

        // Element to differentiate with respect to.
        let k = 1;

        // True partial derivative function.
        let dfk = |x: &Vec<f64>| vec![3.0 * x[0].powi(3) * x[1].powi(2)];

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_vpartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: Vec<f64> = dfk_autodiff(&x0, k);
        let dfk_eval: Vec<f64> = dfk(&x0);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }

    #[test]
    fn test_vpartial_derivative_4() {
        // Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
            V::DVectorT::from_slice(&[x.vget(0).powi(4), x.vget(1).powi(3)])
        }

        // Evaluation point.
        let x0 = array![1.0, 2.0];

        // Element to differentiate with respect to.
        let k = 1;

        // True partial derivative function.
        let dfk = |x: &Array1<f64>| array![0.0, 3.0 * x[1].powi(2)];

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_vpartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: Array1<f64> = dfk_autodiff(&x0, k);
        let dfk_eval: Array1<f64> = dfk(&x0);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }

    #[test]
    fn test_vpartial_derivative_5() {
        // Function to take the partial derivative of.
        fn f<S: Scalar, V: Vector<S>>(x: &V) -> V::DVectorT<S> {
            V::DVectorT::from_slice(&[
                x.vget(0),
                x.vget(2) * 5.0,
                x.vget(1).powi(2) * 4.0 - x.vget(2) * 2.0,
                x.vget(2) * x.vget(0).sin(),
            ])
        }

        // Evaluation point.
        let x0: SVector<f64, 3> = SVector::from_row_slice(&[5.0, 6.0, 7.0]);

        // Element to differentiate with respect to.
        let k = 2;

        // True partial derivative function.
        let dfk =
            |x: &SVector<f64, 3>| SVector::<f64, 4>::from_row_slice(&[0.0, 5.0, -2.0, x[0].sin()]);

        // Partial derivative function obtained via forward-mode automatic differentiation.
        get_vpartial_derivative!(f, dfk_autodiff);

        // Evaluate the partial derivative using both functions.
        let dfk_eval_autodiff: DVector<f64> = dfk_autodiff(&x0, k);
        let dfk_eval: SVector<f64, 4> = dfk(&x0);

        // Test autodiff partial derivative against true partial derivative.
        assert_eq!(dfk_eval_autodiff, dfk_eval);
    }
}
