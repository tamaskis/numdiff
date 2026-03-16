/// Get a function that returns the second-order partial derivative of the provided multivariate,
/// vector-valued function.
///
/// The second-order partial derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function, $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$.
/// * `func_name` - Name of the function that will return the second-order partial derivative of
///   $\mathbf{f}(\mathbf{x})$ with respect to $x_{k}$ at any point $\mathbf{x}\in\mathbb{R}^{n}$.
/// * `param_type` (optional) - Type of the extra runtime parameter `p` that is passed to `f`.
///   Defaults to `[f64]` (implying that `f` accepts `p: &[f64]`).
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $\mathbf{f}(\mathbf{x})$ to
/// evaluate its second-order partial derivative with respect to $x_{k}$.
///
/// # Examples
///
/// ## Basic Example
///
/// Compute the second-order partial derivative of
///
/// $$\mathbf{f}(x,y) = \begin{bmatrix} x^{4}+2x^{2}y \\ y^{3}+xy^{2} \end{bmatrix}$$
///
/// with respect to $x$ at $(x,y)=(2,1)$.
///
/// First, note that we can rewrite this function as
///
/// $$\mathbf{f}(\mathbf{x})=\begin{bmatrix}x_{0}^{4}+2x_{0}^{2}x_{1}\\x_{1}^{3}+x_{0}x_{1}^{2}\end{bmatrix}$$
///
/// where $\mathbf{x}=(x_{0},x_{1})^{T}$ (note that we use 0-based indexing to aid with the
/// computational implementation). We are then trying to find
///
/// $$\frac{\partial^{2} \mathbf{f}}{\partial x_{0}^{2}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}$$
///
/// where $\mathbf{x}_{0}=(2,1)^{T}$.
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
///
/// use numdiff::{get_vpartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
///     let f0 = x.vget(0).powi(4) + S::new(2.0) * x.vget(0).powi(2) * x.vget(1);
///     let f1 = x.vget(1).powi(3) + x.vget(0) * x.vget(1).powi(2);
///     vec![f0, f1]
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![2.0, 1.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 0;
///
/// // Autogenerate the function "d2fkk" that can be used to compute the second-order partial
/// // derivative of f(x) with respect to xₖ at any point x.
/// get_vpartial_derivative2!(f, d2fkk);
///
/// // Evaluate the second-order partial derivative.
/// let d2f_dx0dx0 = d2fkk(&x0, k, &[]);
///
/// // True results:
/// // ∂²f₀/∂x₀² = 12x₀² + 4x₁ = 12(2)² + 4(1) = 52
/// // ∂²f₁/∂x₀² = 0
/// assert_eq!(d2f_dx0dx0[0], 52.0);
/// assert_eq!(d2f_dx0dx0[1], 0.0);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Compute the second-order partial derivative of a parameterized function
///
/// $$\mathbf{f}(\mathbf{x})=\begin{bmatrix}a x_{0}^{4}+b x_{0}^{2}x_{1}\\c x_{1}^{3}+d\sin(e x_{0})\end{bmatrix}$$
///
/// where $a$, $b$, $c$, $d$, and $e$ are runtime parameters.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_vpartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> Vec<S> {
///     let a = S::new(p[0]);
///     let b = S::new(p[1]);
///     let c = S::new(p[2]);
///     let d = S::new(p[3]);
///     let e = S::new(p[4]);
///     let f0 = a * x.vget(0).powi(4) + b * x.vget(0).powi(2) * x.vget(1);
///     let f1 = c * x.vget(1).powi(3) + d * (e * x.vget(0)).sin();
///     vec![f0, f1]
/// }
///
/// // Runtime parameters.
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
/// // Autogenerate the second-order partial derivative function.
/// get_vpartial_derivative2!(f, d2fkk);
///
/// // True second-order partial derivative functions.
/// let d2f0_dx0dx0_true = |x: &[f64]| 12.0 * a * x[0].powi(2) + 2.0 * b * x[1];
/// let d2f0_dx1dx1_true = |_x: &[f64]| 0.0;
/// let d2f1_dx0dx0_true = |x: &[f64]| -d * e * e * (e * x[0]).sin();
/// let d2f1_dx1dx1_true = |x: &[f64]| 6.0 * c * x[1];
///
/// // Compute ∂²f/∂x₀² at x₀ and compare with true function.
/// let d2f_dx0dx0 = d2fkk(&x0, 0, &p);
/// let expected_d2f0_dx0dx0 = d2f0_dx0dx0_true(&x0);
/// let expected_d2f1_dx0dx0 = d2f1_dx0dx0_true(&x0);
/// assert_equal_to_decimal!(d2f_dx0dx0[0], expected_d2f0_dx0dx0, 14);
/// assert_equal_to_decimal!(d2f_dx0dx0[1], expected_d2f1_dx0dx0, 14);
///
/// // Compute ∂²f/∂x₁² at x0 and compare with true function.
/// let d2f_dx1dx1 = d2fkk(&x0, 1, &p);
/// let expected_d2f0_dx1dx1 = d2f0_dx1dx1_true(&x0);
/// let expected_d2f1_dx1dx1 = d2f1_dx1dx1_true(&x0);
/// assert_equal_to_decimal!(d2f_dx1dx1[0], expected_d2f0_dx1dx1, 15);
/// assert_equal_to_decimal!(d2f_dx1dx1[1], expected_d2f1_dx1dx1, 15);
/// ```
///
/// ## Example Passing Custom Parameter Types
///
/// Use a custom parameter struct instead of `f64` values.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_vpartial_derivative2, HyperDual, HyperDualVector};
///
/// struct Data {
///     a: f64,
///     b: f64,
///     c: f64,
///     d: f64,
///     e: f64,
/// }
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> Vec<S> {
///     let a = S::new(p.a);
///     let b = S::new(p.b);
///     let c = S::new(p.c);
///     let d = S::new(p.d);
///     let e = S::new(p.e);
///     let f0 = a * x.vget(0).powi(4) + b * x.vget(0).powi(2) * x.vget(1);
///     let f1 = c * x.vget(1).powi(3) + d * (e * x.vget(0)).sin();
///     vec![f0, f1]
/// }
///
/// // Runtime parameter struct.
/// let p = Data {
///     a: 1.5,
///     b: 2.0,
///     c: 0.8,
///     d: 3.0,
///     e: 0.5,
/// };
///
/// // Evaluation point.
/// let x0 = vec![1.0, -0.5];
///
/// // Autogenerate the second-order partial derivative function, telling the macro to expect a
/// // runtime parameter of type &Data.
/// get_vpartial_derivative2!(f, d2fkk, Data);
///
/// // True second-order partial derivative functions.
/// let d2f0_dx0dx0_true = |x: &[f64]| 12.0 * p.a * x[0].powi(2) + 2.0 * p.b * x[1];
/// let d2f0_dx1dx1_true = |_x: &[f64]| 0.0;
/// let d2f1_dx0dx0_true = |x: &[f64]| -p.d * p.e * p.e * (p.e * x[0]).sin();
/// let d2f1_dx1dx1_true = |x: &[f64]| 6.0 * p.c * x[1];
///
/// // Compute the second-order partial derivatives using both the automatically generated
/// // second-order partial derivative function and the true second-order partial derivative
/// // functions, and compare the results.
/// let d2f_dx0dx0 = d2fkk(&x0, 0, &p);
/// let d2f_dx1dx1 = d2fkk(&x0, 1, &p);
/// assert_equal_to_decimal!(d2f_dx0dx0[0], d2f0_dx0dx0_true(&x0), 14);
/// assert_equal_to_decimal!(d2f_dx0dx0[1], d2f1_dx0dx0_true(&x0), 14);
/// assert_equal_to_decimal!(d2f_dx1dx1[0], d2f0_dx1dx1_true(&x0), 15);
/// assert_equal_to_decimal!(d2f_dx1dx1[1], d2f1_dx1dx1_true(&x0), 15);
/// ```
#[macro_export]
macro_rules! get_vpartial_derivative2 {
    ($f:ident, $func_name:ident) => {
        get_vpartial_derivative2!($f, $func_name, [f64]);
    };
    ($f:ident, $func_name:ident, $param_type:ty) => {
        /// Second-order partial derivative of a multivariate, vector-valued function `f: ℝⁿ → ℝᵐ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_vpartial_derivative2!` macro.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, `x₀ ∈ ℝⁿ`.
        /// * `k` - Element of `x` to differentiate with respect to twice. Note that this uses
        ///   0-based indexing (e.g. `x = (x₀,...,xₖ,...,xₙ₋₁)ᵀ`).
        /// * `p` - Extra runtime parameter. This is a parameter (can be of any arbitrary type)
        ///   defined at runtime that the function may depend on but is not differentiated with
        ///   respect to.
        ///
        /// # Returns
        ///
        /// Second-order partial derivative of `f` with respect to `xₖ`, evaluated at `x = x₀`.
        ///
        /// `(∂²f/∂xₖ²)|ₓ₌ₓ₀ ∈ ℝᵐ`
        fn $func_name<S, V>(x0: &V, k: usize, p: &$param_type) -> Vec<f64>
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Promote the evaluation point to a vector of hyper-dual numbers.
            let mut x0_hyperdual = x0.clone().to_hyper_dual_vector();

            // Take a unit step forward in both hyper-dual directions for the kth component.
            x0_hyperdual.vset(
                k,
                HyperDual::new(x0_hyperdual.vget(k).get_a(), 1.0, 1.0, 0.0),
            );

            // Evaluate the function at the hyper-dual number.
            let f_x0 = $f(&x0_hyperdual, p);

            // Extract the second-order partial derivatives.
            f_x0.iter().map(|component| component.get_d()).collect()
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{HyperDual, HyperDualVector};
    use linalg_traits::{Scalar, Vector};
    use nalgebra::SVector;
    use numtest::*;
    use std::f64::consts::PI;

    #[cfg(feature = "trig")]
    use trig::Trig;

    #[test]
    fn test_vpartial_derivative2_basic() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = [x₀⁴ + 2x₀²x₁, x₁³ + x₀x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            let f0 = x.vget(0).powi(4) + S::new(2.0) * x.vget(0).powi(2) * x.vget(1);
            let f1 = x.vget(1).powi(3) + x.vget(0) * x.vget(1).powi(2);
            vec![f0, f1]
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0 = vec![2.0, 1.0];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(x₀⁴ + 2x₀²x₁) = 12x₀² + 4x₁ = 12(2)² + 4(1) = 52
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀²(x₁³ + x₀x₁²) = 0
        let d2f_dx0dx0 = d2fkk(&x0, 0, &[]);
        assert_equal_to_decimal!(d2f_dx0dx0[0], 52.0, 15);
        assert_equal_to_decimal!(d2f_dx0dx0[1], 0.0, 15);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁²(x₀⁴ + 2x₀²x₁) = 0
        //  --> ∂²f₁/∂x₁² = ∂²/∂x₁²(x₁³ + x₀x₁²) = 6x₁ + 2x₀ = 6(1) + 2(2) = 10
        let d2f_dx1dx1 = d2fkk(&x0, 1, &[]);
        assert_equal_to_decimal!(d2f_dx1dx1[0], 0.0, 15);
        assert_equal_to_decimal!(d2f_dx1dx1[1], 10.0, 15);
    }

    #[test]
    fn test_vpartial_derivative2_polynomial() {
        // Function to test various polynomial orders:
        // f(x₀) = [x₀², x₀³, x₀⁴]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x.vget(0).powi(2), x.vget(0).powi(3), x.vget(0).powi(4)]
        }

        // Define the evaluation point x₀ = 2.0.
        let x0 = vec![2.0];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(x₀²) = 2
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀²(x₀³) = 6x₀ = 6(2) = 12
        //  --> ∂²f₂/∂x₀² = ∂²/∂x₀²(x₀⁴) = 12x₀² = 12(2)² = 48
        let result = d2fkk(&x0, 0, &[]);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 12.0);
        assert_eq!(result[2], 48.0);
    }

    #[test]
    fn test_vpartial_derivative2_multivariate() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁, x₂) = [x₀³, x₁⁴, x₂²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x.vget(0).powi(3), x.vget(1).powi(4), x.vget(2).powi(2)]
        }

        // Define the evaluation point (x₀, x₁, x₂) = (1.0, 2.0, 3.0).
        let x0 = vec![1.0, 2.0, 3.0];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(x₀³) = 6x₀ = 6(1) = 6
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀²(x₁⁴) = 0
        //  --> ∂²f₂/∂x₀² = ∂²/∂x₀²(x₂²) = 0
        let result_x0 = d2fkk(&x0, 0, &[]);
        assert_eq!(result_x0[0], 6.0);
        assert_eq!(result_x0[1], 0.0);
        assert_eq!(result_x0[2], 0.0);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁²(x₀³) = 0
        //  --> ∂²f₁/∂x₁² = ∂²/∂x₁²(x₁⁴) = 12x₁² = 12(2)² = 48
        //  --> ∂²f₂/∂x₁² = ∂²/∂x₁²(x₂²) = 0
        let result_x1 = d2fkk(&x0, 1, &[]);
        assert_eq!(result_x1[0], 0.0);
        assert_eq!(result_x1[1], 48.0);
        assert_eq!(result_x1[2], 0.0);

        // Check ∂²f/∂x₂².
        //  --> ∂²f₀/∂x₂² = ∂²/∂x₂²(x₀³) = 0
        //  --> ∂²f₁/∂x₂² = ∂²/∂x₂²(x₁⁴) = 0
        //  --> ∂²f₂/∂x₂² = ∂²/∂x₂²(x₂²) = 2
        let result_x2 = d2fkk(&x0, 2, &[]);
        assert_eq!(result_x2[0], 0.0);
        assert_eq!(result_x2[1], 0.0);
        assert_eq!(result_x2[2], 2.0);
    }

    #[test]
    fn test_vpartial_derivative2_trig() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = [sin(x₀), cos(x₁)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x.vget(0).sin(), x.vget(1).cos()]
        }

        // Define the evaluation point (x₀, x₁) = (π/2, π/4).
        let x0 = vec![PI / 2.0, PI / 4.0];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(sin(x₀)) = -sin(x₀) = -sin(π/2) = -1
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀²(cos(x₁)) = 0
        let result_x0 = d2fkk(&x0, 0, &[]);
        assert_equal_to_decimal!(result_x0[0], -1.0, 15);
        assert_eq!(result_x0[1], 0.0);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁²(sin(x₀)) = 0
        //  --> ∂²f₁/∂x₁² = ∂²/∂x₁²(cos(x₁)) = -cos(x₁) = -cos(π/4) = -√(2)/2
        let expected = -(2.0_f64.sqrt() / 2.0);
        let result_x1 = d2fkk(&x0, 1, &[]);
        assert_eq!(result_x1[0], 0.0);
        assert_equal_to_decimal!(result_x1[1], expected, 15);
    }

    #[test]
    fn test_vpartial_derivative2_exponential() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = [exp(x₀), x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x.vget(0).exp(), x.vget(1).powi(2)]
        }

        // Define the evaluation point (x₀, x₁) = (1.0, 2.0).
        let x0 = vec![1.0, 2.0];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(exp(x₀)) = exp(x₀) = exp(1) = e
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀²(x₁²) = 0
        let result_x0 = d2fkk(&x0, 0, &[]);
        assert_equal_to_decimal!(result_x0[0], std::f64::consts::E, 15);
        assert_eq!(result_x0[1], 0.0);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁²(exp(x₀)) = 0
        //  --> ∂²f₁/∂x₁² = ∂²/∂x₁²(x₁²) = 2
        let result_x1 = d2fkk(&x0, 1, &[]);
        assert_eq!(result_x1[0], 0.0);
        assert_eq!(result_x1[1], 2.0);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_vpartial_derivative2_with_runtime_parameters() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = [ax₀⁴ + bx₀²x₁, cx₁³ + d sin(ex₀)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> Vec<S> {
            let a = S::new(p[0]);
            let b = S::new(p[1]);
            let c = S::new(p[2]);
            let d = S::new(p[3]);
            let e = S::new(p[4]);
            let f0 = a * x.vget(0).powi(4) + b * x.vget(0).powi(2) * x.vget(1);
            let f1 = c * x.vget(1).powi(3) + d * (e * x.vget(0)).sin();
            vec![f0, f1]
        }

        // Runtime parameters.
        let a = 1.5;
        let b = 2.0;
        let c = 0.8;
        let d = 3.0;
        let e = 0.5;
        let p = [a, b, c, d, e];

        // Define the evaluation point (x₀, x₁) = (1.0, -0.5).
        let x0 = vec![1.0, -0.5];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀² (ax₀⁴ + bx₀²x₁) = 12ax₀² + 2bx₁ = 12(1.5)(1)² + 2(2.0)(-0.5) = 16
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀² (cx₁³ + d sin(ex₀)) = -de² sin(ex₀) = -3(0.5)²sin(0.5(1))
        let d2f_dx0dx0 = d2fkk(&x0, 0, &p);
        let d2f0_dx0dx0_expected = 12.0_f64 * a * (x0[0] as f64).powi(2_i32) + 2.0_f64 * b * x0[1];
        let d2f1_dx0dx0_expected = -d * e * e * (e * x0[0]).sin();
        assert_equal_to_decimal!(d2f_dx0dx0[0], d2f0_dx0dx0_expected, 14);
        assert_equal_to_decimal!(d2f_dx0dx0[1], d2f1_dx0dx0_expected, 14);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁² (ax₀⁴ + bx₀²x₁) = 0
        //  --> ∂²f₁/∂x₁² = ∂²/∂x₁² (cx₁³ + d sin(ex₀)) = 6cx₁ = 6(0.8)(-0.5) = -2.4
        let d2f_dx1dx1 = d2fkk(&x0, 1, &p);
        let d2f0_dx1dx1_expected = 0.0_f64;
        let d2f1_dx1dx1_expected = 6.0_f64 * c * x0[1];
        assert_equal_to_decimal!(d2f_dx1dx1[0], d2f0_dx1dx1_expected, 15);
        assert_equal_to_decimal!(d2f_dx1dx1[1], d2f1_dx1dx1_expected, 15);
    }

    #[test]
    fn test_vpartial_derivative2_custom_params() {
        struct Data {
            a: f64,
            b: f64,
            c: f64,
            d: f64,
            e: f64,
        }

        // Function to take the second-order partial derivative of.
        #[allow(clippy::many_single_char_names)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> Vec<S> {
            let a = S::new(p.a);
            let b = S::new(p.b);
            let c = S::new(p.c);
            let d = S::new(p.d);
            let e = S::new(p.e);
            let f0 = a * x.vget(0).powi(4) + b * x.vget(0).powi(2) * x.vget(1);
            let f1 = c * x.vget(1).powi(3) + d * (e * x.vget(0)).sin();
            vec![f0, f1]
        }

        // Runtime parameter struct.
        let p = Data {
            a: 1.5,
            b: 2.0,
            c: 0.8,
            d: 3.0,
            e: 0.5,
        };

        // Define the evaluation point (x₀, x₁) = (1.0, -0.5).
        let x0 = vec![1.0, -0.5];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk, Data);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(ax₀⁴ + bx₀²x₁) = 12ax₀² + 2bx₁ = 12(1.5)(1)² + 2(2.0)(-0.5) = 16
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀²(cx₁³ + d sin(ex₀)) = -de² sin(ex₀) = -3(0.5)²sin(0.5(1))
        let d2f_dx0dx0 = d2fkk(&x0, 0, &p);
        let d2f0_dx0dx0_expected =
            12.0_f64 * p.a * (x0[0] as f64).powi(2_i32) + 2.0_f64 * p.b * x0[1];
        let d2f1_dx0dx0_expected = -p.d * p.e * p.e * (p.e * x0[0]).sin();
        assert_equal_to_decimal!(d2f_dx0dx0[0], d2f0_dx0dx0_expected, 14);
        assert_equal_to_decimal!(d2f_dx0dx0[1], d2f1_dx0dx0_expected, 14);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁²(ax₀⁴ + bx₀²x₁) = 0
        //  --> ∂²f₁/∂x₁² = ∂²/∂x₁²(cx₁³ + d sin(ex₀)) = 6cx₁ = 6(0.8)(-0.5) = -2.4
        let d2f_dx1dx1 = d2fkk(&x0, 1, &p);
        let d2f0_dx1dx1_expected = 0.0_f64;
        let d2f1_dx1dx1_expected = 6.0_f64 * p.c * x0[1];
        assert_equal_to_decimal!(d2f_dx1dx1[0], d2f0_dx1dx1_expected, 15);
        assert_equal_to_decimal!(d2f_dx1dx1[1], d2f1_dx1dx1_expected, 15);
    }

    #[test]
    fn test_vpartial_derivative2_vector_types() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = [x₀³, x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x.vget(0).powi(3), x.vget(1).powi(2)]
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 3.0).
        let x_nalgebra: SVector<f64, 2> = SVector::from([2.0, 3.0]);

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(x₀³) = 6x₀ = 6(2) = 12
        //  --> ∂²f₁/∂x₀² = ∂²/∂x₀²(x₁²) = 0
        let result_x0 = d2fkk(&x_nalgebra, 0, &[]);
        assert_eq!(result_x0[0], 12.0);
        assert_eq!(result_x0[1], 0.0);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁²(x₀³) = 0
        //  --> ∂²f₁/∂x₁² = ∂²/∂x₁²(x₁²) = 2
        let result_x1 = d2fkk(&x_nalgebra, 1, &[]);
        assert_eq!(result_x1[0], 0.0);
        assert_eq!(result_x1[1], 2.0);
    }

    #[test]
    fn test_vpartial_derivative2_single_component() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = [x₀⁴ + x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x.vget(0).powi(4) + x.vget(1).powi(2)]
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0 = vec![2.0, 1.0];

        // Generate second-order partial derivative function.
        get_vpartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        //  --> ∂²f₀/∂x₀² = ∂²/∂x₀²(x₀⁴ + x₁²) = 12x₀² = 12(2)² = 48
        let result_x0 = d2fkk(&x0, 0, &[]);
        assert_eq!(result_x0[0], 48.0);

        // Check ∂²f/∂x₁².
        //  --> ∂²f₀/∂x₁² = ∂²/∂x₁²(x₀⁴ + x₁²) = 2
        let result_x1 = d2fkk(&x0, 1, &[]);
        assert_eq!(result_x1[0], 2.0);
    }
}
